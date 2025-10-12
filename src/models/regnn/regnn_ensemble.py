import torch
from torch import nn

def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    count = torch.clamp(count, min=1)
    return result / count


class E_GCL(nn.Module):
    """
    Equivariant Graph Convolutional Layer for graph level tasks.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), 
                 residual=True, attention=False, normalize=False, coords_agg='mean', 
                 tanh=False, update_coords=False):  # Default to False
        super(E_GCL, self).__init__()
        
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        self.update_coords = update_coords  # Store the flag
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_nf*2+edges_in_d+1, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(input_nf+hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )
        
        # Only create coordinate MLP if we're updating coordinates
        if self.update_coords:
            layer = nn.Linear(hidden_nf, 1, bias=False)
            torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

            coord_mlp = []
            coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
            coord_mlp.append(act_fn)
            coord_mlp.append(layer)
            if self.tanh:
                coord_mlp.append(nn.Tanh())
            self.coord_mlp = nn.Sequential(*coord_mlp)
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        
        out = self.edge_mlp(out)
        
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        
        out = self.node_mlp(agg)
        
        if self.residual:
            out = x + out
        
        return out, agg

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
            
        return radial, coord_diff

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        
        # Add clipping for stability
        max_trans = 10.0
        trans = torch.clamp(trans, -max_trans, max_trans)
        
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter: %s' % self.coords_agg)
            
        # Additional clipping for stability
        agg = torch.clamp(agg, -max_trans, max_trans)
        coord = coord + agg
        
        return coord

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        
        # Only update coordinates if flag is True
        if self.update_coords:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_feat


class REGCLEnsemble(nn.Module):
    """
    Relational Equivariant GCL Ensemble.
    Handles multiple relation types in the graph.
    """
    def __init__(self, input_nf, hidden_nf, output_nf, num_relations=2, 
                 edges_in_d=0, act_fn=nn.SiLU(), node_agg='sum',
                 residual=True, attention=False, normalize=False, 
                 coords_agg='mean', tanh=False, update_coords=False):  # Default to False
        super(REGCLEnsemble, self).__init__()
        
        self.num_relations = num_relations
        self.node_agg = node_agg
        self.coords_agg = coords_agg
        self.update_coords = update_coords  # Store the flag
        
        # Create a separate E_GCL for each relation type
        self.e_gcl_layers = nn.ModuleList()
        for i in range(num_relations):
            gcl = E_GCL(
                input_nf=input_nf,
                output_nf=output_nf,
                hidden_nf=hidden_nf,
                edges_in_d=edges_in_d,
                act_fn=act_fn,
                residual=residual,
                attention=attention,
                normalize=normalize,
                coords_agg=coords_agg,
                tanh=tanh,
                update_coords=update_coords  # Pass the flag
            )
            self.e_gcl_layers.append(gcl)
        
        # If concatenating outputs from different relation types
        if node_agg == 'cat':
            self.combine_layer = nn.Linear(output_nf * num_relations, output_nf)
    
    def forward(self, h, edge_index, coord, edge_attr=None, edge_type=None, node_attr=None):
        # Process each relation type separately
        h_outputs = []
        coord_outputs = []
        active_relations = 0
        
        for rel_type in range(self.num_relations):
            # Filter edges of this relation type
            if edge_type is None:
                rel_edge_index = edge_index
                rel_edge_attr = edge_attr
            else:
                rel_mask = edge_type == rel_type
                if not rel_mask.any():
                    continue
                
                active_relations += 1
                rel_edge_index = edge_index[:, rel_mask]
                rel_edge_attr = edge_attr[rel_mask] if edge_attr is not None else None
            
            # Process through the relation-specific E_GCL
            rel_h, rel_coord, _ = self.e_gcl_layers[rel_type](
                h, rel_edge_index, coord, rel_edge_attr, node_attr
            )
            
            # Store outputs
            h_outputs.append(rel_h)
            
            # Only store coordinate updates if we're updating coordinates
            if self.update_coords:
                coord_outputs.append(rel_coord - coord)
        
        # Handle empty case (no relations found)
        if not h_outputs:
            return h, coord
        
        # Combine node feature outputs
        if self.node_agg == 'cat':
            if len(h_outputs) == 1:
                h_out = h_outputs[0]
            else:
                h_combined = torch.cat(h_outputs, dim=-1)
                h_out = self.combine_layer(h_combined)
        elif self.node_agg == 'sum':
            h_out = sum(h_outputs)
        else:  # 'mean'
            if len(h_outputs) == 1:
                h_out = h_outputs[0]
            else:
                h_stacked = torch.stack(h_outputs, dim=0)
                h_out = h_stacked.mean(dim=0)
        
        # Only update coordinates if the flag is set
        if self.update_coords and coord_outputs:
            if len(coord_outputs) == 1:
                coord_out = coord + coord_outputs[0]
            elif self.coords_agg == 'sum':
                coord_out = coord + sum(coord_outputs)
            else:  # 'mean'
                coord_stacked = torch.stack(coord_outputs, dim=0)
                coord_out = coord + coord_stacked.mean(dim=0)
                
            # Handle NaN and large values
            coord_out = torch.where(torch.isnan(coord_out), coord, coord_out)
            max_shift = 10.0
            max_diff = torch.clamp(coord_out - coord, -max_shift, max_shift)
            coord_out = coord + max_diff
        else:
            # Return original coordinates unchanged
            coord_out = coord
        
        return h_out, coord_out