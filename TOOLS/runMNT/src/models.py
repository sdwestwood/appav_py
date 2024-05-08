import torch
from torch import nn
import torch.nn.functional as F

from modules.transformer import TransformerEncoder
from itertools import permutations

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        if hyp_params.modality_count > 2:
            self.modalities = ['l', 'a', 'v', 'h'][:hyp_params.modality_count] # v is always the feature matrix, and h is optional (lableed eeg in dataset for now)
        elif hyp_params.modality_count == 2: # just testing if using pupil and HR yeilds better accuracy
            self.modalities = ['a', 'v'] 
        hyp_params = vars(hyp_params)
        shared_dim = hyp_params['shared_dim']
        self.partial_mode = 0
        combined_dim = 0
        for mod in self.modalities:
            setattr(self, f"orig_d_{mod}", hyp_params['orig_d_'+mod])
            
            setattr(self, f"d_{mod}", shared_dim)
            combined_dim += shared_dim
            
            setattr(self, f"{mod}only", hyp_params[f"{mod}only"])
            if hyp_params[f"{mod}only"]:
                self.partial_mode += 1
            setattr(self, f"attn_dropout_{mod}", hyp_params[f"attn_dropout_{mod}"])
        self.num_heads = hyp_params['num_heads']
        self.layers = hyp_params['layers']
        self.relu_dropout = hyp_params['relu_dropout']
        self.res_dropout = hyp_params['res_dropout']
        self.out_dropout = hyp_params['out_dropout']
        self.embed_dropout = hyp_params['embed_dropout']
        self.attn_mask = hyp_params['attn_mask']

        if self.partial_mode == 1:
            combined_dim = (len(self.modalities)-1) * self.d_l   # assuming d_l == d_a == d_v == d_h
        else:
            combined_dim = (len(self.modalities)-1) * combined_dim
        
        output_dim = hyp_params['output_dim']        # This is actually not a hyperparameter :-)
        raw_kernel_size = hyp_params['raw_kernel_size']
        raw_stride = hyp_params['raw_stride']
        feat_kernel_size = hyp_params['feat_kernel_size']
        feat_stride = hyp_params['feat_stride']
        # 1. Temporal convolutional layers
        if 'l' in self.modalities:
            self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=raw_kernel_size, stride = raw_stride, padding=int(feat_kernel_size/2), bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=raw_kernel_size, stride = raw_stride, padding=int(feat_kernel_size/2), bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=feat_kernel_size, stride = feat_stride,padding=0, bias=False)
        if 'h' in self.modalities:
            self.proj_h = nn.Conv1d(self.orig_d_h, self.d_h, kernel_size=raw_kernel_size, stride = raw_stride, padding=int(feat_kernel_size/2), bias=False)
        # 2. Crossmodal Attentions
        if 'l' in self.modalities and self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
            if 'h' in self.modalities:
                self.trans_l_with_h = self.get_network(self_type='lh')
        if self.aonly:
            if 'l' in self.modalities:
                self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
            if 'h' in self.modalities:
                self.trans_a_with_h = self.get_network(self_type='ah')
        if self.vonly:
            if 'l' in self.modalities:
                self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
            if 'h' in self.modalities:
                self.trans_v_with_h = self.get_network(self_type='vh')
        if 'h' in self.modalities and self.honly:
            self.trans_h_with_v = self.get_network(self_type='hv')
            self.trans_h_with_l = self.get_network(self_type='hl')
            self.trans_h_with_a = self.get_network(self_type='ha')
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        if 'l' in self.modalities:
            self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        if 'h' in self.modalities:
            self.trans_h_mem = self.get_network(self_type='h_mem', layers=3)
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        # self.lstm = nn.LSTM(self.d_l, combined_dim)
        self.out_layer = nn.Linear(combined_dim, 1)
        #self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type.endswith('l'):
            embed_dim, attn_dropout = self.d_l, self.attn_dropout_l
        elif self_type.endswith('a'):
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type.endswith('v'):
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type.endswith('h'):
            embed_dim, attn_dropout = self.d_h, self.attn_dropout_h
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = (len(self.modalities)-1)*self.d_l, self.attn_dropout_l
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = (len(self.modalities)-1)*self.d_a, self.attn_dropout_a
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = (len(self.modalities)-1)*self.d_v, self.attn_dropout_v
        elif self_type == 'h_mem':
            embed_dim, attn_dropout = (len(self.modalities)-1)*self.d_h, self.attn_dropout_l
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def get_last_output(self, proj_x, child_mods, mod='l'):
        h = []
        for cm in child_mods:
            h.append(getattr(self,f"trans_{mod}_with_{cm}")(proj_x, child_mods[cm], child_mods[cm]))    # Dimension (L, N, d_l)
        h_ls = torch.cat(h, dim=2)
        h_ls = getattr(self,f"trans_{mod}_mem")(h_ls)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
        return last_h_l
    
    def forward(self, x_l, x_a, x_v=None, x_h=None):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        if not torch.is_tensor(x_v) and not torch.is_tensor(x_h): # we assume one time series and one feature matrix at a minimum, so re-label the two variables that exist, forward fn is dumb
            x_v = x_a.clone().detach()
            x_a = x_l.clone().detach()
            x_l = None
        if torch.is_tensor(x_l):
            x_l = x_l.transpose(1, 2)
            proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
            proj_x_l = proj_x_l.permute(2, 0, 1)
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        if torch.is_tensor(x_h):
            x_h = x_h.transpose(1, 2)
            proj_x_h = x_h if self.orig_d_h == self.d_h else self.proj_h(x_h)
            proj_x_h = proj_x_h.permute(2, 0, 1)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        
        output_proj_dict = {'a': proj_x_a, 'v': proj_x_v}
        if torch.is_tensor(x_l):
            output_proj_dict['l'] = proj_x_l
        if torch.is_tensor(x_h):
            output_proj_dict['h'] = proj_x_h
            
        if torch.is_tensor(x_l) and self.lonly:
            last_h_l = self.get_last_output(proj_x_l, {i:output_proj_dict[i] for i in output_proj_dict if i!='l'}, mod='l')
                
        if self.aonly:
            last_h_a = self.get_last_output(proj_x_a, {i:output_proj_dict[i] for i in output_proj_dict if i!='a'}, mod='a')
                
        if self.vonly:
            last_h_v = self.get_last_output(proj_x_v, {i:output_proj_dict[i] for i in output_proj_dict if i!='v'}, mod='v')
        
        if torch.is_tensor(x_h) and self.honly:
            last_h_h = self.get_last_output(proj_x_h, {i:output_proj_dict[i] for i in output_proj_dict if i!='h'}, mod='h')
        
        if self.partial_mode == 4:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_h_h], dim=1)
        elif self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)
        elif self.partial_mode == 2:
            last_hs = torch.cat([last_h_a, last_h_v], dim=1)
        # A residual block
        # tmp = F.dropout(self.proj1(last_hs), p=self.out_dropout, training=self.training)
        last_hs_proj = self.proj2(F.dropout(self.proj1(last_hs), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)
        return output, last_hs
