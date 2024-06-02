import torch

from dep_utils import JSD
from collections import defaultdict

DISC = 0.5
js_loss = JSD()

class IncParseField(object):
    '''
    Field for incremental parses
    '''
    def __init__(self, example):
        self.source = example['source']
        self.stimulus = example['stimulus']
        self.baseline = example['baseline']
        self.disamb_pos_ambig = example['disamb_pos_ambig']
        self.disamb_pos_base = example['disamb_pos_base']
        self.amb_pos_ambig = example['amb_pos_ambig']
        self.amb_pos_base = example['amb_pos_base']
        self.orig_idx = example['orig_idx']
        self.np_ambig = example['np_ambig']
        self.np_base = example['np_base']
        self.tokenized_stimulus = example['tokenized_stimulus']
        self.tokenized_baseline = example['tokenized_baseline']

        self.parses = example['parses']        
        self.inc_charts = defaultdict(dict)
        self.additions = None

    def get_chart(self, sent_type, mode, remove_additions=False):
        chart = self.inc_charts[sent_type][mode]
        if remove_additions:
            chart = self.delete_row_col(chart)
        return chart
    
    def get_diff_chart(self, mode):
        if self.source in ['classic-nps', 'classic-mvrr']:
            stimulus_chart = self.get_chart('stimulus', mode)
            baseline_chart = self.get_chart('baseline', mode, remove_additions=True)
        elif self.source == 'nnc':
            stimulus_chart = self.get_chart('stimulus', mode, remove_additions=True)
            baseline_chart = self.get_chart('baseline', mode)
        return stimulus_chart - baseline_chart

    def delete_row_col(self, array):
        valid = [r for r in range(len(self.tokenized_baseline)) if r not in self.additions]
        array = array [valid][:, valid] 
        return array

    def detect_added_positions(self):
        added = []
        enumerated = list(enumerate(self.tokenized_baseline))
        for position, token in enumerate(self.tokenized_stimulus):
            while token != enumerated[position][1]:
                added.append(enumerated[position][0])
                enumerated.pop(position)
        self.additions = added

    def compute_charts(self):
        for sent_type in ['stimulus', 'baseline']:
            parse = self.parses[sent_type]
            self.inc_charts[sent_type]['div_last'] = self.last_div(parse)
            self.inc_charts[sent_type]['div_first'] = self.first_div(parse)
            self.inc_charts[sent_type]['div_prev'] = self.prev_div(parse)

    def last_div(self, parse):
        '''
        Compute JS divergence w.r.t. final timestep.
        '''
        token_len = len(parse['rel_attn'])
        ref = parse['s_rel'][-1]
        ref_arc_idx = parse['val_arc_parse'][-1]

        div_tensor = torch.full((token_len, token_len), float('Inf'))

        for step in range(token_len):
            uniform = torch.full((token_len, ref.size(-1)), 1/ref.size(-1))
            out = parse['rel_attn'][step]
            curr_arc_idx = parse['val_arc_parse'][step]

            # distribution of current dep labels in the final timestep.
            target_1 = ref[torch.arange(curr_arc_idx.size(-1)), curr_arc_idx, :].softmax(-1)[1:, :]

            div_1 = js_loss(out[:step+1, :], target_1[:step+1, :])
            div_1 = torch.clamp(div_1, min=0) 

            # Mask where arc is different to the reference, but token can already be observed.
            mask_1 = ref_arc_idx[1:step+2] < curr_arc_idx[1:]
            idx_tensor = torch.full_like(curr_arc_idx[1:], len(curr_arc_idx[1:]))
            highest_arc = torch.maximum(ref_arc_idx[1:step+2], curr_arc_idx[1:])
            mask_2 = highest_arc < idx_tensor
            mask = torch.logical_or(mask_1, mask_2)

            if torch.any(mask):
                # Patch the uniform dist tensor if information about the dep labels are available
                mask_idx = ref_arc_idx[1:step+2][mask]
                start_idx = (mask == True).nonzero(as_tuple=False).squeeze(0).reshape(1,-1)
                patch_vec = parse['s_rel'][step][start_idx+1, mask_idx, :].softmax(-1)
                uniform[start_idx, :] = patch_vec

            target_2 = parse['rel_attn'][-1]

            div_2 = js_loss(uniform[:step+1, :], target_2[:step+1, :])
            div_2 = torch.clamp(div_2, min=0)

            # Compute the JS divergence in two directions
            arc_mask = (curr_arc_idx[1:] != ref_arc_idx[1:step+2]).unsqueeze(1)
            weight = torch.ones_like(div_1)
            weight[arc_mask] = DISC
            weighted_div = weight * div_1 + (1 - weight) * div_2
            div_tensor[step, :step+1] = weighted_div.view(1, -1)

        return div_tensor

    def first_div(self, parse):
        '''
        Compute JS divergence w.r.t. first timestep.
        '''
        token_len = len(parse['rel_attn'])
        ref = torch.cat([rel[-1, :].unsqueeze(0) for rel in parse['rel_attn']])

        ref_arc_idx = []
        for i, arc in enumerate(parse['val_arc_parse']):
            if i == 0:
                ref_arc_idx.append(arc)
            else:
                ref_arc_idx.append(arc[-1].unsqueeze(0))
        ref_arc_idx = torch.cat(ref_arc_idx)

        ref_srel = [s_rel[-1, :, :] for s_rel in parse['s_rel']]
        div_tensor = torch.full((token_len, token_len), float('Inf'))
        idx_mask = torch.arange(1, token_len+1)

        for step in range(token_len):
            uniform = torch.full((token_len, ref.size(-1)), 1/ref.size(-1))
            curr_arc_idx = parse['val_arc_parse'][step]

            # distribution of first dep labels in the current timestep.
            out = parse['s_rel'][step][torch.arange(curr_arc_idx.size(-1)), ref_arc_idx[:step+2], :].softmax(-1)[1:, :]

            div_1 = js_loss(out[:step+1, :], ref[:step+1, :])
            div_1 = torch.clamp(div_1, min=0)

            # Mask where arc is different to the reference, but token can already be observed
            curr_idx_mask = curr_arc_idx[1:] < idx_mask[:step+1]
            ref_idx_mask = ref_arc_idx[1:step+2] < idx_mask[:step+1]
            mask = torch.logical_and(curr_idx_mask, ref_idx_mask)

            if torch.any(mask):
                # Patch the uniform dist tensor if information about the dep labels are available
                start_idx = (mask == True).nonzero(as_tuple=False).squeeze(0)
                for id in start_idx:
                    patch_vec = ref_srel[id.item()][curr_arc_idx[id.item()+1], :].softmax(-1).unsqueeze(0)
                    uniform[id.item(), :] = patch_vec

            out_2 = parse['rel_attn'][step]

            div_2 = js_loss(out_2[:step+1, :], uniform[:step+1, :])
            div_2 = torch.clamp(div_2, min=0)

            # Compute the JS divergence in two directions
            arc_mask = (curr_arc_idx[1:] != ref_arc_idx[1:step+2]).unsqueeze(1)
            weight = torch.ones_like(div_1)
            weight[arc_mask] = DISC
            weighted_div = weight * div_1 + (1 - weight) * div_2
            div_tensor[step, :step+1] = weighted_div.view(1, -1)

        return div_tensor

    def prev_div(self, parse):
        '''
        Compute JS divergence w.r.t. previous timestep.
        '''
        token_len = len(parse['rel_attn'])
        div_tensor = torch.full((token_len, token_len), float('Inf'))
        idx_mask = torch.arange(1, token_len+1)

        for step in range(token_len):
            if step == 0:
                # We assume that at the previous timestep, the distribution is uniform
                out = parse['rel_attn'][step]
                ref = torch.full((1, out.size(-1)), 1/out.size(-1))

                div = torch.clamp(
                    js_loss(out, ref), min=0
                )
                div_tensor[step, :step+1] = div.view(1, -1)

            else:
                uniform = torch.full((token_len, parse['rel_attn'][step].size(-1)), 1/parse['rel_attn'][step].size(-1))
                ref_srel = parse['s_rel'][step-1]
                ref_arc_idx = parse['val_arc_parse'][step-1]
                ref_arc_idx = torch.cat((ref_arc_idx, torch.Tensor([-1]).to(torch.int))) # placeholder for arc at the current step
                curr_arc_idx = parse['val_arc_parse'][step]

                # Exclude token at the current step as it has no reference (assume uniform later)
                # distribution of previous dep labels in the current timestep.
                out = parse['s_rel'][step][torch.arange(curr_arc_idx.size(-1)), ref_arc_idx, :].softmax(-1)[1:-1, :]
                div_1 = js_loss(out, parse['rel_attn'][step-1])
                div_1 = torch.clamp(div_1, min=0)

                # Mask where arc is different to the reference, but token can already be observed
                curr_idx_mask = curr_arc_idx[1:-1] < idx_mask[:step]
                ref_idx_mask = ref_arc_idx[1:-1] < idx_mask[:step]
                mask_1 = torch.logical_and(curr_idx_mask, ref_idx_mask)

                # Mask to the right of the current token, where information about right tokens can already be retrieved
                idx_tensor = torch.full_like(curr_arc_idx[1:-1], len(curr_arc_idx[1:-1]))
                highest_arc = torch.maximum(ref_arc_idx[1:-1], curr_arc_idx[1:-1])
                mask_2 = highest_arc <= idx_tensor

                mask = torch.logical_or(mask_1, mask_2)

                if torch.any(mask):
                    # Patch the uniform dist tensor if information about the dep labels are available
                    start_idx = (mask == True).nonzero(as_tuple=False).squeeze(0)
                    for id in start_idx:
                        patch_vec = ref_srel[id.item()+1][curr_arc_idx[id.item()+1], :].softmax(-1).unsqueeze(0)
                        uniform[id.item(), :] = patch_vec

                # Exclude token at the current step as it has no reference (assume uniform later)
                out_2 = parse['rel_attn'][step][:-1, :]
                div_2 = js_loss(out_2, uniform[:step,:])
                div_2 = torch.clamp(div_2, min=0)

                # Compute the JS divergence in two directions
                arc_mask = (curr_arc_idx[1:-1] != ref_arc_idx[1:-1]).unsqueeze(1)
                weight = torch.ones_like(div_1)
                weight[arc_mask] = DISC
                weighted_div = weight * div_1 + (1 - weight) * div_2

                # Compute the JS divergence of the current token and stack it with the previous ones.
                uniform_buf = torch.full((1, out_2.size(-1)), 1/out_2.size(-1))
                curr_out = parse['rel_attn'][step][-1, :].unsqueeze(0)
                div_curr = js_loss(curr_out, uniform_buf)
                div_curr = torch.clamp(div_curr, min=0)
                weighted_div = torch.cat((weighted_div, div_curr))
                div_tensor[step, :step+1] = weighted_div.view(1, -1)

        return div_tensor