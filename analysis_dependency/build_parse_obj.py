import pandas as pd
import numpy as np
import argparse
import pdb
import os
import pickle

from dep_utils import split_sents
from collections import defaultdict
from tqdm import tqdm

DATA_PATH = '../analysis_meaning/outputs/preprocessed_stimuli.csv'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Build dependency parses for analyses with preprocessed stimuli'
    )

    parser.add_argument(
        '--model', dest='model',
        type=str, required=True
    )

    return parser.parse_args()


def main(args):
    if args.model in ['dep-biaffine-en', 'dep-biaffine-roberta-en']:
        from supar import Parser
    elif args.model in ['en_ewt.electra-base', 'en_ptb.electra-base']:
        from diaparser.parsers import Parser
    else:
        raise NotImplementedError
    
    parser = Parser.load(args.model)
    index2label = parser.REL.vocab

    df = pd.read_csv(DATA_PATH, index_col=0)
    outputs = defaultdict(dict)

    # Dump the data into a single pickle file
    for idx, row in tqdm(df.iterrows()):
        # Columns in csv
        outputs[idx]['source'] = row.source
        outputs[idx]['stimulus'] = row.stimulus
        outputs[idx]['baseline'] = row.baseline
        outputs[idx]['disamb_pos_ambig'] = int(row.disamb_position_ambiguous)
        outputs[idx]['disamb_pos_base'] = int(row.disamb_position_baseline)
        outputs[idx]['amb_pos_ambig'] = int(row.amb_position_ambiguous)
        outputs[idx]['amb_pos_base'] = int(row.amb_position_baseline)
        outputs[idx]['orig_idx'] = row.orig_idx # Can also be non integer for NV
        outputs[idx]['np_ambig'] = int(row.np_ambiguous) if not pd.isna(row.np_ambiguous) else None
        outputs[idx]['np_base'] = int(row.np_baseline) if not pd.isna(row.np_baseline) else None

        # Extract parses for stimulus and baseline
        outputs[idx]['tokenized_stimulus'] = split_sents(outputs[idx]['stimulus'])
        outputs[idx]['tokenized_baseline'] = split_sents(outputs[idx]['baseline'])
        
        outputs[idx]['parses'] = {sent_type: {} for sent_type in ['stimulus', 'baseline']}

        for sent_type in ['stimulus', 'baseline']:
            tokens = outputs[idx]['tokenized_' + sent_type]
            
            arc_preds = np.empty((len(tokens), len(tokens)))
            arc_preds.fill(np.inf)

            partial_sents_lst = []
            rel_preds_lst = []
            arc_attn_lst = []
            rel_attn_lst = []
            s_rel_lst = []
            val_arc_parse_lst = []

            for step in range(1, len(tokens)+1):
                partial_sents = tokens[:step]

                if args.model in ['dep-biaffine-en', 'dep-biaffine-roberta-en']:
                    partial_preds = parser.predict([partial_sents], prob=True)[0]
                    arc_attn = partial_preds.probs.cpu()
                    rel_attn = partial_preds.rel_attn[1:, :].cpu()
                    s_rel = partial_preds.s_rel.cpu()
                    valid_arc_parse = partial_preds.arc_preds.cpu()
                elif args.model in ['en_ewt.electra-base', 'en_ptb.electra-base']:
                    partial_preds = parser.predict([partial_sents], prob=True)
                    arc_attn = partial_preds.probs[0].cpu()
                    rel_attn = partial_preds.rel_attn[0][1:,:].cpu()
                    s_rel = partial_preds.s_rel[0].cpu()
                    valid_arc_parse = partial_preds.arc_preds[0].cpu()

                arc_preds[step-1][:step] = np.array(partial_preds.arcs, dtype=np.int8)

                rel_preds_id = rel_attn.argmax(-1).tolist()

                if args.model in ['dep-biaffine-en', 'dep-biaffine-roberta-en']:
                    assert partial_preds.rels == [index2label[idx] for idx in rel_preds_id], "Dependency relation prediction is not equal"
                    assert partial_preds.arcs == valid_arc_parse[1:].tolist(), "Arc prediction is not equal"
                elif args.model in ['en_ewt.electra-base', 'en_ptb.electra-base']:
                    assert partial_preds.rels[0] == [index2label[idx] for idx in rel_preds_id], "Dependency relation prediction is not equal"
                    assert partial_preds.arcs[0] == valid_arc_parse[1:].tolist(), "Arc prediction is not equal"

                # For arc_attn_lst and rel_attn_lst, axis 0 is timestep
                arc_attn_lst.append(arc_attn)
                rel_attn_lst.append(rel_attn)
                partial_sents_lst.append(partial_sents)
                rel_preds_lst.append(partial_preds.rels if args.model in ['dep-biaffine-en', 'dep-biaffine-roberta-en'] else partial_preds.rels[0])
                s_rel_lst.append(s_rel)
                val_arc_parse_lst.append(valid_arc_parse)
            
            outputs[idx]['parses'][sent_type]['partial_inputs'] = partial_sents_lst
            outputs[idx]['parses'][sent_type]['arc_preds'] = arc_preds
            outputs[idx]['parses'][sent_type]['rel_preds'] = rel_preds_lst
            outputs[idx]['parses'][sent_type]['arc_attn'] = arc_attn_lst
            outputs[idx]['parses'][sent_type]['rel_attn'] = rel_attn_lst
            outputs[idx]['parses'][sent_type]['s_rel'] = s_rel_lst
            outputs[idx]['parses'][sent_type]['val_arc_parse'] = val_arc_parse_lst

    file_name = 'preprocessed_' + args.model
    with open(os.path.join(os.getcwd(), 'outputs', file_name + '.pkl'), 'wb') as f:
        pickle.dump(outputs, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)