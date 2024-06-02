#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A class to represent the incremental states of a sentence for all layers.
"""

from typing import Callable, List, Optional, Union, Tuple

import numpy as np
import torch

from sklearn.metrics.pairwise import paired_distances
from utils import tokenize

MODES = ('first', 'last', 'previous')
DEFAULT_VALUE = 0


def distance(*x: np.array) -> np.array:
    """Define a default distance function."""
    return paired_distances(*x, metric='cosine')


class IncStates:
    """Structure for restart-incremental states."""
    def __init__(self,
                 sentence: str,
                 additions: Optional[List[int]] = None,
                 tokenized: bool = False):
        self.sentence = sentence.split() if tokenized else tokenize(sentence)
        self.additions = additions
        self.n_layers: Optional[int] = None
        self.emb_dim: Optional[int] = None
        self.states: Optional[Union[np.array, torch.tensor]] = None

    @property
    def n_tokens(self) -> int:
        """Return number of tokens in the sentence."""
        return len(self.sentence)

    def _empty_chart(self, dims: List[int], tensor: bool = False):
        """Return an empty array with given dimensions."""
        if tensor:
            return torch.ones(*dims) * torch.nan
        return np.ones(dims) * np.nan


class IncStatesAnalyser(IncStates):
    """Structure for restart-incremental states with metrics."""
    def __init__(self,
                 sentence: str,
                 modes: Tuple[str] = MODES,
                 additions: Optional[List[int]] = None,
                 states: Optional[np.array] = None,
                 tokenized: bool = False):
        super().__init__(sentence, additions, tokenized)
        self.states = states
        self.n_layers, _, _, self.emb_dim = states.shape
        self.inc_charts: Optional[np.array] = None
        self.dist_func: Optional[Callable] = None
        self.modes = modes

    def shorten_sentence(self) -> List[str]:
        """Return sentence without the extra added tokens."""
        return [token for idx, token in enumerate(self.sentence)
                if idx not in self.additions]

    def _delete_rows_cols(self, chart: np.array) -> np.array:
        """Remove rows and columns in states corresponding to added tokens."""
        # https://stackoverflow.com/a/63564120
        valid = [r for r in range(self.n_tokens) if r not in self.additions]
        shortened_chart = chart[valid][:, valid]
        return shortened_chart

    def get_layer_state(self, layer: int,
                        remove_additions: bool = False) -> np.array:
        """Return states for a ayer, possibly without extra tokens."""
        chart = self.states[layer]
        if remove_additions:
            chart = self._delete_rows_cols(chart)
        return chart

    def get_mode_layer_chart(self, mode: str, layer: int,
                             remove_additions: bool = False) -> np.array:
        """Return chart for a mode/layer, possibly without extra tokens"""
        chart = self.inc_charts[mode][layer]
        if remove_additions:
            chart = self._delete_rows_cols(chart)
        return chart

    def _get_metric(self, states: np.array, token: int, step: int,
                    mode: str) -> float:
        """Return a metric for one cell in the chart."""
        current = [states[step, token]]
        if mode == 'first':
            ref = [states[token, token]]
        elif mode == 'last':
            ref = [states[-1, token]]
        elif mode == 'previous':
            if step == token:
                return DEFAULT_VALUE
            ref = [states[step - 1, token]]
        else:
            raise NotImplementedError
        sim = self.dist_func(ref, current).item()
        return sim

    def _build_distance_chart(self, states: np.array, mode: str) -> np.array:
        """Return distance chart computed for a given mode and distance."""
        dist_chart = self._empty_chart([self.n_tokens, self.n_tokens])
        for token in range(self.n_tokens):
            for step in range(token, self.n_tokens):
                dists = self._get_metric(states, token, step, mode)
                dist_chart[step, token] = dists
        return dist_chart

    def compute_distance_charts(self, dist_func: Callable = distance):
        """Fill in the incremental charts with computed distances."""
        self.dist_func = dist_func
        dims = self.n_layers, self.n_tokens, self.n_tokens
        self.inc_charts = {m: self._empty_chart(dims) for m in self.modes}

        for mode in self.modes:
            for layer, states in enumerate(self.states):
                chart = self._build_distance_chart(states, mode)
                self.inc_charts[mode][layer] = chart


class IncStatesCreator(IncStates):
    """Structure for extracting restart-incremental states from a model."""
    def __init__(self,
                 sentence: str,
                 model: str,
                 additions: Optional[List[int]] = None,
                 tokenized: bool = False,):
        super().__init__(sentence, additions, tokenized)
        assert model in ['bert', 'roberta', 'gpt2', 'opt']
        self.model = model

    def _set_dims(self, tokenizer, model):
        """Set own dimensions given the model."""
        # get number of layers and embedding dim dinamically
        encoded_input = tokenizer(" ".join(self.sentence), return_tensors='pt')
        output = model(**encoded_input, output_hidden_states=True)

        self.n_layers = len(output['hidden_states'])
        self.emb_dim = output['hidden_states'][0].shape[-1]

    def _sanity_check_encoding(self, encoded_input: torch.tensor, tokenizer,
                               timestep: int):
        """Ensure that subtokenization does not happen."""
        token_ids = encoded_input['input_ids'][0]
        if self.model in ['bert', 'roberta']:
            # check that prefix length matches
            # - 2 because of initial CLS and final SEP added tokens
            assert token_ids.shape[0] - 2 == timestep + 1
            # check that initial and first token can be discarded
            assert token_ids[0] == tokenizer.cls_token_id
            assert token_ids[-1] == tokenizer.sep_token_id
            untokenized = tokenizer.convert_ids_to_tokens(token_ids)[1:-1]
        elif self.model in ['gpt2']:
            untokenized = tokenizer.convert_ids_to_tokens(token_ids)
        elif self.model in ['opt']:
            assert token_ids[0] == tokenizer.bos_token_id
            assert token_ids.shape[0] - 1 == timestep + 1
            untokenized = tokenizer.convert_ids_to_tokens(token_ids)[1:]
        else:
            raise NotImplementedError
        # check the actual created tokens are fully formed as in original
        lowercase_tokens = [w.lower() for w in self.sentence[:timestep + 1]]
        untokenized = [w.replace('Ġ', '').lower().strip() for w in untokenized]
        assert untokenized == lowercase_tokens

    def build_states(self, tokenizer, model):
        """Extract incremental states from a model."""

        self._set_dims(tokenizer, model)
        dims = self.n_layers, self.n_tokens, self.n_tokens, self.emb_dim
        self.states = self._empty_chart(dims, tensor=True)

        for timestep in range(0, self.n_tokens):
            prefix = " ".join(self.sentence[:timestep+1])

            encoded_input = tokenizer(prefix, return_tensors='pt')
            # make sure that no subtokenisation occurs
            self._sanity_check_encoding(encoded_input, tokenizer, timestep)
            output = model(**encoded_input, output_hidden_states=True)

            for layer in range(self.n_layers):
                # ignore batch dim, ignore start and end symbols
                out = output['hidden_states'][layer].squeeze(0)
                if self.model in ['bert', 'roberta']:
                    # bert and roberta have cls and sep tokens
                    out = out[1:-1]
                elif self.model == 'opt':
                    # opt has a </s> bos token
                    out = out[1:]
                self.states[layer][timestep][:timestep + 1] = out.detach()

        # loose check that the remaining nans are those in the upper part
        # of the tensor for all layers and features
        empty_cells = sum(i for i in range(1, self.n_tokens))
        n_nans = empty_cells * self.emb_dim * self.n_layers
        assert self.states.isnan().sum() == n_nans

        self.states = self.states.numpy()
