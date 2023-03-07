from argparse import ArgumentParser
import numpy as np
import sys

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cosine

import torch

from transformers import MarianMTModel, MarianTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration

import networkx as nx
import matplotlib.pyplot as plt

def integerize(val: float, power: int=1000) -> int:
    return int(val*power)


def get_dist_mtrx(emb_array: torch.Tensor) -> torch.Tensor:
    '''returns an upper-triangle distance matrix'''
    return torch.Tensor([[integerize(cosine(emb1, emb2)) if idx1 >= idx2 else 0 for idx1, emb1 in enumerate(emb_array)] for idx2, emb2 in enumerate(emb_array)])


def generate_mst(input_ids, tokenizer, hidden_states) -> None:

    input_tokens = [tokenizer.decode(i) for i in input_ids[-1]]
    input_matrix = csr_matrix(get_dist_mtrx(hidden_states[-1][-1]))

    mst = minimum_spanning_tree(input_matrix)
    rows, cols = mst.nonzero()
    values = mst[rows, cols].tolist()[-1]
    input_edges = [(input_tokens[i], input_tokens[j], val) for i, j, val in zip(rows, cols, values)]

    # Define the graph
    G = nx.Graph()
    G.add_nodes_from(input_tokens)
    G.add_weighted_edges_from(input_edges)

    # Compute the minimum spanning tree
    T = nx.minimum_spanning_tree(G)

    # Visualize the graph and the minimum spanning tree
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edges(T, pos, edge_color='r', width=2)
    plt.savefig("mst_tree.png")


def main(args):

    if args.input_sentence:
        input_sentence = args.input_sentence
    else:
        input_sentence = 'I am thinking about "Lucy in the sky with diamonds", which was sung by the Beatles.'

    model_name = "Helsinki-NLP/opus-mt-en-fr"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # model_name = "t5-small"
    # tokenizer = T5Tokenizer.from_pretrained(model_name)
    # model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer.encode(input_sentence, return_tensors='pt')
    output = model.generate(input_ids, return_dict_in_generate=True, output_attentions=True, output_hidden_states=True)

    encoder_hidden_states = output.encoder_hidden_states
    decoder_hidden_states = output.decoder_hidden_states

    # the dimension is "7" because, 1 embedding layer + 6 layers
    print('Encoder hidden states shape:', len(encoder_hidden_states), encoder_hidden_states[-1].shape)
    print('Decoder hidden states shape:', len(decoder_hidden_states), len(decoder_hidden_states[-1]), decoder_hidden_states[-1][-1].shape)

    output = model.generate(input_ids)
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in output]
    print(tgt_text)

    generate_mst(input_ids, tokenizer, encoder_hidden_states)

    # TODO: Visualize decoder space. Dimensions are different from encoder's


if __name__ == "__main__":
    argp = ArgumentParser()
    # argp.add_argument('experiment_config')
    argp.add_argument('--input-sentence', default='')
    # argp.add_argument('--seed', default=0, type=int,
    #     help='sets all random seeds for (within-machine) reproducibility')
    args = argp.parse_args()

    sys.exit(main(args))