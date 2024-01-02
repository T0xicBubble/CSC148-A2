"""
Assignment 2 starter code
CSC148, Winter 2023

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for item in text:
        if item in freq_dict:
            freq_dict[item] += 1
        else:
            freq_dict[item] = 1
    return freq_dict


def _sort_frequency_dict(freq_dict: dict[int, int]) -> dict[int, int]:
    """Swaps the keys and values of a dictionary and then sorts the
    dictionary based on values of original dictionary.
    """
    # Swap the keys and values so it can be sorted
    swapped_dict = {}
    for key in freq_dict:
        value = freq_dict[key]
        if value in swapped_dict:
            swapped_dict[value].append(key)
        else:
            swapped_dict[value] = [key]

    # Sort the swapped list
    sorted_dict = {}
    sorted_keys = sorted(swapped_dict)
    for key in sorted_keys:
        sorted_dict[key] = swapped_dict[key]
    return sorted_dict


def _sort_huffman(freq_lst: list, huff_lst: list) -> None:
    """Sorts the last element of a list into its correct place

    Precondition: freq_lst and huffman_lst are the same length
     and both have at least 1 value

    Annoying Pyta style conventions made me change parameter name from
    "huffman_lst" to "huff_lst"
    >>> x = [2, 4, 5, 6, 3]
    >>> y = ['a', 'c', 'd', 'e', 'b']
    >>> _sort_huffman(x, y)
    >>> x
    [2, 3, 4, 5, 6]
    >>> y
    ['a', 'b', 'c', 'd', 'e']
    >>> x = [3]
    >>> y = ['a']
    >>> _sort_huffman(x, y)
    >>> x
    [3]
    >>> y
    ['a']
    """
    end_value = freq_lst[-1]
    for i in range(len(freq_lst) - 1, -1, -1):
        if freq_lst[i] > end_value:
            freq_lst[i + 1], freq_lst[i] = freq_lst[i], freq_lst[i + 1]
            # Forced to change var name cause of line below, too long or
            # wrong indent
            huff_lst[i + 1], huff_lst[i] = huff_lst[i], huff_lst[i + 1]


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    # Create a list with frequencies in increasing order
    # Example Freq Dict {65: 1, 66: 2, 67: 1}
    sorted_dict = _sort_frequency_dict(freq_dict)  # {1: [65, 67], 2: [66]}
    # Turn values into frequencies
    freq_list = [i for i in sorted_dict for j in range(len(sorted_dict[i]))]
    huffman_list = []
    # Turn all values into huffman trees
    for freq in sorted_dict:
        for value in sorted_dict[freq]:
            huffman_list.append(HuffmanTree(value))

    # Catch case when we only have 1 value
    if len(huffman_list) == 1:
        # Return a tree with a single left node
        return HuffmanTree(None, huffman_list[0], HuffmanTree(80085))
    # Make tree using method for level traversal, Q method
    # 1. Take out 2 lowest values
    # 2. Put in sum of two values
    # 3. Sort that new value into correct place
    # 4. Repeat until only 1 value in both lists
    while len(huffman_list) > 1:
        first_freq = freq_list.pop(0)
        second_freq = freq_list.pop(0)
        # The lower the index the lower the frequency
        left_tree = huffman_list.pop(0)
        right_tree = huffman_list.pop(0)
        freq_list.append(first_freq + second_freq)
        huffman_list.append(HuffmanTree(None, left_tree, right_tree))
        _sort_huffman(freq_list, huffman_list)
        # TO SORT LIST EVERYTIME, USE INSERTION SORT
        # ON FREQ LIST AND COPY ACTIONS MADE ONTO HUFFMAN LIST
    return huffman_list[0]


def _create_codes(tree: HuffmanTree, code_dict: dict, code: str = "") -> None:
    """
    Given a tree and dictionary, map all the leafs to codes
    Left = 0
    Right = 1
    """
    if tree.is_leaf():
        code_dict[tree.symbol] = code
    else:
        # Check for 1 value case
        _create_codes(tree.left, code_dict, code + "0")
        _create_codes(tree.right, code_dict, code + "1")


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> get_codes(HuffmanTree(None, HuffmanTree(1, None, None), None))
    {1: '0'}
    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(None, HuffmanTree(5), HuffmanTree(6)))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 5: "10", 6: "11"}
    True
    """
    code_key = {}
    # Check if tree only has one value
    if tree.right is None:
        return {tree.left.symbol: "0"}
    _create_codes(tree, code_key)
    return code_key


def _trees_in_postorder(tree: HuffmanTree) -> list[HuffmanTree]:
    """ Makes a list of huffman trees in postorder excluding the leafs

    """
    lst_of_nodes = []
    if tree.is_leaf():
        return []
    else:
        # Check for special case
        if tree.right is None:
            return [tree]
        left_tree = _trees_in_postorder(tree.left)
        right_tree = _trees_in_postorder(tree.right)
        lst_of_nodes.extend(left_tree)
        lst_of_nodes.extend(right_tree)
        lst_of_nodes.extend([tree])
    return lst_of_nodes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # Use postorder traversal and then use the list generated to number trees
    trees = _trees_in_postorder(tree)
    count = 0
    for a_tree in trees:
        a_tree.number = count
        count += 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    code_dict = get_codes(tree)
    total_weight = 0
    total_freq = 0
    for value in freq_dict:
        total_weight += freq_dict[value] * len(code_dict[value])
        total_freq += freq_dict[value]
    return total_weight / total_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    # Make a giant string with all the codes added up
    str_repr = ""
    byte_repr = bytearray()
    for byte in text:
        str_repr += codes[byte]
        while len(str_repr) > 8:
            sliced_str = str_repr[:8]
            str_repr = str_repr[8:]
            byte_repr.append(bits_to_byte(sliced_str))
    if len(str_repr) > 0:
        byte_repr.append(bits_to_byte(str_repr))
    return bytes(byte_repr)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    # [a, b, c, d]
    # a = 0 if left tree is leaf, else 1
    # b = tree.symbol of tree.left if is leaf,
    # otherwise tree.number of tree.left
    # c = 0 if right tree is leaf, else 1
    # d = tree.symbol of tree.right if is leaf,
    # otherwise tree.number of tree.right
    trees = _trees_in_postorder(tree)
    # Represent each tree as [a, b, c, d]
    byte_repr = bytearray()
    a = 0
    b = 0
    c = 0
    d = 0
    for curr_tree in trees:
        # Is left a leaf
        if curr_tree.left.is_leaf():
            a = 0
            b = curr_tree.left.symbol
        else:
            a = 1
            b = curr_tree.left.number
        # Is right a leaf
        if curr_tree.right.is_leaf():
            c = 0
            d = curr_tree.right.symbol
        else:
            c = 1
            d = curr_tree.right.number
        byte_repr.append(a)
        byte_repr.append(b)
        byte_repr.append(c)
        byte_repr.append(d)
    return bytes(byte_repr)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = [ReadNode(0, 10, 0, 12), ReadNode(0, 5, 0, 7),  \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)))
    >>> lst = [ReadNode(0, 10, 1, 0)]
    >>> generate_tree_general(lst, 0)
    HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(80085, None, None))
    >>> lst = [ ReadNode(1, 1, 1, 2), ReadNode(0, 10, 0, 12), \
    ReadNode(0, 5, 0, 7)]
    >>> generate_tree_general(lst, 0)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    >>> lst = [ ReadNode(0, 3, 1, 3), ReadNode(1, 2, 1, 0), \
    ReadNode(0, 1, 0, 2), ReadNode(0, 4, 0 ,5)]
    >>> generate_tree_general(lst, 1)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(1, None, None), \
HuffmanTree(2, None, None)), \
HuffmanTree(None, HuffmanTree(3, None, None), \
HuffmanTree(None, HuffmanTree(4, None, None), \
HuffmanTree(5, None, None))))
    >>> lst = [ReadNode(0, 10, 0, 20), ReadNode(1, 0, 0, 30)]
    >>> generate_tree_general(lst, 1)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(20, None, None)), HuffmanTree(30, None, None))
    >>> lst = [ReadNode(0, 10, 0, 20), ReadNode(0, 30, 1, 0), ReadNode(1, 1, 0, 40)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(30, None, None), \
HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(20, None, None))), HuffmanTree(40, None, None))
    >>> lst = [ReadNode(0,1,0,2),ReadNode(0,3,0,4),ReadNode(1,0,1,1),\
    ReadNode(0,5,0,6),ReadNode(0,7,0,8),ReadNode(1,3,1,4),ReadNode(1,2,1,5)]
    >>> generate_tree_general(lst, 6)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, \
HuffmanTree(1, None, None), HuffmanTree(2, None, None)), HuffmanTree(None, \
HuffmanTree(3, None, None), HuffmanTree(4, None, None))), HuffmanTree(None, \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(6, None, None)), \
HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(8, None, None))))
    """
    # Nodes numbers are based on index
    # Turn all nodes into huffman trees
    trees = []
    for node in node_lst:
        # Whenever the left or right nodes are not leafs,
        # replace them with None for now
        lt = node.l_type
        ld = node.l_data
        rt = node.r_type
        rd = node.r_data
        new_tree = HuffmanTree(None)
        if lt == 0:
            new_tree.left = HuffmanTree(ld)
        if rt == 0:
            new_tree.right = HuffmanTree(rd)
        trees.append(new_tree)

    # Check for case where only 1 node
    if len(node_lst) == 1:
        trees[0].right = HuffmanTree(80085)
        return trees[0]

    # Now fill in the missing nodes
    for i in range(len(node_lst)):
        node = node_lst[i]
        curr_tree = trees[i]
        lt = node.l_type
        ld = node.l_data
        rt = node.r_type
        rd = node.r_data
        if lt == 1:
            curr_tree.left = trees[ld]
        if rt == 1:
            curr_tree.right = trees[rd]
    return trees[root_index]


# def _find_num(tree: HuffmanTree, count: int) -> int:
#     if tree.right.is_leaf():
#         if tree.left.is_leaf():
#     else:
#         return _find_num(tree.right)
#     pass

def _tree_postorder_help(n_lst: list[ReadNode], n: ReadNode)\
        -> HuffmanTree:
    """
    Generates a huffman tree given a node list in postorder
    """
    length = len(n_lst) - 1
    lt = n.l_type
    ld = n.l_data
    rt = n.r_type
    rd = n.r_data
    new_tree = HuffmanTree(None)
    if lt == 0:
        new_tree.left = HuffmanTree(ld)
    else:
        new_tree.left = _tree_postorder_help(n_lst[:length], n_lst[length - 2])
    if rt == 0:
        new_tree.right = HuffmanTree(rd)
    else:
        new_tree.right = _tree_postorder_help(n_lst[:length], n_lst[length - 1])
    return new_tree


def _postorder_help(n_lst: list[ReadNode], n: ReadNode)\
        -> HuffmanTree:
    """
    Generates a huffman tree given a node list in postorder
    """
    length = len(n_lst) - 1
    lt = n.l_type
    ld = n.l_data
    rt = n.r_type
    rd = n.r_data
    new_tree = HuffmanTree(None)
    if rt == 0 and lt == 0:
        new_tree.right = HuffmanTree(rd)
        new_tree.left = HuffmanTree(ld)
    elif rt == 1 and lt == 0:
        new_tree.right = _postorder_help(n_lst[:length], n_lst[length - 1])
        new_tree.left = HuffmanTree(ld)
    elif rt == 0 and lt == 1:
        new_tree.right = HuffmanTree(rd)
        new_tree.left = _postorder_help(n_lst[:length], n_lst[length - 1])
    else:
        new_tree.right = _postorder_help(n_lst[:length], n_lst[length - 1])
        new_tree.left = _postorder_help(n_lst[:length], n_lst[length - 2])
    return new_tree


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)))
    >>> lst = [ReadNode(0, 10, 1, 0)]
    >>> generate_tree_postorder(lst, 0)
    HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(80085, None, None))
    >>> lst = [ReadNode(0, 10, 0, 20), ReadNode(1, 0, 0, 30)]
    >>> generate_tree_postorder(lst, 1)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(20, None, None)), HuffmanTree(30, None, None))
    >>> lst = [ReadNode(0, 10, 0, 20), ReadNode(0, 30, 1, 0)]
    >>> generate_tree_postorder(lst, 1)
    HuffmanTree(None, HuffmanTree(30, None, None), \
HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(20, None, None)))
    >>> lst = [ReadNode(0, 10, 0, 20), ReadNode(0, 30, 1, 0), ReadNode(1, 1, 0, 40)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(30, None, None), \
HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(20, None, None))), HuffmanTree(40, None, None))
    """
#     >>> lst = [ReadNode(0,1,0,2),ReadNode(0,3,0,4),ReadNode(1,0,1,0), \
# ReadNode(0,5,0,6),ReadNode(0,7,0,8),ReadNode(1,0,1,0),ReadNode(1,0,1,0)]
#     >>> generate_tree_postorder(lst, 6)
#     HuffmanTree(None, HuffmanTree(None, HuffmanTree(None, \
# HuffmanTree(1, None, None), HuffmanTree(2, None, None)), HuffmanTree(None, \
# HuffmanTree(3, None, None), HuffmanTree(4, None, None))), HuffmanTree(None, \
# HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(6, None, None)), \
# HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(8, None, None))))
# Since list is in postorder, root index is last element
# Check for special case where only 1 node
    if len(node_lst) == 1:
        node = node_lst[0]
        lt = node.l_type
        ld = node.l_data
        rt = node.r_type
        rd = node.r_data
        tree = HuffmanTree(None)
        if lt == 0:
            tree.left = HuffmanTree(ld)
        else:
            tree.left = HuffmanTree(80085)
        if rt == 0:
            tree.right = HuffmanTree(rd)
        else:
            tree.right = HuffmanTree(80085)
        return tree
    return _postorder_help(node_lst, node_lst[root_index])


def _swap_dictionary(freq_dict: dict[int, str]) -> dict[str, int]:
    """Swaps the keys and values of a dictionary

    Precondition: each key can have at most 1 value

    >>> _swap_dictionary({'a': 1, 'b': 2, 3: 'c'})
    {1: 'a', 2: 'b', 'c': 3}
    """
    # Swap the keys and values so it can be sorted
    swapped_dict = {}
    for key in freq_dict:
        value = freq_dict[key]
        swapped_dict[value] = key
    return swapped_dict


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    >>> tree = build_huffman_tree(build_frequency_dict(b'0'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'0', get_codes(tree)), len(b'0'))
    b'0'
    """
    # decode = ''
    # message = bytearray()
    #
    # key = _swap_dictionary(get_codes(tree))
    # longest = 0
    # for i in key:
    #     if len(i) > longest:
    #         longest = len(i) - 1
    # for number in text:
    #     decode += byte_to_bits(number)
    #     begin = 0
    #     end = 1
    #     while len(decode) > longest and size > 0:
    #         sub_string = decode[begin: end]
    #         # Update end until a match is found, then update start and end
    #         if sub_string in key:
    #             message.append(key[sub_string])
    #             decode = decode[end:]
    #             begin = 0
    #             end = 1
    #             size -= 1
    #         else:
    #             end += 1
    # return bytes(message)

    decode = ''
    message = bytearray()

    key = _swap_dictionary(get_codes(tree))
    longest = 0
    for i in key:
        if len(i) > longest:
            longest = len(i)
    # b'\x06\xba\xee\xc0' [6, 186, 238, 192]
    for number in text:
        decode += byte_to_bits(number)
        begin = 0
        end = 1
        while len(decode) > longest and size > 0:
            sub_string = decode[begin: end]
            # Update end until a match is found, then update start and end
            if sub_string in key:
                message.append(key[sub_string])
                decode = decode[end:]
                begin = 0
                end = 1
                size -= 1
            else:
                end += 1
    begin = 0
    end = 1
    while size > 0:
        sub_string = decode[begin: end]
        # Update end until a match is found, then update start and end
        if sub_string in key:
            message.append(key[sub_string])
            decode = decode[end:]
            begin = 0
            end = 1
            size -= 1
        else:
            end += 1

    return bytes(message)

    # decode = ''
    # message = bytearray()
    # key = _swap_dictionary(get_codes(tree))
    # for number in text:
    #     decode += byte_to_bits(number)
    # begin = 0
    # end = 1
    # size -= 1
    #
    # while size > -1:
    #     sub_string = decode[begin: end]
    #     # Update end until a match is found, then update start and end
    #     if sub_string in key:
    #         message.append(key[sub_string])
    #         begin = end
    #         end += 1
    #         size -= 1
    #     else:
    #         end += 1
    # return bytes(message)

    # decode = ''
    # message = bytearray()
    # key = _swap_dictionary(get_codes(tree))
    # for number in text:
    #     decode += byte_to_bits(number)
    # begin = 0
    # end = 1
    # size -= 1
    #
    # while size > -1:
    #     sub_string = decode[begin: end]
    #     # Update end until a match is found, then update start and end
    #     if sub_string in key:
    #         message.append(key[sub_string])
    #         decode = decode[end:]
    #         begin = 0
    #         end = 1
    #         size -= 1
    #     else:
    #         end += 1
    #
    # return bytes(message)

    # decode = ''
    # message = bytearray()
    # key = _swap_dictionary(get_codes(tree))
    # for number in text:
    #     decode += byte_to_bits(number)
    # begin = 0
    # end = 1
    #
    # while size > 0:
    #     sub_string = decode[begin: end]
    #     # Update end until a match is found, then update start and end
    #     if sub_string in key:
    #         message.append(key[sub_string])
    #         decode = decode[end:]
    #         begin = 0
    #         end = 1
    #         size -= 1
    #     else:
    #         end += 1
    #
    # return bytes(message)

    # decode = ''
    # message = bytearray()
    #
    # key = _swap_dictionary(get_codes(tree))
    # longest = 0
    # for i in key:
    #     if len(i) > longest:
    #         longest = len(i)
    # for number in text:
    #     decode += byte_to_bits(number)
    #     begin = 0
    #     end = 1
    #     while len(decode) > longest and size > 1:
    #         sub_string = decode[begin: end]
    #         # Update end until a match is found, then update start and end
    #         if sub_string in key:
    #             message.append(key[sub_string])
    #             decode = decode[end:]
    #             begin = 0
    #             end = 1
    #             size -= 1
    #         else:
    #             end += 1
    # for i in decode:
    #     if decode[:i] in key:
    #         message.append(key[sub_string])
    # return bytes(message)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    ans_key = _sort_frequency_dict(freq_dict)
    sorted_values = [j for i in ans_key for j in ans_key[i]]
    q = [tree]
    while len(q) > 0:
        curr_tree = q.pop(-1)
        if curr_tree.is_leaf():
            curr_tree.symbol = sorted_values.pop(0)
        elif tree.right is None or tree.left is None:
            return tree
        else:
            q.append(curr_tree.left)
            q.append(curr_tree.right)


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
