
def retrieve_idxs(full_list, sub_list):
    id_map = {id(e): i for i, e in enumerate(full_list)}
    return [id_map[id(e)] for e in sub_list]

def next_pow_of_2(n):
    '''return the next power of two greater than or equal to n (e.g., 6 --> 8, 9 --> 16, 32 --> 32'''
    return 1 << (n - 1).bit_length()
