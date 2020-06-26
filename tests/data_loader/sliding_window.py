#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 



# author: xiaoy li 




def construct_sliding_windows(sequence_length, sliding_window_size):
    """
    construct sliding windows for BERT processing
    :param sequence_length: e.g. 9
    :param sliding_window_size: e.g. 4
    :return: [(0, 4, [1, 1, 1, 0]), (2, 6, [0, 1, 1, 0]), (4, 8, [0, 1, 1, 0]), (6, 9, [0, 1, 1])]
    """
    sliding_windows = []
    stride = int(sliding_window_size / 2)
    start_index = 0
    end_index = 0
    while end_index < sequence_length:
        end_index = min(start_index + sliding_window_size, sequence_length)
        left_value = 1 if start_index == 0 else 0
        right_value = 1 if end_index == sequence_length else 0
        mask = [left_value] * int(sliding_window_size / 4) + [1] * int(sliding_window_size / 2) \
               + [right_value] * (sliding_window_size - int(sliding_window_size / 2) - int(sliding_window_size / 4))
        mask = mask[: end_index - start_index]
        sliding_windows.append((start_index, end_index, mask))
        start_index += stride
    assert sum([sum(window[2]) for window in sliding_windows]) == sequence_length
    return sliding_windows




if __name__ == "__main__":
    seq_len = 512
    slid_win_size = 128
    windows = construct_sliding_windows(seq_len, slid_win_size)
    print(windows[0])
    print(len(windows[0][-1]))
    print(windows[1])








