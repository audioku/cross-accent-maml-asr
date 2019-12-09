
from utils.data import is_contain_chinese_word, get_word_segments_per_language

def calculate_lm_score(seq, lm, vocab):
    """
    seq: (1, seq_len)
    id2label: map
    """
    seq_str = "".join(vocab.id2label[char.item()] for char in seq[0]).replace(
        vocab.PAD_CHAR, "").replace(vocab.SOS_CHAR, "").replace(vocab.EOS_CHAR, "")
    seq_str = seq_str.replace("  ", " ")

    seq_arr = get_word_segments_per_language(seq_str)
    seq_str = ""
    for i in range(len(seq_arr)):
        if is_contain_chinese_word(seq_arr[i]):
            for char in seq_arr[i]:
                if seq_str != "":
                    seq_str += " "
                seq_str += char
        else:
            if seq_str != "":
                seq_str += " "
            seq_str += seq_arr[i]

    seq_str = seq_str.replace("  ", " ").replace("  ", " ")

    if seq_str == "":
        return -999, 0, 0

    score, oov_token = lm.evaluate(seq_str)    
    
    # a, b = lm.evaluate("除非 的 不会 improve 什么 东西 的 这些 esperience")
    # a2, b2 = lm.evaluate("除非 的 不会 improve 什么 东西 的 这些 experience")
    # print(a, a2)
    return -1 * score / len(seq_str.split()) + 1, len(seq_str.split()) + 1, oov_token