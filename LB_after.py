import os
import json
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_res(path_in):
    old_ent = []
    new_ent = []
    with open(path_in, "r", encoding="utf-8") as f1:
        for line in tqdm.tqdm(f1):
            line_dict = json.loads(line)
            old_ent.append(line_dict['old_entropy'])
            new_ent.append(line_dict['new_entropy'])
            # js = json.dumps(line_dict, ensure_ascii=False)
        f1.close()
    return old_ent, new_ent

def ent_select(in_file, ent, out_file):
    with open(out_file, "w", encoding="utf-8") as f0:
        with open(in_file, "r", encoding="utf-8") as f1:
            ind = 0
            for line in tqdm.tqdm(f1):
                line_dict = json.loads(line)
                js = json.dumps(line_dict, ensure_ascii=False)
                if ent[0][ind] - ent[1][ind]>0:
                    f0.write(js + '\n')
                ind += 1
            f1.close()
        f0.close()

#
if __name__ == '__main__':

    old_ent, new_ent = get_res("entropy_10m_zh.jsonl")
    in_file = "10m_zh_refine.jsonl"
    out_file = "10m_zh_entropy.jsonl"

    ent_select(in_file, (old_ent, new_ent), out_file)

    plt.style.use('ggplot')
    plt.hist(np.array(old_ent)-np.array(new_ent), bins=20, color = 'steelblue',
         edgecolor = 'black')
    plt.xlabel('Entropy')
    plt.ylabel('Number')
    plt.show()

    print(pd.Series(np.array(old_ent)-np.array(new_ent)).describe())
    norm = (np.array(old_ent) - np.array(new_ent)) / max(np.array(old_ent) - np.array(new_ent))
    print(pd.Series(norm).describe())
    print('Finish')