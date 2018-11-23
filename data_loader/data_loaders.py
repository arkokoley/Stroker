from torchvision import datasets, transforms
import torch
from base import BaseDataLoader
import numpy as np
import pandas as pd


class UJIDataLoader(BaseDataLoader):
    """
    UJI data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.data_dir = data_dir
        self.dataset = self.load()
        super(UJIDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def load(self):
        df = pd.read_pickle(self.data_dir+"ujipenchars2.txt.pkl")
        out = []
        labels = list(np.unique(df.label))
        for i in range(len(df)):
            out.append([df.loc[i].seq, labels.index(df.loc[i].label)])
        print (out[-1])
        return out

    def create(self, file_name):
        with open(self.data_dir + file_name) as f:
            lines = f.readlines()

        character_dict = self.isolate_char(lines)
        # np.array()
        df = pd.DataFrame(character_dict)
        df.to_pickle(self.data_dir+file_name+".pkl")
        return

    def isolate_char(self, lines):
        chars = {'seq': [], 'label': [], 'label_ix': []}
        char_set = []
        for i in range(len(lines)):
            words = lines[i].split()
            if words[0] == 'WORD':
                num_strokes = int(lines[i+1].strip().split()[1])
                character = words[1]
                sequence = self.stroke_sequence([ lines[i+2+k].strip() for k in range(num_strokes) ])
                chars['seq'].append(self.normalize(sequence))
                chars['label'].append(character)
                char_set.append(character)
        for c in chars['label']:
            for i, x in enumerate(np.unique(char_set)):
                if(x == c):
                    chars['label_ix'].append(i)
        return chars

    def stroke_sequence(self, strokes):
        seq = []
        for s in strokes:
            a = map(lambda x: int(x), s.split('#')[1].split())
            l = zip(*[iter(a)]*2)
            seq.extend(list(l))
        return seq

    def normalize(self, sequence):
        mean = [ sum(y)/float(len(y)) for y in zip(*sequence)]
        mean = tuple(mean)
        return list(map(lambda x: [round(x[0] - mean[0], 3), round(x[1] - mean[1], 3)], sequence))

if __name__ == "__main__":
    loader = UJIDataLoader('data/',1,True,0.2,2)
    # loader.create("ujipenchars2.txt")