from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import pandas as pd


class UJIDataLoader(BaseDataLoader):
    """
    UJI data loading demo using BaseDataLoader
    """
    # def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
    def __init__(self, data_dir):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = self.load("ujipenchars2.txt")
        # super(UJIDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    
    def load(self, file_name):
        with open(self.data_dir + file_name) as f:
            lines = f.readlines()

        character_dict = self.isolate_char(lines)

        df = pd.DataFrame(character_dict)
        df.to_csv(file_name+".csv")
        return df

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
        # norm = []
        mean = [ sum(y)/float(len(y)) for y in zip(*sequence)]
        mean = tuple(mean)
        return list(map(lambda x: (round(x[0] - mean[0], 3), round(x[1] - mean[1], 3)), sequence))
        # return norm

i = UJIDataLoader("data/")
