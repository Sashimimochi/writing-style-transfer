import os


class MakeJaDataset:
    def __init__(self):
        self.POS = 5
        self.NEG = 1
        self.DATA_DIR = 'mydata'
        self.DOMAIN_A = 'domain_a.txt'
        self.DOMAIN_B = 'domain_b.txt'

    def make_dataset(self, filepath: str, label: int):
        '''{"star": label, "text": "text"}'''
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data.append('{{"stars": {label}, "text": "{text}"}}'.format(label=label, text=line.rstrip('\n').replace('"', '')))

        return data

    def build_ja_data(self):
        data = self.make_dataset(filepath=os.path.join(self.DATA_DIR, self.DOMAIN_A), label=5.0)
        data += self.make_dataset(filepath=os.path.join(self.DATA_DIR, self.DOMAIN_B), label=1.0)
        with open('mydata/mydata.json', 'w', encoding='utf-8') as f:
            f.write('\n'.join(data))