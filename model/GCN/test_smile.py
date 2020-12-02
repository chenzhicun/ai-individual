from pysmiles import read_smiles

max_length = 0
element_set = set()
hcount_set = set()
charge_set = set()
aromatic_set = set()

def process_smile_file(filepath):
    global max_length
    with open(filepath, 'r') as f:
        smiles = f.read().split('\n')[1:]
        for smile in smiles:
            if smile == '':
                continue
            smile = smile.split(',')[1]
            # print(smile)
            # x = input()
            if smile == '':
                print('fuck')
                continue
            mol = read_smiles(smile)
            if max_length < len(mol.nodes()):
                max_length = len(mol.nodes())
            hcount_list = mol.nodes(data='hcount')
            element_list = mol.nodes(data='element')
            charge_list = mol.nodes(data='charge')
            aromatic_list = mol.nodes(data='aromatic')
            
            for i in range(len(hcount_list)):
                try:
                    hcount_set.add(hcount_list[i])
                except:
                    pass
                try:
                    element_set.add(element_list[i])
                except:
                    pass
                try:
                    charge_set.add(charge_list[i])
                except:
                    pass
                try:
                    aromatic_set.add(aromatic_list[i])
                except:
                    pass

process_smile_file('./data/train/names_smiles.txt')
process_smile_file('./data/validation/names_smiles.txt')
process_smile_file('./data/test/names_smiles.txt')

print('max_length:', max_length)
print('element:')
print(len(element_set))
print(element_set)
print('hcount:')
print(len(hcount_set))
print(hcount_set)
print('charge:')
print(len(charge_set))
print(charge_set)
print('aromatic:')
print(len(aromatic_set))
print(aromatic_set)