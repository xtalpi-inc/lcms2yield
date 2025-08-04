import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.data import Data
from rdkit import Chem, RDLogger

x_map = {
    'atomic_symbol':['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'I', 'Si'],
    'atomic_num': list(range(0, 119)),
    'chirality': ['CHI_UNSPECIFIED',
                  'CHI_TETRAHEDRAL_CW',
                  'CHI_TETRAHEDRAL_CCW',
                  'CHI_OTHER',
                  ],
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'num_radical_electrons': list(range(0, 5)),
    'hybridization': ['UNSPECIFIED',
                      'S',
                      'SP',
                      'SP2',
                      'SP3',
                      'SP3D',
                      'SP3D2',
                      'OTHER',
                      ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': ['misc',
                  'SINGLE',
                  'DOUBLE',
                  'TRIPLE',
                  'AROMATIC',
                  ],
    'stereo': ['STEREONONE',
               'STEREOZ',
               'STEREOE',
               'STEREOCIS',
               'STEREOTRANS',
               'STEREOANY',
               ],
    'is_conjugated': [False, True],
}




class MolDataset(Dataset):
    def __init__(self, mol_smi, properties=None, rdkit_desc_dict=None, node_mask_idxs_list=None):
        super().__init__()
        self.mol_smi = mol_smi
        self.properties = properties if properties is not None else len(self.mol_smi)*[0.0]
        self.rdkit_desc_dict = rdkit_desc_dict
        self.node_mask_idxs_list = node_mask_idxs_list
        assert len(self.mol_smi) ==  len(self.properties),f'SMILES与属性的个数不一致! smiles: {len(self.mol_smi)} property:{len(self.properties)}'

    def get(self, idx):
        if not isinstance(idx, int):
            raise TypeError("index can only be an integer.")
        g_mol = from_smiles(self.mol_smi[idx], with_hydrogen=True)
        if self.node_mask_idxs_list is not None:
            g_mol.mask = torch.tensor(self.node_mask_idxs_list[idx],dtype=torch.float)
        return g_mol, self.properties[idx]

    def len(self):
        return len(self.mol_smi)



def from_smiles(smiles: str, 
                with_hydrogen: bool = False,
                kekulize: bool = False):
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (string, optional): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """


    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    xs = []
    x = []
    for atom in mol.GetAtoms():
        x = []
        x += one_of_k_encoding_unk(atom.GetSymbol(),x_map['atomic_symbol'])                   # 原子种类
        x += one_of_k_encoding(str(atom.GetChiralTag()),x_map['chirality'])                   # 原子手性
        x += one_of_k_encoding(atom.GetTotalDegree(),x_map['degree'])                         # 原子度
        x += one_of_k_encoding(atom.GetFormalCharge(),x_map['formal_charge'])                 # 形式电荷
        x += one_of_k_encoding(atom.GetTotalNumHs(),x_map['num_hs'])                          # 氢原子个数
        x += one_of_k_encoding(atom.GetNumRadicalElectrons(),x_map['num_radical_electrons'])  # 激活电子
        x += one_of_k_encoding(str(atom.GetHybridization()),x_map['hybridization'])           # 杂化类型
        x += one_of_k_encoding(atom.GetIsAromatic(),x_map['is_aromatic'])                     # 是否在芳香烃内
        x += one_of_k_encoding(atom.IsInRing(),x_map['is_in_ring'])                           # 是否在环上
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.float).view(-1, len(x))

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e+=one_of_k_encoding(str(bond.GetBondType()), e_map['bond_type'])
        e+=one_of_k_encoding(str(bond.GetStereo()), e_map['stereo'])
        e+=one_of_k_encoding(bond.GetIsConjugated(), e_map['is_conjugated'])
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, 13)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
