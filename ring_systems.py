#!/usr/bin/env python

import sys
from rdkit import Chem
import pandas as pd
from tqdm.auto import tqdm


class RingSystemFinder:
    def __init__(self):
        """
        Initialize susbstructure search objects to identify key functionality
        """
        self.ring_db_pat = Chem.MolFromSmarts("[#6R,#18R]=[OR0,SR0,CR0,NR0]")
        self.ring_atom_pat = Chem.MolFromSmarts("[R]")

    def tag_bonds_to_preserve(self, mol):
        """ Assign the property "protected" to all ring carbonyls, etc.
        :param mol: input molecule
        :return: None
        """
        for bnd in mol.GetBonds():
            bnd.SetBoolProp("protected", False)
        for match in mol.GetSubstructMatches(self.ring_db_pat):
            bgn, end = match
            bnd = mol.GetBondBetweenAtoms(bgn, end)
            bnd.SetBoolProp("protected", True)

    @staticmethod
    def cleave_linker_bonds(mol):
        """ Cleave bonds that are not in rings and not protected
        :param mol: input molecule
        :return: None
        """
        frag_bond_list = []
        for bnd in mol.GetBonds():
            if not bnd.IsInRing() and not bnd.GetBoolProp("protected") and bnd.GetBondType() == Chem.BondType.SINGLE:
                frag_bond_list.append(bnd.GetIdx())

        if len(frag_bond_list):
            frag_mol = Chem.FragmentOnBonds(mol, frag_bond_list)
            Chem.SanitizeMol(frag_mol)
            return frag_mol
        else:
            return mol

    def cleanup_fragments(self, mol, keep_dummy=False):
        """
        split a molecule containing multiple ring systems into individual ring systems
        :param keep_dummy: retain dummy atoms
        :param mol: input molecule
        :return: a list of SMILES corresponding to individual ring systems
        """
        frag_list = Chem.GetMolFrags(mol, asMols=True)
        ring_system_smiles_list = []
        for frag in frag_list:
            if frag.HasSubstructMatch(self.ring_atom_pat):
                for atm in frag.GetAtoms():
                    if atm.GetAtomicNum() == 0:
                        if keep_dummy:
                            atm.SetProp("atomLabel", "R")
                        else:
                            atm.SetAtomicNum(1)
                        atm.SetIsotope(0)
                # Convert explict Hs to implicit
                frag = Chem.RemoveAllHs(frag)
                ring_system_smiles_list.append(Chem.MolToSmiles(frag))
        return ring_system_smiles_list

    def find_ring_systems(self, mol, keep_dummy=False):
        """
        find the ring systems for an RDKit molecule
        :param keep_dummy: retain dummy atoms
        :param mol: input molecule
        :return: a list of SMILES corresponding to individual ring systems
        """
        self.tag_bonds_to_preserve(mol)
        frag_mol = self.cleave_linker_bonds(mol)
        ring_system_smiles_list = self.cleanup_fragments(frag_mol, keep_dummy=keep_dummy)
        return ring_system_smiles_list


def test_ring_system_finder():
    mol = Chem.MolFromSmiles("CC(=O)[O-].CCn1c(=O)/c(=C2\Sc3ccccc3N2C)s/c1=C\C1CCC[n+]2c1sc1ccccc12")
    ring_system_finder = RingSystemFinder()
    ring_system_finder.find_ring_systems(mol)


def create_ring_dictionary(input_smiles, output_csv):
    """
    read a SMILES file, extract ring systems, write out ring systems and frequency
    :param input_smiles: input SMILES file
    :param output_csv: output csv file
    :return: None
    """
    df = pd.read_csv(input_smiles, sep=" ", names=["SMILES", "Name"])
    ring_system_smiles_list = []
    for smi in tqdm(df.SMILES):
        mol = Chem.MolFromSmiles(smi)
        ring_system_finder = RingSystemFinder()
        ring_system_smiles_list += ring_system_finder.find_ring_systems(mol)
    df_out = pd.DataFrame(pd.Series(ring_system_smiles_list).value_counts())
    df_out.index.name = "ring_system"
    df_out.columns = ['count']
    df_out = df_out.reset_index()
    df_out.to_csv(output_csv)


if __name__ == "__main__":
    create_ring_dictionary(sys.argv[1], sys.argv[2])
