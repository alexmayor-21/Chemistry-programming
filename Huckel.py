import argparse, sys
from collections import Counter
import math
import numpy as np
import sys
import unittest

class Huckel:
    
    """
    Class used to calculate energies of delocalised systems using Huckel's theory.
    """
    platonic_adjacency = {4: {0:[1,2,3], 1:[2,3], 2:[3]},
            6: {0:[1,2,3,4], 1:[2,3,5], 2:[3,5], 3:[4,5], 4:[5]},
            8: {0:[1,3,4], 1:[2,5], 2:[3,6], 3:[7], 4:[5,7], 5:[6], 6:[7]}}
                
    def __init__(self, n, cyclic=False, platonic=False, alpha=0, beta=-1, verbose=True):
        if n < 2:
            raise ValueError("Molecule isn't defined")
        if platonic and n not in platonic_adjacency.keys():
            raise ValueError("Platonic solid with given number vertices doesn't exist")
        if cyclic and platonic:
            raise ValueError("Molecule cannot be both cyclic and platonic") 
        
        self.n = n
        self.cyclic = cyclic
        self.platonic = platonic
        self.alpha = alpha
        self.beta = beta
        self._bonds = self._getBonds()
        self.eig = self._getEig()
        if verbose: print(self)
        
    def _getBonds(self):
        """
        Return a dictionary where keys are atom positions and values are lists of positions, greater than the key,
        that are connected to it.
        """
        if self.platonic:
            return platonic_adjacency[n]
        else:
            bonds = {i: [i+1] for i in range(self.n-1)}
            if self.cyclic: bonds[0].append(self.n-1)
            return bonds

    def _getEig(self):
        """
        Construct Huckel matrix and return energy levels and with degeneracies as a sorted list of tuples.
        """
        hMat = np.zeros((self.n, self.n))
        np.fill_diagonal(hMat, self.alpha)
        for row in range(self.n):
            for col in range(row+1, self.n, 1):
                if col in self._bonds[row]:
                    hMat[row, col] = self.beta
                    hMat[col, row] = self.beta
        evals, evecs = np.linalg.eig(hMat)
        # run-time assertion: method works only in Python 3
        if sys.version_info[0] > 3:
            assert math.isclose(sum(evals), 0.), "Panic! Huckel energies don't sum to 0!"
        return sorted(Counter(evals).items(), reverse=True)  
    
    def __str__(self):
        """
        Print energies and degeneracies with formatting.
        """
        print_statement = list()
        for energy, degen in self.eig:
            if energy > 0: print_statement.append(" %.3f\t%i" % (energy, degen))
            else: print_statement.append("%.3f\t%i" % (energy, degen))
        print_statement = "Energy\tDegeneracy\n" + '\n'.join(print_statement) + '\n'
        return print_statement

class TestOutputValues(unittest.TestCase):

    """
    Unit testing against known values 
    """
    def test_butadiene(self):
        energs, degs = zip(*Huckel(4, verbose=False).eig)
        energs = set([round(energ, 3) for energ in energs])
        self.assertEqual(energs, set([1.618, 0.618, -1.618, -0.618]))

    def test_cyclobutadiene(self):
        energs, degs = zip(*Huckel(4, cyclic=True, verbose=False).eig)
        energs = set([round(energ, 3) for energ in energs])
        self.assertEqual(energs, set([0., -2., +2.]))

if __name__ == '__main__':          
    parser = argparse.ArgumentParser()
    parser.add_argument("n", help="number of atoms in molecule", type=int)
    parser.add_argument("-cyclic", help="flag for cyclic molecule", action="store_true")
    parser.add_argument("-platonic", help="flag for platonic solid", action="store_true")
    parser.add_argument("-unittest", help="flag for unit testing", action="store_false")
    parser.add_argument("--alpha", default=0, help="value of alpha (AO energy)", type=int)
    parser.add_argument("--beta", default=-1, help="value of beta (resonance integral)", type=int)
    args = parser.parse_args()
    Huckel(n=args.n, cyclic=args.cyclic, platonic=args.platonic, alpha=args.alpha, beta=args.beta)