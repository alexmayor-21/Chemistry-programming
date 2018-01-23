from glob import glob 
import re
import numpy as np
from math import pi, sqrt
from decimal import Decimal
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse, sys

class Freq_NM:
    
    """
    Class used to calculate frequencies of stretch and bend normal modes of
    molecules HXH from energies calculations using SCF
    
    Example usage:
    
    Freq_NM(<path to folder>) -- initialise class and compute normal mode frequencies
    print(<class instance>) -- print normal mode frequencies
    <class instance>.plot() -- plot 3D energy surface
    <class instance>.freqr -- frequency of stretch normal mode
    <class instance>.freqtheta -- frequency of bend normal mode
    
    """
    
    def __init__(self, path):
        """
        Get data from text files into a numpy array and process
        """
        self.name = re.search(r'(\w+)outfiles', path).group(1)
        numfiles = len(glob('%s/*.out' % path))
        self.data = np.zeros((numfiles, 3))

        get_params = re.compile('r(\d+\.?\d+)theta(\d+\.?\d+)')
        get_energy = re.compile('E\(\w+\)\s+=\s+([-+]?\d+\.?\d+)')

        for i, doc in enumerate(glob('%s/*.out' % path)):
            params = re.search(get_params, doc)
            r = params.group(1)
            theta = params.group(2)
            with open(doc, 'r') as idoc:
                text = idoc.read()
                [match] = re.finditer(get_energy, text)
                energy = match.group(1)
            self.data[i] = [r, theta, energy]
        
        self.freqr, self.freqtheta = self._get_nmfreq()
    
    def _get_nmfreq(self):
        """
        Calculate normal mode frequencies
        """
        data = self.data
        hartree_to_J = 4.35974*10**(-18)
        NA = 6.02214086*10**(23)
    
        minindex = np.argmin(data[:,2])
        [minr, mintheta, minenergy] = data[minindex].squeeze()
        
        dataminr = data[np.where(data[:,1]==mintheta)]
        p = np.polyfit((dataminr[:,0]-minr)**2, dataminr[:,2], 1)
        kr = p[0]*2*hartree_to_J/(10**(-10*2))
        mu1 = 2*10**(-3)/NA
        freqr = 1/(2*pi)*sqrt(kr/mu1)
        
        datamintheta = data[np.where(data[:,0]==minr)]
        p = np.polyfit((datamintheta[:,1]-mintheta)**2, datamintheta[:,2], 1)
        ktheta = p[0]*2*hartree_to_J
        mu2 = 0.5*10**(-3)/NA
        freqtheta = 1/(2*pi)*sqrt(ktheta/(mu2*(minr*10**(-10))**2))
        
        return (freqr, freqtheta)
    
    def plot(self):
        """
        Plot the energy surface
        """
        xunique = len(np.unique(self.data[:,0]))
        yunique = len(np.unique(self.data[:,1]))

        x_changes_first = len(np.unique(self.data[0:xunique,0])) == xunique
        if x_changes_first:
            shape = (yunique, xunique)
        else:
            shape = (xunique, yunique)

        X = self.data[:,0].reshape(shape)
        Y = self.data[:,1].reshape(shape)
        Z = self.data[:,2].reshape(shape)
        fig = plt.figure(figsize=(8.5,8.5))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap = cm.coolwarm)
        ax.set_xlabel(u'r / \u212B')
        ax.set_ylabel(u'\u0398')
        ax.set_zlabel(u'Energy / hartree')
        ax.set_title('Energy surface for %s' % self.name)
        
    def __str__(self):
        return '\n%s\nfreq_r =\t%.4E\nfreq_theta =\t%.4E' % \
                (self.name, Decimal(self.freqr), Decimal(self.freqtheta))
        
    def __repr__(self):
        return 'normal modes of %s' % self.name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="Unix path to folder with .out files", type=str)
    parser.add_argument("--output", help="Unix path where save plots", type=str)
    parser.add_argument("-defaultH2O", help="Run command on precomputed H2O energies", action="store_false")
    parser.add_argument("-defaultH2S", help="Run command on precomputed H2S energies", action="store_false")
    args = parser.parse_args()
    
    if args.defaultH2O:
        FOLDER = '/Users/alexmayorov/Downloads/Ex2 Files-20171213/H2Ooutfiles'
        inst = Freq_NM(FOLDER)
        print inst
        inst.plot()
        
    elif args.defaultH2S:
        FOLDER2 = '/Users/alexmayorov/Downloads/Ex2 Files-20171213/H2Soutfiles'
        inst = Freq_NM(FOLDER)
        print inst
        inst.plot()
        
    else:
        inst = Freq_NM(arg.folder)
        print inst
        inst.plot()
        if arg.output:
            plt.savefig(arg.output)