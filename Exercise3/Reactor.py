import numpy as np
from math import exp
import matplotlib.pyplot as plt
import re
import random
import argparse, sys

class Reactor(object):
    
    """
    Class mimicking a chemical reactor
    """
    
    class Reaction(object):
        """
        Class storing chemical species involved in a reaction and its
        kinetic parameters
        """
        def __init__(self, reactants, products, const):
            """
            reactants and products -- as a dictionary with keys corresponding to chemical species
            and values to their coefficients (all coefficients are positive)
            const -- rate constant
            """
            self.reactants = reactants
            self.products = products
            self.const = const
            
        def get_rate(self, concs):
            """
            Get rate of the equation at the concentrations specified by concs
            concs should be a dictionary with keys corresponding to reactants
            and values to their concentration
            """
            rate = self.const
            for (species, stoichiom) in self.reactants.items():
                rate *= concs[species]**stoichiom
            return rate
        
    def __init__(self, concs, reacts, timestep=1e-6):
        """
        concs -- initial concentrations of all chemical species as a dictionary, with
        species as keys and their concentrations as values
        reacts -- chemical reactions in format [("A+B->2C+D", rate constant), ...]
        timestep -- time-step used for Euler integration
        """
        self.concs = concs
        self.reacts = list()
        for (react, const) in reacts:
            (reactants, products) = Reactor.process_react(react)
            self.reacts.append(Reactor.Reaction(reactants, products, const))
        self.timestep = timestep
    
    @staticmethod
    def process_react(stri):
        """
        Convert chemical reactions from the format "A+B->2C+D" to the format 
        {"chemical species": stoichiometric coeff, ...} that is accepted by Reaction class
        """
        [reactants, products] = stri.split('->')
        find_tuple = re.compile('(\d*)(\w*)')
        
        def get_dict(stri):
            li = [term.strip() for term in stri.split('+')]
            species_dict = dict()
            for term in li:
                match = re.search(find_tuple, term)
                stoichiom = int(match.group(1)) if match.group(1) else 1
                species = match.group(2)
                species_dict[species] = stoichiom
            return species_dict
        
        reactants = get_dict(reactants)
        products = get_dict(products)
        return reactants, products
    
    def _update(self):
        """
        Get updates in concentrations using current concentrations and kinetic parameters of each reaction
        """
        self.updates = {species: 0. for species in self.concs.keys()}
        for react in self.reacts:
            reactants_names = [reactant[0] for reactant in react.reactants]
            rate = react.get_rate(self.concs)
            for (name, stoichiom) in react.products.items():
                self.updates[name] += rate * stoichiom * self.timestep
            for (name, stochiom) in react.reactants.items():
                self.updates[name] -= rate * stoichiom * self.timestep
        for (species, update) in self.updates.items():
            self.concs[species] += update
    
    def jump_forward(self, jump_size):
        """
        jump_size -- the number of steps to jump forward by
        """
        for i in range(jump_size):
            self._update()
            
    def converge(self, jump_size=500, convergence_limit=1e-5, verbose=False):
        """
        Run chemical reactor until converged in concentrations of all chemical species,
        convergence is achieved when all species change their concentrations by less than
        10**(-5) (or alternative convergence_limit) after 500 (or alternative jump_size) steps
        
        jump_size -- the number of steps to jump forwar by
        convergence_limit -- when all concentrations are changed by less than the convergence_limit
        after a jump, declare convergence
        verbose -- how much information to print out in real-time, can be False, True or 'extra'
        
        """
        def converged():
            diffs = [abs(self.concs[i]-freeze_concs[i]) for i in self.concs]
            return all([diff < convergence_limit for diff in diffs]) 
        
        freeze_concs = dict(self.concs)
        self.jump_forward(jump_size)
        count = 0
        while not converged():
            freeze_concs = dict(self.concs)
            self.jump_forward(jump_size)
            count += 1
            if verbose == 'extra':
                print(self)
        
        if verbose:
            print("Converged after %i iterations with timestep = %f" % (count, self.timestep))
            print(self)
    
    def __str__(self):
        """
        Format and print current concentrations
        """
        header = 'Concentrations\n'
        conc_str = '\n'.join('%s:\t%.4f' % (name, conc) 
                            for (name, conc) in self.concs.items())
        return header + conc_str

class Diffusion(object):
    
    """
    A system of interconnected cells each running a set of chemical reactions and linked
    to each other by diffusion of chemical species
    DEPENDS ON Reactor class
    """
        
    def __init__(self, dim, diffconst, concs_default, reacts, freq_special=None, concs_special=None, timestep=1e-6):
        """
        dim -- (list of) spatial dimension(s) great than 1
        diffconst -- 0.0 <= diffusion coefficient <= 0.5 (each species diffuses as difference in conc * diffconst)
        concs and reacts -- concentrations and reactions as required by the Reactor class
        prob_special -- probability that a given cell is special
        concs_special -- initial concentrations of special cells
        """
        self.dim = dim
        self.diffconst = diffconst
        self.species = concs_default.keys()
        # CHANGE TO INCREASE DIMENSIONALITY
        self.cells = list()
        for i in range(dim):
            if not i % freq_special:
                self.cells.append(Reactor(dict(concs_special), dict(reacts), timestep))
            else:
                self.cells.append(Reactor(dict(concs_default), dict(reacts), timestep))
    
    def _diffuse(self):
        """
        Diffuse from each cell to its neighbours
        """
        for cell in self.cells:
            cell.interm_concs = dict(cell.concs) # storing intermediate concentrations
        for i in range(self.dim-1):
            cell1 = self.cells[i]
            cell2 = self.cells[i+1]
            for species, cons in cell1.concs.items():
                update = (cell1.concs[species] - cell2.concs[species]) * self.diffconst
                cell1.interm_concs[species] -= update
                cell2.interm_concs[species] += update
        for cell in self.cells:
            cell.concs = cell.interm_concs
            
    def jump_forward(self, jump_size=100):
        """
        jump_size -- number of steps to jump forward by
        Jump forward by jump_size number of time_steps"""
        for step in range(jump_size):
            for cell in self.cells:
                cell._update()
            self._diffuse()
            
    def plot1d(self, species=None):
        """
        Plot concentration vs cell number of a particular chemical species or species
    
        species -- string or list of string of species to plot
        """
        def get_conc_list(speci):
            concs = list()
            for i in range(self.dim):
                cell = self.cells[i]
                concs.append(cell.concs[speci])
            return concs

        plt.figure(figsize=(10,10))
        plt.xlabel('cell number')
        if species == None:
            species = self.species
        if isinstance(species, list):
            for speci in species:
                concs = get_conc_list(speci)
                plt.plot(range(self.dim), concs, legend=speci)
            plt.ylabel('[chemical species]')
            plt.legend()
        else:
            concs = get_conc_list(species)
            plt.plot(range(self.dim), concs)
            plt.ylabel('[%s]' % species)

        plt.show()
        
    def plot2d(self, species):
        # CHANGE TO INCREASE DIMENSIONALITY
        # can use plt.imshow(<numpy array of concentrations>) to visualise 2D systems
        return

def urea_folding_experiment(outarr, outfig, verbose=False, ureaconc_range=np.arange(0., 8., .25), concs={'D': 1., 'I': 0., 'N': 0.}):
    """
    Experiment investigating the effect of denaturant (urea) concentration 
    on the folding of a two-domain protein with cooperative folding
    D <-> I <-> N
    D -- fully unfolded
    I -- single domain folded
    N -- both domains folded
    """
    if verbose:
    	print ("D <-> I <-> N\nD -- fully unfolded\nI -- single domain folded\nN -- both domains folded\n")
    	print ("Initial concentrations:\n" + '\n'.join('%s\t%.3f M' % (species, concs) for (species, concs) in concs.items()) + '\n')
    
    reacts = [('D->I', 26000*exp(-1.68*ureaconc)),
              ('I->D', 6e-2*exp(0.96*ureaconc)),
              ('I->N', 730*exp(-1.72*ureaconc)),
              ('N->I', 7.5e-4*exp(1.20*ureaconc))]
    results = np.zeros((len(ureaconc_range), 4))
    for i, ureaconc in enumerate(ureaconc_range):
        print ('Started run %i of %i at [urea] = %.3f M ' % (i+1, len(ureaconc_range), ureaconc))
        # rate constants given in the handout
        exper = Reactor(concs, reacts)
        exper.converge()
        results[i] = [ureaconc, exper.concs['D'], exper.concs['I'], exper.concs['N']]
    
    if outarr:    
        np.savetxt(outarr, results, delimiter=',')
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.plot(results[:,0], results[:,1], '-o', color='b', label='D')
    ax.plot(results[:,0], results[:,2], '-o', color='g', label='I')
    ax.plot(results[:,0], results[:,3], '-o', color='r', label='N')
    ax.set_ylabel("Fraction of species", fontsize=20)
    ax.set_xlabel("[Urea]/M", fontsize=20)
    ax.set_title("Equilibrium fraction of protein states against urea concentration", fontsize=20)
    plt.legend()
    if outfig:
        ax.tick_params(labelsize=20)
    plt.savefig(outfig)
    plt.show()

def BZ_experiment(outarr, outfig, verbose=False, concs={'A': 0.06, 'B': 0.06, 'P': 0, 'Q': 0, 'X': 10**-9.8, 'Y': 10**-6.52, 'Z': 10**-7.32}):
    """
    Experiment investigating oscillations of Belousov-Zhabotinsky reaction,
    using Oregonator model
    A+Y->X+P
    X+Y->P
    B+X->2X+Z
    2X->Q
    Z->Y
    """
    if verbose:
        print ("A+Y->X+P\nX+Y->P\nB+X->2X+Z\n2X->Q\nZ->Y\n")
        print ("Initial concentrations / M:\n" + '\n'.join('%s\t%.3f' % (species, concs) for (species, concs) in concs.items()) + '\n')
    # rate constants given in the handout
    reacts = [('A+Y->X+P', 1.34),
             ('X+Y->P', 1.6e9),
             ('B+X->2X+Z', 8e3),
             ('2X->Q', 4e7),
             ('Z->Y', 1)]
    results = np.zeros((1+480000, 1+3))
    results[0] = [0, concs['X'], concs['Y'], concs['Z']]
    exper = Reactor(concs, reacts)
    for step in range(480000):
        if not (step % 1000):
            print ('Started run %i of 480000' % (step+1))
        exper.jump_forward(500)
        results[step+1] = [step*500*exper.timestep, exper.concs['X'], exper.concs['Y'], exper.concs['Z']]
    
    if outarr:
        np.savetxt(outarr, results, delimiter=',')
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.semilogy(results[:, 0], results[:, 1], color='b', label='X')
    ax.semilogy(results[:, 0], results[:, 2], color='g', label='Y')
    ax.semilogy(results[:, 0], results[:, 3], color='r', label='Z')
    ax.set_ylabel("log[concentration / M]", fontsize=20)
    ax.set_xlabel("time / s", fontsize=20)
    ax.set_title("Oscillations of concentrations with time", fontsize=20)
    plt.legend()
    if outfig:
        plt.savefig(outfig)
    plt.show()

if __name__ == '__main__':            
    parser = argparse.ArgumentParser()
    parser.add_argument('-protein_folding', action='store_true', help='Run experiment investigating the effect of the concentration ' + \
   													 		 		  'of a denaturant (urea) on the folding of a two-domain protein that folds cooperatively')
    parser.add_argument('-BZ', action='store_true', help='Run experiment investigating oscillations of a Belousov-Zhabotinsky reaction using the Oregonator model')
    parser.add_argument('-outarr', default=None, help='Unix path to write a numpy array with intermediate concentrations of the run (optional)')
    parser.add_argument('-outfig', default=None, help='Unix path to save the resulting figure (optional')
    args = parser.parse_args()
    if args.protein_folding:
    	urea_folding_experiment(args.outarr, args.outfig, verbose=True)
    elif args.BZ:
    	BZ_experiment(args.outarr, args.outfig, verbose=True)
    else:
    	parser.print_help()