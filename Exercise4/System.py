import numpy as np
from math import exp
import copy
import argparse
import sys

class System(object):
    """System modelling 3D configuration of particles and optimising it using a pairwise potential function

    Args:
        num_particles (int): Number of particles
        pfun (function): Pairwise potential function that takes in two Numpy arrays and returns a number  
        dim (int, default): Dimensionality of system

    Attributes:
        dim (int): Dimensionality of system
        num_particles (int): Number of particles
        pfun (function): Pairwise potential function
        particles (list of Numpy arrays): Current onfiguration of system
        potential (float): Current potential
        gradients (list of Numpy arrays): Gradients of each particle's potential
        lambd (float): Proportionality constant between a particle's gradient and its position update, 
            where new position = current position - gradient * lambd

    """

    def __init__(self, num_particles, pfun, dim=3):
        self.pfun = pfun
        self.num_particles = num_particles
        self.dim = dim
        # lambd is optimised by instance methods as the system is converging so it doesn't matter
        # if its initial value is too large
        self.lambd = 1e10
        # particles' positions are randomly initialised in a 1x1x1 cube
        self.particles = [np.random.rand(self.dim) for num in range(num_particles)]
        self.get_potential()
        self.get_numeric_grad()

    def get_potential(self):
        """Recomputes system's potential"""

        self.potential = 0.
        for ind, i in enumerate(self.particles):
            for j in self.particles[ind+1:]:
                self.potential += self.pfun(i, j)

    def get_numeric_grad(self, delta=1e-6):
        """Recmputes the gradient of a single particle's potential
        if the particle's coordinate is x, 1-dimensional grad = 1/(2*delta) * (potential at x+delta - potential at x-delta)

        Args:
            delta(float, default): numerical stepsize

        """
        def active_potential(particle, partners):
            """Computes potential of a single particle due to its interaction partners"""
            active_potential = 0.
            for partner in partners:
                active_potential += self.pfun(particle, partner)
            return active_potential 
        
        self.gradients = list()
        # delta_matrix is just an accessory matrix used for the ease of computing pot_plus and pot_minus
        delta_matrix = np.identity(self.dim) * delta 
        for i, particle in enumerate(self.particles):
            partners = self.particles[:i] + self.particles[(i+1):]
            gradient = np.zeros(self.dim)
            for ind in range(self.dim):
                pot_plus = active_potential(particle+delta_matrix[ind], partners)
                pot_minus = active_potential(particle-delta_matrix[ind], partners)
                gradient[ind] = 1/(2*delta) * (pot_plus - pot_minus)
            self.gradients.append(gradient)
            
    def _update(self):
        """Update positions of all particles, based on their gradients, and recompute system potential and gradients"""

        for particle, gradient in zip(self.particles, self.gradients):
            update = gradient*self.lambd
            # if the norm of gradient * lambd is more than 1., the particle experiences a strong repulsive potential 
            # when close to another particules, so, to avoid overshooting, limit the position update to 1. 
            # and reduce lambda by a factor of 2
            if np.linalg.norm(update) > 1.:
                update = update / np.linalg.norm(update) * 0.5
                self.lambd /= 2
            particle -= update
        self.get_potential()
        self.get_numeric_grad()

    def converge(self, convergence_limit=1e-4, verbose=True):
        """Update system's configuration until a (local) minimum is reached

        Args:
            convergence_limit (float, default): if all particles change their position by less than this value
                and enough steps have been taken, declare converge
            verbose (bool, default): flag to print intermediate steps during iteration
                
        """
        frac_change = float("inf")
        steps = 0
        
        def converged():
            updates = (np.linalg.norm(grad) for grad in self.gradients)
            return all(abs(update) < convergence_limit for update in updates)
        
        # convergence declared when the fractional change in energy is less than convergence_limit and when at least 500 steps have been taken
        # the second convergence criterion ensures that we don't declare convergence when the particles haven't yet had a chance to approach each other
        # as is sometimes the case at the start of the iteration     
        while not converged() or (steps < 500): 
            # store copy of the systems configuration to backtrack if overshot the minimum
            memory = copy.deepcopy(self.__dict__)
            self._update()
            frac_change = (self.potential - memory['potential']) / abs(memory['potential'])
            if frac_change > 0:
                # backtrack and reduce lambda by a factor of 2
                self.__dict__ = memory
                self.lambd /= 2
            else:
                if (steps % 100 == 0) and verbose:
                    print ('{:<4} steps: U = {:.3E}, \u0394U / U = {:.3E}'.format(steps, self.potential, frac_change))
                steps += 1
        
        if verbose:
            print ('\nConverged after %i steps' % steps)
            print ('Equilibrium U = %.3E' % self.potential)
    
    def global_converge(self, num_trials=10, convergence_limit=1e-4, verbose=True):
        trials = list()
        for i in range(num_trials):
            self.__init__(self.num_particles, self.pfun)
            self.converge(convergence_limit=convergence_limit, verbose=False)
            print ('Trial #{:<2}: U = {:.3E}'.format(i+1,self.potential))
            trials.append(copy.deepcopy(self.__dict__))

        best_trial = min(trials, key=lambda trial: trial['potential'])
        self.__dict__ = best_trial
        if verbose:
            print ('Optimal U = %.3E' % self.potential)
            
    def to_XYZ(self, path):
        """Write system configuration in XYZ format at location specified by path

        Args:
            path (str): Unix path to location where to write configuration
            
        """
        with open(path, 'w') as f:
            f.write(str(self.num_particles) + '\n')
            f.write("Geometry of system \n")
            for particle in self.particles:
                # the element does not matter
                f.write("He %.4f %.4f %.4f\n" % (particle[0], particle[1], particle[2]))

    def __str__(self):
        if self.dim != 3:
            print ("Can't print for dimensions other than 3")
            return
        output = 'particle\tx\ty\tz\n'
        for i, particle in enumerate(self.particles):
            output += "%i\t\t%.3f\t%.3f\t%.3f\n" % (i+1, particle[0], particle[1], particle[2])
        return output

    @staticmethod
    def Lennard_Jones(rm=0.5, epsilon=1):
        """Returns pairwise Lennard_Jones potential function with predefined parameters
        
        Agrs:
            rm (float, default): Equilibrium separation
            epsilon (float, default): Energy at equilibrum separation
            
        """
        def Lennard_Jones_pfun(nparr1, nparr2):
            r = np.linalg.norm(nparr1 - nparr2)
            # the potential is epsilon*((rm / r)**12 - 2*(rm / r)**6), but it's preferrable
            # to compute (rm / r)**12 by squaring the second term
            frac_exp = (rm / r)**6
            return epsilon*(frac_exp**2 - 2*frac_exp)
        return Lennard_Jones_pfun

    @staticmethod
    def Morse(sigma=1, De=1, re=0.5):
        """Returns pairwise Morse potential function with predefined parameters
        
        Args:
            De (float, default): Energy at infinite separation
            re (float, default): Equilibrium separation
            sigma (float, default): Paramater determining the steepness of the potential well
            
        """
        def Morse_pfun(nparr1, nparr2):
            r = np.linalg.norm(nparr1 - nparr2)
            return De*(1 - exp(-(r-re)/sigma))**2
        return Morse_pfun
    
    @staticmethod
    def sample_output(direc):
        """Writing optimised configurations for a range of particles for manual inspection and unittesting
        
        Args:
            direc (str): Unix path to directory where to write outputs
            
        """ 
        for n in [2,3,4,6,7,8]:
            system = System(n, System.Morse(sigma=1, De=1, re=0.5))
            system.converge(verbose=False)
            system.to_XYZ(direc + '/Morse_%i.xyz' % n)
        for n in [12,20]:
            system = System(n, System.Morse(sigma=1, De=1, re=0.5))
            system.converge_global(verbose=False)
            system.to_XYZ(direc + '/Morse_%i.xyz' % n)
        # convergence with LJ potential and n > 10 is slow 
        for n in [2,3,4]:
            system = System(n, System.Lennard_Jones(rm=0.5, epsilon=1))
            system.converge(verbose=False)
            system.to_XYZ(direc + '/LJ_%i.xyz' % n)  
        for n in [7,8]:
            system = System(n, System.Lennard_Jones(rm=0.5, epsilon=1))
            system.converge_global(verbose=False)
            system.to_XYZ(direc + '/LJ_%i.xyz' % n)              

if __name__ == '__main__':
    class MyParser(argparse.ArgumentParser):
    # overriding error method to print help message
    # when script is called without arguments (default was to print error)
        def error(self, message):
                self.print_help()
                sys.exit()
    
    parser = MyParser()
    parser.add_argument('-n', help="Number of particles in the system", type=int)
    parser.add_argument('-p', '--potential', help="Potential for optimisation, can be 'Morse' or 'LJ' (i.e. Lennard-Jones)", type=str)
    parser.add_argument('-g', '--globally',  help='Optimise system 10 times to find global minimum (recommended for Lennard_Jones potential with n > 6)', action='store_true')
    parser.add_argument('-o', '--outfile', help="Unix path to .xyz file where to write system's configuration(optional)")
    args = parser.parse_args()

    if args.n == None:
        parser.print_help()
        sys.exit()
    if args.potential == 'Morse': 
        potential = System.Morse()
    elif args.potential == 'LJ':
        potential = System.Lennard_Jones()
    else:
        raise ValueError('Only Morse and LJ potentials are supported')
    system = System(args.n, potential)
    if args.globally:
        system.global_converge()
    else:
        system.converge()
    if args.o:
        system.to_XYZ(args.o)
        print ("Printed XYZ file")
    print ()