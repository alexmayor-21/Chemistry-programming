import numpy as np
from math import exp
from decimal import Decimal
from pprint import pprint
import copy
import argparse
import sys

class System(object):
    """System modelling 3D configuration of particles and optimising it using a pairwise potential function

    Args:
        num_particles (int): Number of particles
        pfun (function): Pairwise potential function 
        particles (list of Numpy arrays, optional): Predefined starting configuration
        dim (int, default): Dimensionality of system
        lambd (float, default): Proportionality constant between a particle's gradient and its position update, 
            where new position = current position - gradient * lambd

    Attributes:
        dim (int): Dimensionality of system
        num_particles (int): Number of particles
        pfun (function): Pariwise potential function
        particles (list of Numpy arrays): Current onfiguration of system
        potential (float): Current potential
        gradients (list of Numpy arrays): Gradients of each particle's potential
        lambd (float): Proportionality constant for updating a particle's position,
            note that lambd is optimised by bound methods during iteration

    """
    
    def __init__(self, num_particles, pfun, particles=None, dim=3, lambd=1e2):
        self.pfun = pfun
        self.num_particles = num_particles
        self.dim = dim
        self.lambd = lambd
        if particles:
            self.particles = particles
        else:
            # particles' positions are initiated randomly in a 1x1x1 cube
            self.particles = [np.random.rand(self.dim)
                              for num in range(num_particles)]
        self.get_potential()
        self.get_numeric_grad()

    def get_potential(self):
        """Recomputes system's potential"""
        self.potential = 0.
        for ind, i in enumerate(self.particles):
            for j in self.particles[ind+1:]:
                self.potential += self.pfun(i, j)

    def get_active_potential(self, particle, partners):
        """Computes potential of a single particle due to its interaction partners

        Args:
            particle (Numpy array): position of particle
            partners (list of Numpy arrays): configuration of interaction partners

        Returns:
            active_potential (float): potential

        """
        active_potential = 0.
        for partner in partners:
            active_potential += self.pfun(particle, partner)
        return active_potential

    def get_numeric_grad(self, delta=1e-6):
        """Recmputes the gradient of a single particle's potential
        if the particle's coordinate is x, 1-dimensional grad = 1/(2*delta) * (potential at x+delta - potential at x-delta)

        Args:
            delta(float, default) -- numerical stepsize

        """
        self.gradients = list()
        delta_matrix = np.identity(self.dim) * delta 
        for i, particle in enumerate(self.particles):
            partners = self.particles[:i] + self.particles[(i+1):]
            gradient = np.zeros(self.dim)
            for ind in range(self.dim):
                pot_plus = self.get_active_potential(particle+delta_matrix[ind], partners)
                pot_minus = self.get_active_potential(particle-delta_matrix[ind], partners)
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

    def converge(self, convergence_limit=1e-6):
        """Update system's configuration until a (local) minimum is reached

        Args:
            convergence_limit (float, default) -- the smallest fractional change in energy at which to declare convergence
                if enough steps have been taken

        """
        frac_change = float("inf")
        steps = 1
        
        # convergence declared when the fractional change in energy is less than convergence_limit and when at least 500 steps have been taken
        # the second convergence criterion ensures that we don't declare convergence when the particles haven't yet had a chance to approach each other
        # as is sometimes the case at the start of the iteration     
        while (abs(frac_change) > convergence_limit) or (steps < 500): 
            # store copy of the systems configuration to backtrack if overshot the minimum
            memory = copy.deepcopy(self.__dict__)
            self._update()
            frac_change = (self.potential - memory['potential']) / abs(memory['potential'])
            if frac_change > 0:
                # backtrack and reduce lambda by a factor of 2
                self.__dict__ = memory
                self.lambd /= 2
            else:
                if (steps % 100 == 0):
                    print (u'%i steps: U = %.3E, \u0394U / U = %.3E' % (steps, self.potential, frac_change))
                steps += 1

        print ('\nConverged after %i steps' % steps)
        print ('Equilibrium U = %.3E' % self.potential)
        print ('Printed XYZ file\n')
    
    def to_XYZ(self, path):
        """
        Write system configuration in XYZ format at location specified by path

        Args:
            path (str) -- Unix path to location where to write configuration
        """
        with open(path, 'w') as f:
            f.write(str(self.num_particles) + '\n')
            f.write("Geometry of system \n")
            for i, particle in enumerate(self.particles):
                f.write("C %.4f %.4f %.4f\n" % (particle[0], particle[1], particle[2]))

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
        """
        Returns pairwise Lennard_Jones potential function with predefined parameters
        rm -- equilibrium separation
        epsilon -- energy at equilibrum separation
        """
        def Lennard_Jones_pfun(nparr1, nparr2):
            r = np.linalg.norm(nparr1 - nparr2)
            return epsilon*((rm / r)**12 - 2*(rm / r)**6)
        return Lennard_Jones_pfun

    @staticmethod
    def Morse(sigma=1, De=1, re=0.5):
        """
        Returns pairwise Morse potential function with predefined parameters
        De -- energy at infinite separation
        re -- equilibrium separation
        sigma -- paramater determining the steepness of the potential well
        """
        def Morse_pfun(nparr1, nparr2):
            r = np.linalg.norm(nparr1 - nparr2)
            return De*(1 - exp(-(r-re)/sigma))**2
        return Morse_pfun               

if __name__ == '__main__':
	class MyParser(argparse.ArgumentParser):
	# overriding error method to print help message
	# when script is called without arguments (default was to print error)
	    def error(self, message):
	    	self.print_help()
	    	sys.exit()

	parser = MyParser()
	parser.add_argument('-n', help="Number of particles in the system", type=int)
	parser.add_argument('-potential', help="Potential for optimisation, can be 'Morse' or 'LJ' (i.e. Lennard-Jones)", type=str)
	parser.add_argument('-o', help="Unix path to folder where to write .xyz file (optional)")
	args = parser.parse_args()
	if args.n == None:
		parser.print_help()
		sys.exit()
	if args.potential == 'Morse':
		sys = System(args.n, System.Morse())
		sys.converge()
	elif args.potential == 'LJ':
		sys = System(args.n, System.Lennard_Jones())
		sys.converge()
	else: 
		raise ValueError('Only Morse and LJ potentials are supported')
	if args.o:
		sys.to_XYZ(args.o)

