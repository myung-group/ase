"""Test CombineMM forces by combining tip3p and tip4p with them selves, and
   by combining tip3p with tip4p and testing againts numerical forces """
from math import cos, sin, pi
import numpy as np
from ase import Atoms
from ase.calculators.tip3p import TIP3P, rOH, angleHOH
from ase.calculators.tip4p import TIP4P
from ase.calculators.combine_mm import CombineMM
from ase.calculators.qmmm import LJInteractionsGeneral
from ase.calculators.tip3p import epsilon0 as eps3
from ase.calculators.tip3p import sigma0 as sig3
from ase.calculators.tip4p import epsilon0 as eps4
from ase.calculators.tip4p import sigma0 as sig4

def make_atoms():
    r = rOH
    a = angleHOH * pi / 180
    dimer = Atoms('H2OH2O',
		  [(r * cos(a), 0, r * sin(a)),
		   (r, 0, 0),
		   (0, 0, 0),
		   (r * cos(a / 2), r * sin(a / 2), 0),
		   (r * cos(a / 2), -r * sin(a / 2), 0),
		   (0, 0, 0)])

    dimer = dimer[[2, 0, 1, 5, 3, 4]]
    # put O-O distance in the cutoff range
    dimer.positions[3:, 0] += 2.8

    return dimer


dimer = make_atoms()
rc = 3
for (TIPnP, (eps, sig), nm) in zip([TIP3P, TIP4P], 
                                   ((eps3, sig3),(eps4, sig4)),
                                   [3, 4]):
    dimer.calc = TIPnP(rc=rc, width=1.0)
    F1 = dimer.get_forces()

    lj = LJInteractionsGeneral(np.array([sig, 0, 0]), 
                               np.array([eps, 0, 0]),
                               np.array([sig, 0, 0]),
                               np.array([eps, 0, 0]), 
                               3, 3, rc, 1.0)
    dimer.calc = CombineMM([0, 1, 2], nm, nm, 
                           TIPnP(rc=rc, width=1.0), 
                           TIPnP(rc=rc, width=1.0),
                           lj, rc=rc, width=1.0)
    F2 = dimer.get_forces()
    Fn = dimer.calc.calculate_numerical_forces(dimer)
    dF = F2-Fn
    dF = F1-F2
    print(TIPnP)
    print(dF)
    assert abs(dF).max() < 2e-6



