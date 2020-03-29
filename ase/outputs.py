from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Sequence, Union

import numpy as np


class Properties(Mapping):
    def __init__(self):
        self._dct = {}

    def _get_dimension(self, name):
        return self._dct.get(name)

    def __len__(self):
        return len(self._dct)

    def __iter__(self):
        return iter(self._dct)

    def __getitem__(self, name):
        return self._dct[name]

    def setvalue(self, name, value):
        if name in self._dct:
            # Which error should we raise for already existing property?
            raise ValueError(f'{name} already set')

        prop = all_properties[name]
        value = prop.normalize(value)
        shape = np.shape(value)

        if not prop.shape_is_consistent(self, value):
            shape = np.shape(value)
            raise ValueError(f'{name} has bad shape: {shape}')

        for i, spec in enumerate(prop.shapespec):
            if not isinstance(spec, str) or spec in self._dct:
                continue
            self.setvalue(spec, shape[i])

        self._dct[name] = value

    def __repr__(self):
        clsname = type(self).__name__
        return f'({clsname}({self._dct})'


all_properties = {}


class Property(ABC):
    def __init__(self, name, dtype, shapespec):
        self.name = name
        assert dtype in [float, int]  # Others?
        self.dtype = dtype
        self.shapespec = shapespec

    @abstractmethod
    def normalize(self, value):
        ...


    def shape_is_consistent(self, properties: Properties, value) -> bool:
        """Return whether shape of values is consistent with properties.

        For example, forces of shape (7, 3) are consistent
        unless properties already have "natoms" with non-7 value.
        """
        shapespec = self.shapespec
        shape = np.shape(value)
        if len(shapespec) != len(shape):
            return False
        for dimspec, dim in zip(shapespec, shape):
            if isinstance(dimspec, str):
                dimspec = properties._dct.get(dimspec, dim)
            if dimspec != dim:
                return False
        return True

    def __repr__(self) -> str:
        typename = {float: 'float', int: 'int'}[self.dtype]
        shape = ', '.join(str(dim) for dim in self.shapespec)
        return f'Property({self.name!r}, dtype={typename}, shape=[{shape}])'


class ScalarProperty(Property):
    def __init__(self, name, dtype):
        super().__init__(name, dtype, tuple())

    def normalize(self, value):
        if not np.isscalar(value):
            raise TypeError('Expected scalar')
        return self.dtype(value)


class ArrayProperty(Property):
    def normalize(self, value):
        if np.isscalar(value):
            raise TypeError('Expected array, got scalar')
        return np.asarray(value, dtype=self.dtype)


ShapeSpec = Union[str, int]


def defineprop(
        name: str,
        dtype: type = float,
        shape: Union[ShapeSpec, Sequence[ShapeSpec]] = tuple()
) -> Property:
    """Create, register, and return a property."""

    if isinstance(shape, (int, str)):
        shape = (shape,)

    shape = tuple(shape)
    if len(shape) == 0:
        prop = ScalarProperty(name, dtype)
    else:
        prop = ArrayProperty(name, dtype, shape)
    all_properties[name] = prop
    return prop


# Atoms, energy, forces, stress:
defineprop('natoms', int)
defineprop('energy', float)
defineprop('energies', float, shape='natoms')
defineprop('free_energy', float)
defineprop('forces', float, shape=('natoms', 3))
defineprop('stress', float, shape=6)
defineprop('stresses', float, shape=('natoms', 6))

# Electronic structure:
defineprop('nbands', int)
defineprop('nkpts', int)
defineprop('nspins', int)
defineprop('fermi_level', float)
defineprop('kpoint_weights', float, shape='nkpts')
defineprop('ibz_kpoints', float, shape=('nkpts', 3))
defineprop('eigenvalues', float, shape=('nspins', 'nkpts', 'nbands'))
defineprop('occupations', float, shape=('nspins', 'nkpts', 'nbands'))

# We might want to allow properties that are part of Atoms, such as
# positions, numbers, pbc, cell.  It would be reasonable for those
# concepts to have a formalization outside the Atoms class.



#def to_singlepoint(self, atoms):
#    from ase.calculators.singlepoint import SinglePointDFTCalculator
#    return SinglePointDFTCalculator(atoms,
#                                    efermi=self.fermi_level,

# We can also retrieve (P)DOS and band structure.  However:
#
# * Band structure may require bandpath, which is an input, and
#   may not necessarily be easy or possible to reconstruct from
#   the outputs.
#
# * Some calculators may produce the whole BandStructure object in
#   one go (e.g. while parsing)
#
# * What about HOMO/LUMO?  Can be obtained from
#   eigenvalues/occupations, but some codes provide real data.  We
#   probably need to distinguish between HOMO/LUMO inferred by us
#   versus values provided within the output.
#
# * HOMO is sometimes used as alternative reference energy for
#   band structure.
#
# * What about spin-dependent (double) Fermi level?
#
# * What about 3D arrays?  We will almost certainly want to be
#   connected to an object that can load dynamically from a file.
