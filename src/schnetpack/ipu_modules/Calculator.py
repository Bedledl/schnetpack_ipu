from typing import Union, List, Dict

import poptorch
import torch

from schnetpack.md import System

from schnetpack import AtomisticTask, properties
from schnetpack.md.calculators import MDCalculator


class IPUCalculator(MDCalculator):
    def __init__(
            self,
            model: AtomisticTask,
            force_key: str,
            energy_unit: Union[str, float],
            position_unit: Union[str, float],
            energy_key: str = None,
            stress_key: str = None,
            required_properties: List = [],
            property_conversion: Dict[str, Union[str, float]] = {},
            run_on_ipu=True,
            n_atoms=None,
            n_molecules=None,
            n_neighbors=None
    ):
        super(IPUCalculator, self).__init__(
            required_properties,
            force_key,
            energy_unit,
            position_unit,
            energy_key,
            stress_key,
            property_conversion
        )
        if run_on_ipu:
            self.model = poptorch.inferenceModel(model)
        else:
            self.model = model

        self.n_atoms = n_atoms
        self.n_molecules = n_molecules
        self.k = n_neighbors

    def calculate(self, system: System):
        inputs = self._get_system_molecules(system)
        self.results = self.model(inputs)
        print(self.results)
        self._update_system(system)

    def _get_system_molecules(self, system: System):
        inputs = super(IPUCalculator, self)._get_system_molecules(system)
        inputs[properties.n_molecules] = system.n_replicas
        if self.n_atoms and self.n_molecules and self.k:
            inputs[properties.idx_i] = torch.arange(self.n_atoms * self.n_molecules)\
                .repeat_interleave(self.k)
            inputs[properties.offsets] = torch.tensor([[0, 0, 0]])\
                .repeat(self.n_atoms * self.n_molecules * self.k, 1)
        return inputs
