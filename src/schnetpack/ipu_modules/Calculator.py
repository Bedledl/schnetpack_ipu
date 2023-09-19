from functools import partial
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
            n_neighbors: int,
            energy_key: str = None,
            stress_key: str = None,
            required_properties: List = [],
            property_conversion: Dict[str, Union[str, float]] = {},
            run_on_ipu=True,
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
            model.eval()
            model.to(torch.float32)
            self.model = poptorch.inferenceModel(model)
        else:
            self.model = model

        self.n_neighbors = n_neighbors

    def calculate(self, system: System):
        inputs = self._get_system_molecules(system)
        self.results = self.model(inputs)
        self._update_system(system)

    def _get_system_molecules(self, system: System):
        inputs = super(IPUCalculator, self)._get_system_molecules(system)
        inputs[properties.n_molecules] = system.n_replicas
        inputs[properties.idx_i] = torch.arange(system.total_n_atoms)\
            .repeat_interleave(self.n_neighbors)
        inputs[properties.offsets] = torch.tensor([[0, 0, 0]])\
            .repeat(system.total_n_atoms * self.n_neighbors, 1)
        return inputs


class BenchmarkCalculator(IPUCalculator):
    """
    This calculator is only for running benchmarks.
    """
    def compile_model(self, system):
        """
        This method compiles the model with the inputs given by the system.
        """
        inputs = self._get_system_molecules(system)
        self.model.compile(inputs)

    def get_model_call(self, system):
        """This method returns a callable that runs the model and that can be used for a benchmark"""
        inputs = self._get_system_molecules(system)
        return partial(self.model.forward, inputs)
