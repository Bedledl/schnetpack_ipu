import numpy as np
import torch
from schnetpack import units as spk_units
from schnetpack.md import Simulator
from schnetpack.md.simulation_hooks import ThermostatHook


class MultiTempLangevinThermostat(ThermostatHook):
    def __init__(self, temperatures: torch.Tensor, time_constant: float):
        super(MultiTempLangevinThermostat, self).__init__(0, time_constant)
        self.temperatures = temperatures

        self.register_uninitialized_buffer("thermostat_factors")
        self.register_uninitialized_buffer("c1")
        self.register_uninitialized_buffer("c2")

    def _init_thermostat(self, simulator: Simulator):
        """
        Initialize the Langevin coefficient matrices based on the system and simulator properties.

        Args:
            simulator (schnetpack.simulator.Simulator): Main simulator class containing information on the time step,
                                                        system, etc.
        """
        replicas = len(self.temperatures)
        if replicas != simulator.system.n_replicas:
            raise ValueError(f"This thermostat has {replicas} number of temperatures "
                             f"but the system has {simulator.system.n_replicas} replicas."
            )

        masses = simulator.system.masses

        # Get mass and temperature factors
        self.thermostat_factors = torch.sqrt(
            masses * spk_units.kB * self.temperatures.view(-1, 1, 1)
        )

        # Initialize friction coefficients
        gamma = (
                torch.ones(1, device=simulator.device, dtype=simulator.dtype)
                / self.time_constant
        )

        # Initialize coefficient matrices
        c1 = torch.exp(-0.5 * simulator.integrator.time_step * gamma)
        c2 = torch.sqrt(1 - c1 ** 2)

        self.c1 = c1[:, None, None]
        self.c2 = c2[:, None, None]

    def _apply_thermostat(self, simulator):
        # Apply the Langevin thermostat to each replica with its corresponding temperature
        for replica_idx in range(len(self.temperatures)):
            # Apply the Langevin thermostat to the replica using its temperature
            self.__apply_langevin(simulator, replica_idx)

    def __apply_langevin(self, simulator, replica_idx):
        momenta = simulator.system.momenta[replica_idx]
        thermostat_factor = self.thermostat_factors[replica_idx]

        # Generate random noise
        thermostat_noise = torch.randn_like(momenta)

        # Apply thermostat
        momenta = (
                self.c1 * momenta + thermostat_factor * self.c2 * thermostat_noise
        )

        simulator.system.momenta[replica_idx] = momenta
