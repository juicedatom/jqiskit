from typing import Optional, Dict, List

import numpy as np

from .gates import CX, CY, CZ, Hadamard, SWAP, SQRTNOT, Parametric
from .backend import run_program, get_ground_state, preprocess_swaps, preprocess_parametric, get_counts


class QuantumCircuit:
    """An object-oriented frontend to the more functional backend to allow a nicer api to build a circuit."""
    def __init__(self, n_qubits: int) -> None:
        # A list to hold the program contents.
        self.program = []

        if n_qubits <= 0:
            raise ValueError(
                f'Number of qubits must be strictly positive, got {n_qubits}')

        # The number of qubits this circuit is built for.
        self.n_qubits = n_qubits

    def __str__(self):
        """Helper function to print the circuit."""
        return '\n'.join(str(ele) for ele in self.program)

    def _check_bounds(self, index: int):
        """Check the bounds of a given index to make sure it's within the number of qubits.

        Args:
            index: index to check.

        Throws:
            throws if the index is not [0, n_qubits).
        """
        if index < 0 or index >= self.n_qubits:
            raise ValueError(f'Index out of bounds exception: {index}.')

    def h(self, target: int) -> None:
        """Add a h gate to the circuit.

        Args:
            target: qubit index to mutate.
        """
        self._check_bounds(target)
        self.program.append(Hadamard(target))

    def sqrtnot(self, target: int) -> None:
        """Add a SQRT NOT gate to the circuit.

        Args:
            target: qubit index to mutate.
        """
        self._check_bounds(target)
        self.program.append(SQRTNOT(target))

    def cx(self, control: int, target: int) -> None:
        """Add a cx gate to the circuit.

        Args:
            control: The control bit.
            target: The bit to mutate.
            
        """
        self._check_bounds(control)
        self._check_bounds(target)
        self.program.append(CX(control, target))

    def cy(self, control: int, target: int) -> None:
        """Add a cy gate to the circuit.

        Args:
            control: The control bit.
            target: The bit to mutate.
            
        """
        self._check_bounds(control)
        self._check_bounds(target)
        self.program.append(CY(control, target))

    def cz(self, control: int, target: int) -> None:
        """Add a cz gate to the circuit.

        Args:
            control: The control bit.
            target: The bit to mutate.
            
        """
        self._check_bounds(control)
        self._check_bounds(target)
        self.program.append(CZ(control, target))

    def swap(self, p: int, q: int) -> None:
        """Add a swap gate to the circuit.

        Args:
            p: The index of the qubit to swap with q.
            q: The index of the qubit to swap with p.
        """
        self._check_bounds(control)
        self._check_bounds(target)
        self.program.append(SWAP(p, q))

    def parametric(self, unitary_str: str, *targets: List[int]) -> None:
        """Add a parametric gate to the circuit.

        Args:
            unitary_str: A sympy-parsable string containing a valid unitary matrix.
            targets: The list of targets to fit into the unitary matrix.
        """
        for target in targets:
            self._check_bounds(target)
        self.program.append(Parametric(unitary_str, targets))

    def measure(self,
                num_shots: int = 1000,
                feed_dict: Optional[Dict[str, complex]] = None,
                initial_state: Optional[np.ndarray] = None) -> Dict[str, int]:
        """Perform a monte-carlo simulation of the current circuit.

        Args:
            num_shots: The number of shots in the simulation.
            feed_dict: A dictionary to feed any parameterized gates.
            initial_state: The initial state vector for the simulation. Assumes the 0 vector by default.

        Returns:
            A dictionary from states -> counts.
        """
        # Assume the ground state if the user does not give the initial state.
        if initial_state is None:
            initial_state = get_ground_state(self.n_qubits)

        # The initial state must fully enumerate all possible states.
        if 2**self.n_qubits != len(initial_state):
            raise ValueError(
                f'Invalid initial state, must have exactly 2^{self.n_qubits} elements., instead found {len(initial_state)}'
            )

        if num_shots < 0:
            raise ValueError(
                'Number of shots must be greater than 0. Instead, got {num_shots}'
            )

        # Preprocess the instructions s.t. they can be fed into the engine.
        processed = preprocess_parametric(self.program, feed_dict)
        processed = preprocess_swaps(processed)

        # For each gate, build each operator and perform the multiplication.
        final_state = run_program(processed, self.n_qubits, initial_state)

        # Run the monte-carlo simulation.
        return get_counts(final_state, num_shots), final_state
