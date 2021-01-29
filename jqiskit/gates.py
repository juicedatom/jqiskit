from dataclasses import dataclass
from typing import Tuple, List

from sympy.parsing.sympy_parser import parse_expr
import numpy as np

@dataclass(frozen=True)
class Instruction:
    """Immutable function to hold instruction information."""

    # The bits that this instruction operates on.
    targets: Tuple
    # The unitary operator of the instruction.
    unitary: np.array
    # f(x, y) == f(y, x)?
    commutative: bool

    def __str__(self):
        """Pretty-print instruction."""
        return f'{self.__class__.__name__}: {self.targets}'

    def __post_init__(self):
        """Check if the gate is unitary."""
        conj = np.conjugate(self.unitary)
        left = np.allclose(self.unitary @ conj.T, np.eye(self.unitary.shape[0]))
        right = np.allclose(conj.T @ self.unitary, np.eye(self.unitary.shape[0]))
        if not (left and right):
            raise ValueError('Gate matrix not unitary!')

class Hadamard(Instruction):
    """Implementation of the Hadamard Gate."""

    def __init__(self, target: int) -> None:
        """Create the gate.

        Args:
            target: The bit to mutate.
        """
        unitary = 1 / np.sqrt(2) * np.array([
            [1,  1],
            [1, -1]
        ])
        super().__init__((target,), unitary, True)

class SQRTNOT(Instruction):
    """Implementation of the square root of the cx gate."""

    def __init__(self, target: int) -> None:
        """Build the gate.

        Args:
            target: The bit to mutate.
            
        """
        unitary = 1 / 2 * np.array([
            [1 + 1j,  1 - 1j],
            [1 - 1j,  1 + 1j]
        ])
        super().__init__((target,), unitary, True)

class CX(Instruction):
    """Implementation of the CNOT gate."""

    def __init__(self, control, target) -> None:
        """Build the gate.

        Args:
            control: The control bit.
            target: The bit to mutate.
            
        """
        unitary = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        super().__init__((control, target), unitary, False)

class CY(Instruction):
    """Implementation of the controlled pauli-y gate."""

    def __init__(self, control, target) -> None:
        """Build the gate.

        Args:
            control: The control bit.
            target: The bit to mutate.
            
        """
        unitary = np.array([
            [1, 0,  0,   0],
            [0, 1,  0,   0],
            [0, 0,  0, -1j],
            [0, 0, 1j,   0]
        ])
        super().__init__((control, target), unitary, False)

class CZ(Instruction):
    """Implementation of the controlled pauli-z gate."""

    def __init__(self, control, target) -> None:
        """Build the gate.

        Args:
            control: The control bit.
            target: The bit to mutate.
            
        """
        unitary = np.array([
            [1, 0, 0,  0],
            [0, 1, 0,  0],
            [0, 0, 1,  0],
            [0, 0, 0, -1]
        ])
        super().__init__((control, target), unitary, False)

class SWAP(Instruction):
    """Implementation of the SWAP gate."""

    def __init__(self, p: int, q: int) -> None:
        """ Build the gate.

        Args:
            p: bit to swap with q.
            q: bit to swap with p.
        """
        unitary = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        super().__init__((p, q), unitary, True)


class CSWAP(Instruction):
    """Implementation of the SWAP gate."""

    def __init__(self, control: int, p: int, q: int):
        """ Build the gate.

        Args:
            control: bit to control the flip.
            p: bit to swap with q.
            q: bit to swap with p.
        """
        unitary = np.array([
            [1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1.],
        ])
        super().__init__((control, p, q), unitary, True)

class Parametric(Instruction):
    def __init__(self, unitary_str: str, targets: List[int]):
        unitary = np.array(parse_expr(unitary_str))

        if len(unitary.shape) != 2:
            raise ValueError('Unitary matrix must be two dimensional.')

        if unitary.shape[0] != unitary.shape[1]:
            raise ValueError('Unitary matrix must be square!')

        n_args = np.log2(unitary.shape[0])

        if not np.isclose(n_args, np.round(n_args)):
            raise ValueError('Invalid dimensions.')

        n_args = int(n_args)
        print(n_args)

        if len(targets) != n_args:
            raise ValueError('Must include the same number of targets as required by the unitary matrix.')

        super().__init__(targets, unitary, n_args == 1)

    def __post_init__(self) -> None:
        """We override the post-init check to be unitary until it becomes a fully-realized instruction."""
