from typing import Tuple, Dict, List
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.random import choice

from .gates import Instruction, SWAP, Parametric


def get_ground_state(num_qubits: int) -> np.ndarray:
    """ Build the zero state given a fixed number of qubits.

    Args:
        num_qubits: number of qubits

    Returns:
        A vector of size 2**num_qubits with all zeroes except first instructionment which is 1.
    """
    vec = np.zeros(2**num_qubits)
    vec[0] = 1
    return vec


def get_operator(total_qubits: int, gate_unitary: np.ndarray, target_qubits: Tuple) -> np.ndarray:
    """Given a unitary operator, builds an operator to run on a specific set of contiguous qubits.

    Args:
        total_qubits: The total number of qubits that the new operator will adhere to.
        gate_unitary: The unitary operator to modify.
        target_qubits: The qubits that the unitary will operate on. These qubits must be strictly contiguous,
            i.e. 2, 3 or 4, 5 NOT 4, 6.

    Returns:
        A 2 ^ total_qubits x 2 ^ total_qubits operator.
    """
    # This formulation assumes that all numbers are sorted and consecutive.
    if len(target_qubits) > 1 and target_qubits == list(range(min(target_qubits), max(target_qubits) + 1)):
        raise ValueError(f'Target qubits must be sorted and consecutive. Got {target_qubits}')

    # If the number of states matches the number of rows of the gate, then return the matrix.
    if 2**total_qubits == gate_unitary.shape[0]:
        return gate_unitary

    # This is the smallest qubit in the list by construction.
    min_qubit_index = target_qubits[0]

    before = gate_unitary if min_qubit_index == 0 else np.kron(np.eye(2**min_qubit_index), gate_unitary)
    qubits_after = total_qubits - min_qubit_index - len(target_qubits)
    return np.kron(before, np.eye(2**(qubits_after)))

def preprocess_parametric(program: List[Instruction], feed_dict: Dict[str, complex]) -> List[Instruction]:
    """For all parametric instructions in the list, evaluate them given the feed_dict variables.

    Args:
        program: A list of instructions to parse.
        feed_dict: A mapping of string variables to complex replacements.

    Returns:
        A new list of instructions without any Parametric gates.
    """
    evaluate_vectorized = np.vectorize(lambda cell: complex(cell.evalf(subs=feed_dict)))
    ret = []
    for instruction in program:
        if isinstance(instruction, Parametric):
            unitary = evaluate_vectorized(instruction.unitary)
            ret.append(Instruction(targets     = instruction.targets,
                                   unitary     = unitary,
                                   commutative = instruction.commutative))
        else:
            ret.append(instruction)
    return ret

def preprocess_swaps(program: List[Instruction]) -> List[Instruction]:
    """Generate an equivalent list of constructions s.t. all gates have strictly contiguous inputs.

    This is accomplished by inserting swaps before and after a instruction that has operations on
    non contiguous wires. For example,

    [Op(3, 6)] -> [Swap(5, 6), Swap(4, 5), Op(3, 4), Swap(4, 5), Swap(5, 6)]

    In addition, this preprocessing will also add swaps to gates that have out-of-order inputs.

    [Op(4, 3)] -> [Swap(3, 4), Op(3, 4), Swap(3, 4)]

    And in the case where both of the above need to be done, the operations are compounded,

    [Op(5, 3)] -> [Swap(4, 5), Swap(3, 4), Op(3, 4), Swap(3, 4), Swap(4, 5)]

    This currently handles only two inputs, but could easily be extended to handle gates of any
    input size by generalizing the algorithm below. Note that any already valid instructions should
    be unaffected.

    [Op(3, 4)] -> [Op(3, 4)]

    Args:
        program: A list of gates.

    Returns:
        A new list of gates that is algebraically equivalent, but has strictly contiguous inputs.
    """
    ret = []
    for instruction in program:
        targets = instruction.targets

        if len(targets) == 1:
            # Single target instructions don't need any preprocessing.
            ret.append(instruction)
        elif len(targets) == 2:
            # Two target instructions might need some preprocessing.
            min_idx = min(targets)
            max_idx = max(targets)
            n_flips = max_idx - min_idx - 1

            # Bring the max index down to the min index. Note that this becomes a no-op if the
            # wires are successive. e.g. n_flips: 0 = max_idx: 5 - min_idx: 4 - 1
            for flip_idx in range(n_flips):
                ret.append(SWAP(max_idx - flip_idx - 1, max_idx - flip_idx))

            # All unitary operators assume wires that are in order (4, 5) != (5, 4). if that
            # If this is not true, then we need to flip the wires.
            invert_arguments = targets[0] > targets[1] and not instruction.commutative

            if invert_arguments:
                ret.append(SWAP(min_idx, min_idx + 1))

            # Finally! We can now add the gate that we have been trying to add this entire time.
            ret.append(Instruction((min_idx, min_idx + 1), instruction.unitary, instruction.commutative))

            # Flip the wires back if we flipped them previously.
            if invert_arguments:
                ret.append(SWAP(min_idx, min_idx + 1))

            # Send the max index back to the original spot.
            for flip_idx in reversed(range(n_flips)):
                ret.append(SWAP(max_idx - flip_idx - 1, max_idx - flip_idx))

        else:
            raise NotImplementedError('This simulator does not yet handle instructions with > 2 arguments.')
    return ret

def run_program(program: List[Instruction], n_qubits: int, initial_state: np.ndarray) -> np.ndarray:
    """Run a program given a list of instructions.

    Args:
        program: The list of instructions to use.
        n_qubits: The max number of qubits on the instruction.
        initial_state: The initial state of the simulation.

    Returns:
        The new state after running the program.
    """
    operator = np.eye(len(initial_state))
    for instruction in program:
        operator = operator @ get_operator(n_qubits, instruction.unitary, instruction.targets)
    return initial_state.dot(operator)

def _format_binary(num: int, padding: int) -> str:
    """Format a number in binary."""
    return format(num, f'#0{padding + 2}b')[2:]

def get_counts(state_vector: np.ndarray, num_shots: int) -> Dict[str, int]:
    """Run a monte-carlo simulation to sample a state vector.

    Args:
        state_vector: The state vector to sample.
        num_shots: The number of shots in the simulation.

    Returns:
        A dictionary of counts for each binary state.
    """
    # Technically if this is weighted by the same scalar, we don't need to normalize
    # if we really cared about efficiency.
    probs = np.abs(state_vector)**2 / np.linalg.norm(state_vector)**2
    states = [_format_binary(idx, int(np.log2(len(state_vector)))) for idx in range(len(state_vector))]
    samples = choice(states, num_shots, p = probs)
    counts = defaultdict(int)
    for sample in samples: counts[sample] += 1
    return dict(counts)

