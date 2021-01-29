from typing import Tuple, Dict, List, Iterable
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


def preprocess_parametric(program: List[Instruction],
                          feed_dict: Dict[str, complex]) -> List[Instruction]:
    """For all parametric instructions in the list, evaluate them given the feed_dict variables.

    Args:
        program: A list of instructions to parse.
        feed_dict: A mapping of string variables to complex replacements.

    Returns:
        A new list of instructions without any Parametric gates.
    """
    evaluate_vectorized = np.vectorize(
        lambda cell: complex(cell.evalf(subs=feed_dict)))
    ret = []
    for instruction in program:
        if isinstance(instruction, Parametric):
            unitary = evaluate_vectorized(instruction.unitary)
            ret.append(
                Instruction(targets=instruction.targets,
                            unitary=unitary,
                            commutative=instruction.commutative))
        else:
            ret.append(instruction)
    return ret


def _generate_swap_indices(targets: Iterable[int]) -> List[Tuple[int, int]]:
    """Given a list of indices, return the list of swaps required to move all indices towards the lowest index in the given order.

    For example, let us assume that we are given [3, 6]. Then in an array, this would look like the following,

                  _ _ _ 0 _ _ 1
    init          0 1 2 3 4 5 6

    the '3' index will remain where it is, as it is first in the given order. The '1' must be moved over to the
    '0', or the 3 index. To do that, we perform the following swaps,

                  _ _ _ 0 _ 1 _ 
    swap[5, 6]    0 1 2 3 4 5 6

                  _ _ _ 0 1 _ _ 
    swap[4, 5]    0 1 2 3 4 5 6

    Therefore we would return [(5, 6), (4, 5)]

    Here's another example. What if the 3, 6 were swapped? Meaning, our input looked like [6, 3]

                  _ _ _ 1 _ _ 0
    init          0 1 2 3 4 5 6

    As before, we look for the lowest value (0) and move it towards the lowest index (3)

                  _ _ _ 1 _ 0 _ 
    swap[5, 6]    0 1 2 3 4 5 6

                  _ _ _ 1 0 _ _ 
    swap[4, 5]    0 1 2 3 4 5 6

                  _ _ _ 0 1 _ _ 
    swap[3, 4]    0 1 2 3 4 5 6

    Since the 1 is already in the correct position, all we need to return is 

        [(5, 6), (4, 5), (3, 4)]


    Args:
        targets: An ordered iterable of indices [i_1, i_2, ..., i_n] that are meant to appear in the order given.

    Returns:
        A list of tuples containing flip instructions. All indices within the tuples are guaranteed to be of the form
        (a, a + 1) where a is an integer greater than or equal to 0.
    """
    swap_list = []
    min_target = min(targets)
    max_target = max(targets)
    
    offset = max_target - min_target
    tmp = np.full((max_target - min_target + 1, ), np.nan)
    
    for idx, target in enumerate(targets):
        tmp[target - min_target] = idx
    for idx in range(len(targets)):
        tmp_idx = np.where(tmp == idx)[0][0]
        for jdx in reversed(range(idx + 1, tmp_idx + 1)):
            swap_list.append((jdx - 1 + min_target, jdx + min_target))
            tmp[jdx], tmp[jdx - 1] =  tmp[jdx - 1], tmp[jdx]
            
    return swap_list

def preprocess_swaps(program: Iterable[Instruction]) -> List[Instruction]:
    """Generate an equivalent list of constructions s.t. all gates have strictly contiguous inputs.
 
    If all the operators have striclty contiguous inputs, then it becomes easier to generate
    operations on them using simple rules like I x A x I x I, etc...

    This is accomplished by inserting swaps before and after a instruction that has operations on
    non contiguous wires. For example,
 
    [Op(5, 3)] -> [Swap(4, 5), Swap(3, 4), Op(3, 4), Swap(3, 4), Swap(4, 5)]
 
    # This will also leave the base case as a no-op.
    [Op(3, 4)] -> [Op(3, 4)]
 
    Args:
        program: A list of gates.
 
    Returns:
        A new list of gates that is algebraically equivalent, but has strictly contiguous inputs.
    """
    ret = []
    for instruction in program:
        # Grab the min target for reference.
        min_target = min(instruction.targets)
        # Generate a list of swap indices.
        swap_indices = _generate_swap_indices(instruction.targets)
        # Convert those swap indices into SWAP operations.
        swaps = [SWAP(idx, jdx) for (idx, jdx) in swap_indices]
        # Assuming the swapping will work, the new instruction should be correctly contiguous.
        new_instruction_targets = tuple(range(min_target, min_target + len(instruction.targets)))
        # Build the new operator.
        op = Instruction(new_instruction_targets, instruction.unitary, instruction.commutative)
        # The new set of instructions will swap the gates s.t. they line up, then run the operator, and
        # finally undo what it just did.
        ret += swaps + [op] + list(reversed(swaps))
    return ret


def get_operator(total_qubits: int, instruction: Instruction) -> np.ndarray:
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
    if len(instruction.targets) > 1 and not np.array_equal(
            instruction.targets,
            list(range(min(instruction.targets),
                       max(instruction.targets) + 1))):
        raise ValueError(
            f'Target qubits must be sorted and consecutive. Got {instruction.targets}'
        )

    # Make sure that the number of qubits is less tahn the given indices.
    if max(instruction.targets) >= total_qubits:
        raise IndexError('Index out of bounds exception.')

    # If the number of states matches the number of rows of the gate, then return the matrix.
    if 2**total_qubits == instruction.unitary.shape[0]:
        return instruction.unitary

    # This is the smallest qubit in the list by construction.
    min_qubit_index = instruction.targets[0]

    before = instruction.unitary if min_qubit_index == 0 else np.kron(
        np.eye(2**min_qubit_index), instruction.unitary)
    qubits_after = total_qubits - min_qubit_index - len(instruction.targets)
    return np.kron(before, np.eye(2**(qubits_after)))


def run_program(program: List[Instruction], n_qubits: int,
                initial_state: np.ndarray) -> np.ndarray:
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
        operator = operator @ get_operator(n_qubits, instruction)
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
    states = [
        _format_binary(idx, int(np.log2(len(state_vector))))
        for idx in range(len(state_vector))
    ]
    samples = choice(states, num_shots, p=probs)
    counts = defaultdict(int)
    for sample in samples:
        counts[sample] += 1
    return dict(counts)
