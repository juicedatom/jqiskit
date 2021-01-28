import numpy as np

from jqiskit.backend import get_ground_state, preprocess_parametric, preprocess_parametric, preprocess_swaps
from jqiskit.gates import Parametric, Instruction, SWAP

def test_ground_state() -> None:
    """Test generation of a ground state."""
    state = get_ground_state(1)
    np.testing.assert_allclose(state, [1., 0.])

    state = get_ground_state(2)
    np.testing.assert_allclose(state, [1., 0., 0., 0.])

    state = get_ground_state(3)
    np.testing.assert_allclose(state, [1., 0., 0., 0., 0., 0., 0., 0.,])


def test_get_operator() -> None:
    pass


def test_preprocess_parametric() -> None:
    """Test the parametric evaluation of the Parametric gates."""

    # Validate the null circuit.
    assert preprocess_parametric([], {}) == []

    # Validate a very simple circuit.
    circuit_in = [Parametric('[[1,2],[1 + 3j,4]]', [0])]
    circuit_out = preprocess_parametric(circuit_in, {})

    np.testing.assert_array_equal(circuit_out[0].unitary, [[1., 2.],[1.0 + 3.0j, 4.]])
    assert circuit_in[0].targets == circuit_out[0].targets

    # Validate a simple circuit with replacement.
    circuit_in = [Parametric('[[1,2],[1 + 3j,4 * alpha]]', [0])]
    circuit_out = preprocess_parametric(circuit_in, {'alpha': 10.0})

    np.testing.assert_array_equal(circuit_out[0].unitary, [[1., 2.],[1.0 + 3.0j, 4. * 10.]])
    assert circuit_in[0].targets == circuit_out[0].targets

    # For the same circuit as before, add a second non-parametric gates.
    circuit_in.append(Instruction([0], [[0., 0.],[0., 0.,]], True))
    circuit_out = preprocess_parametric(circuit_in, {'alpha': 10.0})
    assert len(circuit_out) == 2
    np.testing.assert_array_equal(circuit_out[0].unitary, [[1., 2.],[1.0 + 3.0j, 4. * 10.]])
    assert circuit_in[0].targets == circuit_out[0].targets
    assert circuit_in[1] == circuit_out[1]


def test_preprocess_swaps() -> None:
    """Test the swap-preprocessor."""

    # Preprocess an empty program.
    assert preprocess_swaps([]) == []

    # Preprocess a single instruction that's already valid (so nothing should happen).
    dummy_instruction = Instruction([0, 1], [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], False)
    processed = preprocess_swaps([dummy_instruction])
    assert len(processed) == 1

    # Preprocess a single instruction that needs its leads flipped.
    dummy_instruction = Instruction([1, 0], [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], False)
    processed = preprocess_swaps([dummy_instruction])
    assert len(processed) == 3
    assert isinstance(processed[0], SWAP)
    assert processed[0].targets == (0, 1)
    assert isinstance(processed[2], SWAP)
    assert processed[2].targets == (0, 1)

    # Preprocess a single instruction that needs to be pushed over.
    dummy_instruction = Instruction([3, 6], [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], False)
    processed = preprocess_swaps([dummy_instruction])
    assert len(processed) == 5
    assert isinstance(processed[0], SWAP)
    assert processed[0].targets == (5, 6)

    assert isinstance(processed[1], SWAP)
    assert processed[1].targets == (4, 5)

    assert isinstance(processed[3], SWAP)
    assert processed[3].targets == (4, 5)

    assert isinstance(processed[4], SWAP)
    assert processed[4].targets == (5, 6)

    # Preprocess a single instruction that needs to be pushed over and flipped.
    dummy_instruction = Instruction([6, 3], [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]], False)
    processed = preprocess_swaps([dummy_instruction])
    assert len(processed) == 7
    assert isinstance(processed[0], SWAP)
    assert processed[0].targets == (5, 6)

    assert isinstance(processed[1], SWAP)
    assert processed[1].targets == (4, 5)

    assert isinstance(processed[2], SWAP)
    assert processed[3].targets == (3, 4)

    assert isinstance(processed[4], SWAP)
    assert processed[4].targets == (3, 4)

    assert isinstance(processed[1], SWAP)
    assert processed[5].targets == (4, 5)

    assert isinstance(processed[1], SWAP)
    assert processed[6].targets == (5, 6)

