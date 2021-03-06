import numpy as np
import pytest

from jqiskit.backend import get_ground_state, preprocess_parametric, preprocess_parametric, preprocess_swaps, get_counts, get_operator, run_program
from jqiskit.gates import Parametric, Instruction, SWAP, Hadamard, CX


def test_ground_state() -> None:
    """Test generation of a ground state."""
    state = get_ground_state(1)
    np.testing.assert_allclose(state, [1., 0.])

    state = get_ground_state(2)
    np.testing.assert_allclose(state, [1., 0., 0., 0.])

    state = get_ground_state(3)
    np.testing.assert_allclose(state, [
        1.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
    ])


def test_get_operator() -> None:
    """Get operator from individual gates."""

    # Make sure that too many input are caught.
    with pytest.raises(ValueError):
        get_operator(10, SWAP(1, 5))

    with pytest.raises(IndexError):
        get_operator(3, SWAP(5, 6))

    # Make sure grabbing the unitary from a minimal circuit is a no-op.
    np.testing.assert_array_equal(get_operator(2, SWAP(0, 1)),
                                  SWAP(0, 1).unitary)


def test_preprocess_parametric() -> None:
    """Test the parametric evaluation of the Parametric gates."""

    # Validate the null circuit.
    assert preprocess_parametric([], {}) == []

    # Validate a very simple circuit.
    circuit_in = [Parametric('[[1.0, 0.0],[0.0, 1.0]]', [0])]
    circuit_out = preprocess_parametric(circuit_in, {})

    np.testing.assert_array_equal(circuit_out[0].unitary,
                                  [[1.0, 0.0], [0.0, 1.0]])
    assert circuit_in[0].targets == circuit_out[0].targets

    # Validate a simple circuit with replacement.
    circuit_in = [Parametric('[[1, 0],[0, exp(1.0j*theta)]]', [0])]
    theta = np.pi / 3
    circuit_out = preprocess_parametric(circuit_in, {'theta': theta})

    np.testing.assert_allclose(circuit_out[0].unitary,
                               [[1., 0.], [0.0, np.exp(1.0j * theta)]])
    assert circuit_in[0].targets == circuit_out[0].targets

    # For the same circuit as before, add a second non-parametric gates.
    circuit_in.append(Instruction([0], np.eye(2), True))
    circuit_out = preprocess_parametric(circuit_in, {'theta': theta})
    assert len(circuit_out) == 2
    np.testing.assert_allclose(circuit_out[0].unitary,
                               [[1., 0.], [0.0, np.exp(1.0j * theta)]])
    assert circuit_in[0].targets == circuit_out[0].targets
    assert circuit_in[1] == circuit_out[1]


def test_preprocess_swaps() -> None:
    """Test the swap-preprocessor."""

    # Preprocess an empty program.
    assert preprocess_swaps([]) == []

    # Preprocess a single instruction that's already valid (so nothing should happen).
    dummy_instruction = Instruction([0, 1], np.eye(4), False)
    processed = preprocess_swaps([dummy_instruction])
    assert len(processed) == 1

    # Preprocess a single instruction that needs its leads flipped.
    dummy_instruction = Instruction([1, 0], np.eye(4), False)
    processed = preprocess_swaps([dummy_instruction])
    assert len(processed) == 3
    assert isinstance(processed[0], SWAP)
    assert processed[0].targets == (0, 1)
    assert isinstance(processed[2], SWAP)
    assert processed[2].targets == (0, 1)

    # Preprocess a single instruction that needs to be pushed over.
    dummy_instruction = Instruction([3, 6], np.eye(4), False)
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
    dummy_instruction = Instruction([6, 3], np.eye(4), False)
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


def test_run_program() -> None:
    """Test running an actual program.

    Note that this will only be testing limited functionality given that run_program requires
    the code to be preprocessed. The rest of this will be tested on the api side.
    """
    # Test no-op program.
    new_state = run_program([], 1, np.array([1.0, 0.0]))
    np.testing.assert_almost_equal(new_state, [1.0, 0.0])

    # One h gate should bring us into super position.
    new_state = run_program([Hadamard(0)], 1, np.array([1.0, 0.0]))
    np.testing.assert_almost_equal(new_state, [1 / np.sqrt(2), 1 / np.sqrt(2)])

    # Two h gates should invert each other.
    new_state = run_program([Hadamard(0), Hadamard(0)], 1, np.array([1.0,
                                                                     0.0]))
    np.testing.assert_almost_equal(new_state, [1.0, 0.0])

    # CX Example from the task.
    new_state = run_program([Hadamard(0), CX(0, 1)], 2,
                            np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(new_state,
                                   [1 / np.sqrt(2), 0., 0., 1 / np.sqrt(2)])

    new_state = run_program([Hadamard(0), CX(0, 1)], 2,
                            np.array([0.0, 1.0, 0.0, 0.0]))
    np.testing.assert_almost_equal(new_state,
                                   [0.0, 1 / np.sqrt(2), 1 / np.sqrt(2), 0.0])


def test_get_counts() -> None:
    """Test realizing a state."""

    # Set the random seed for the monte-carlo.
    np.random.seed(0)

    # Test the ground state.
    state = get_ground_state(2)

    counts = get_counts(state, 100)
    assert len(counts) == 1
    assert counts['00'] == 100

    # Test a more complex state by back-calculating the probability.
    probs = np.array([0.25, 0.50, 0.125, 0.125])
    state = np.sqrt(probs)

    counts = get_counts(state, 10000)
    assert len(counts) == 4
    assert counts['00'] == 2546  # ~25.0%
    assert counts['01'] == 4961  # ~50.0%
    assert counts['10'] == 1230  # ~12.5%
    assert counts['11'] == 1263  # ~12.5%
