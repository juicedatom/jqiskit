import numpy as np

from jqiskit.backend import get_ground_state, preprocess_parametric, preprocess_parametric
from jqiskit.gates import Parametric, Instruction

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
    pass
