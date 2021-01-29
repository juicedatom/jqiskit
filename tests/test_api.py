import numpy as np

from jqiskit.api import QuantumCircuit


def test_simple_program() -> None:
    """This tests a very simple 1 qbit circuit."""

    # Set a static seed.
    np.random.seed(0)

    # Test no-op program.
    qc = QuantumCircuit(1)
    counts, state = qc.measure(num_shots=100)

    assert len(counts) == 1
    assert counts['0'] == 100
    np.testing.assert_allclose(state, [1.0, 0.0])

    counts, state = qc.measure(num_shots=100,
                               feed_dict={},
                               initial_state=np.array([0.0, 1.0]))

    # Should still be a no-op on the '1' state.
    assert len(counts) == 1
    assert counts['1'] == 100
    np.testing.assert_allclose(state, [0.0, 1.0])

    # Adding an h gate should bring the circuit into superposition.
    qc.h(0)
    counts, state = qc.measure(num_shots=100)
    assert len(counts) == 2
    np.testing.assert_allclose(state, [1 / np.sqrt(2), 1 / np.sqrt(2)])

    # Adding another should bring us out of super position.
    qc.h(0)
    counts, state = qc.measure(num_shots=100)
    assert len(counts) == 1
    assert counts['0'] == 100
    np.testing.assert_allclose(state, [1.0, 0.0])

    # This is the same example from the example.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    _, state = qc.measure(num_shots=100)
    np.testing.assert_allclose(state, [1 / np.sqrt(2), 0., 0., 1 / np.sqrt(2)])
