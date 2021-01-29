import numpy as np
import pytest

from jqiskit.gates import Parametric, Instruction


def test_unitary() -> None:
    """Validate that non-unitary gates are caught."""
    with pytest.raises(ValueError):
        Instruction([0], np.array([[1., 2.], [3., 4.]]), False)


def test_parametric_gate_inputs() -> None:
    """Test all validation for parametric gates."""

    # Validate a non two-dimensional matrix
    with pytest.raises(ValueError):
        Parametric('[1]', [])

    # Validate that the smallest possible gate compiles.
    Parametric('[[1,0],[0,1]]', [0])

    # Make sure that the user is required to put in a single argument.
    with pytest.raises(ValueError):
        Parametric('[[1,0],[0,1]]', [])

    with pytest.raises(ValueError):
        Parametric('[[1,0],[0,1]]', [1, 2])

    # Validate that the smallest possible gate compiles.
    Parametric('[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]', [0, 1])

    with pytest.raises(ValueError):
        Parametric('[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]', [0])

    with pytest.raises(ValueError):
        Parametric('[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]', [0, 1, 2])
