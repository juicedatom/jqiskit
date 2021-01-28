import pytest
from jqiskit.gates import Parametric

def test_parametric_gate_inputs() -> None:
    """Test all validation for parametric gates."""

    # Validate a non two-dimensional matrix
    with pytest.raises(ValueError):
        Parametric('[1]', [])

    # Validate a non-square matrix.
    with pytest.raises(ValueError):
        Parametric('[[1,2],[3,4],[5,6]]', [])

    # Validate that the smallest possible gate compiles.
    Parametric('[[1,2],[3,4]]', [0])

    # Make sure that the user is required to put in a single argument.
    with pytest.raises(ValueError):
        Parametric('[[1,2],[3,4]]', [])

    with pytest.raises(ValueError):
        Parametric('[[1,2],[3,4]]', [1, 2])

    # Validate that the smallest possible gate compiles.
    Parametric('[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]', [0, 1])

    with pytest.raises(ValueError):
        Parametric('[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]', [0])

    with pytest.raises(ValueError):
        Parametric('[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]', [0, 1, 2])
