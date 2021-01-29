# JQiskit (**J**osh-**Qiskit**)
This module was written as the respones to Task 3 of the QOSF mentorship program. It is written purely as an exersize to learn about quantum computing. Ultimately, it implements a very primitive version of `qiskit.QuantumCircuit` with limited functionality. This was really my first time learning how all the gates work under the hood. After doing this the first time, I think that there's a lot of I would've changed about the structure of the code. For example, I probably would have made controlled gates a feature of an individual gate, rather than coding each gate by hand.

If I were to continue this project, I would plumb the sympy matrices all the way through the simulation s.t. the matrix can actually be differentialble. I think it's set up s.t. it is very possible. That way you could differentiate your circuit and do cooler optimization type things.

## Installation
Simply install the package in the top level directory by running,

```
pip install -e .
```

That should install the jqiskit library and allow for it's api usage.


## Examples

You will find several examples of usage in the `notebooks/` directory. Note that in order to run the `QiskitComparison` notebook you'll need to `pip install qiskit`. I didn't include it in the `setup.py` because it's not needed in the actual `jqiskit` library.

## Testing

In order to run the unit tests, simply run `pytest` in the `tests/` module. If you want to install and run the tests at the same time, then you could try

```
pip install -e . && pytest
```

in this directory. `pytest` should find all the `test_*.py` files and automatically test them.
