"""
bindings
========
Foreign-function bindings to the native Monte Carlo engines. Both the pure-C
audit kernel and the OpenMP-parallel C++ engine expose an identical C ABI, so
either can stand in for the NumPy reference pricer:

>>> from bindings import price_contract_c, price_contract_cpp

If a native shared object has not been built yet, importing its wrapper still
succeeds; the :class:`FileNotFoundError` (with build instructions) is raised
only when you actually call the pricer.
"""
from .c_pricer_wrapper import price_contract_c          # noqa: F401
from .cpp_pricer_wrapper import price_contract_cpp, has_pybind_module  # noqa: F401

__all__ = ["price_contract_c", "price_contract_cpp", "has_pybind_module"]
