from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        # "primes.py",
        # "fib.pyx",
        "helloworld.pyx",
        "_cdnmf_fast.pyx",
        annotate=True
    ),
)
