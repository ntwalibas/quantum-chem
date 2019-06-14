#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of series of experiments trying to reproduce the results in quantum chemistry
using quantum computing. Contributions, corrections and clarifications welcome!

Author          : Ntwali Bashige
Copyright       : Copyright 2019 - Ntwali Bashige
License         : MIT
Version         : 0.0.1
Author          : Ntwali Bashige
Email           : ntwali.bashige@gmail.com
"""

# SciPy
import numpy as np
from matplotlib import pyplot as pp
from scipy.optimize import minimize

# PyQUIL
import pyquil.api as api
from pyquil.gates import *
from pyquil.quil import DefGate
from pyquil import Program, get_qc
from pyquil.paulis import PauliSum, PauliTerm
from pyquil.parameters import Parameter, quil_sin, quil_cos

# Grove
from grove.pyvqe.vqe import VQE


# We define a few gates that don't come natively with PyQuil but we need for the UCC ansatz
phi = Parameter("phi")
# We define the RX gate with a -1 phase
nrx = np.array([
        [- quil_cos(phi / 2), 1j * quil_sin(phi / 2)],
        [1j * quil_sin(phi / 2), -quil_cos(phi / 2)]
])
nrx_def = DefGate("NRX", nrx, [phi])
NRX = nrx_def.get_constructor()

# We define the RY gate with a -1 phase
nry = np.array([
        [-quil_cos(phi / 2), quil_sin(phi / 2)],
        [-quil_sin(phi / 2), -quil_cos(phi / 2)]
])
nry_def = DefGate("NRY", nry, [phi])
NRY = nry_def.get_constructor()


# Ansatz definition
def ansatz(params):
    prog = Program()
    prog += nrx_def
    prog += nry_def

    # Prepare the Hartree-Fock mean-field reference state
    prog += RX(np.pi, 1)

    # Implement UCC
    prog += RY(np.pi / 2, 0)
    prog += NRX(np.pi / 2)(1) #
    prog += CNOT(0, 1)
    prog += PHASE(params[0], 1)
    prog += CNOT(0, 1)
    prog += NRY(np.pi / 2)(0) #
    prog += RX(np.pi / 2, 1)

    # Apply necessary post-rotation before expectation estimation
    prog += RY(np.pi / 2, 0)
    prog += RY(np.pi / 2, 1)

    return prog


def main():
    # We keep track of how the expectation (energy) changes with the variational parameter and the number of steps
    params_changes = []
    steps_changes = []
    expectations_changes = []
    step = 0

    # Make the necessary initializations
    qvm = api.QVMConnection()
    molecule_coefficients = [-1.0524, 0.01128, 0.3979, 0.3979, 0.1809]

    def II(params):
        zi = PauliSum([PauliTerm.from_list([('Z', 0), ('I', 1)], coefficient = molecule_coefficients[0])])
        return VQE.expectation(ansatz(params), zi, None, qvm)

    def ZZ(params):
        zz = PauliSum([PauliTerm.from_list([('Z', 0), ('Z', 1)], coefficient = molecule_coefficients[1])])
        return VQE.expectation(ansatz(params), zz, None, qvm)

    def ZI(params):
        zi = PauliSum([PauliTerm.from_list([('Z', 0), ('I', 1)], coefficient = molecule_coefficients[2])])
        return VQE.expectation(ansatz(params), zi, None, qvm)

    def IZ(params):
        iz = PauliSum([PauliTerm.from_list([('I', 0), ('Z', 1)], coefficient = molecule_coefficients[3])])
        return VQE.expectation(ansatz(params), iz, None, qvm)

    def XX(params):
        xx = PauliSum([PauliTerm.from_list([('X', 0), ('X', 1)], coefficient = molecule_coefficients[4])])
        return VQE.expectation(ansatz(params), xx, None, qvm)

    def expectation(params):
        nonlocal step
        expec = II(params) + ZI(params) + IZ(params) + ZZ(params) + XX(params)

        # We keep track of the current minimization step count and new ansatz parameter with respect to the expectation value
        steps_changes.append(step)
        step = step + 1
        params_changes.append(params[0])
        expectations_changes.append(expec)

        # We return the newest expectation value
        return expec

    initial_params = [0.0]
    minimum = minimize(expectation, initial_params, method = "Nelder-Mead", options= {"initial_simplex": np.array([[0.0], [0.05]]), 'xatol': 1.0e-2})
    print(minimum.fun)

    # We make a plot for steps count vs expectation
    pp.xlabel("Minimization step")
    pp.ylabel("Expectation")
    pp.plot(steps_changes, expectations_changes)
    pp.show()

    # We make a plot for ansatz parameter vs expectation
    pp.xlabel("Ansatz parameter")
    pp.ylabel("Expectation")
    pp.plot(params_changes, expectations_changes)
    pp.show()

if __name__ == "__main__":
    main()
