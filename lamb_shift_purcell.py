#############################################################################################
# Copyright (c) 2025 Christian Križan. All rights reserved.
# Owner retains all applicable rights wherever possible without prior infringement.
#
# Chalmers University of Technology granted usage rights
# for academic non-profit purposes as part of the
# default rules set by the university.
#
# This file is part of the "Josephson junction resistance manipulation"
# project, available at
# https://github.com/christiankrizan/Josephson-junction-resistance-manipulation/
# 
# Licensed under the terms described in the LICENSE file, at:
# https://github.com/christiankrizan/Josephson-junction-resistance-manipulation/blob/master/LICENSE.
#
# Provided strictly for academic, non-commercial use.
# Contact the author for negotiation of profit-generating use.
# 
# Provided "AS IS" without warranty.
# The author assumes no liability for damage, loss, or misuse
# resulting from this code.
#############################################################################################

def find_anharmonicity_from_lamb_shift(
    resonator_frequency_Hz,
    qubit_frequency_Hz,
    dispersive_shift_Hz,
    lamb_shift_Hz
    ):
    ''' Given the resonator-to-qubit detuning Δ, the dispersive shift 2·χ,
        and the Lamb shift L, find what the transmon anharmonicity is.
    '''
    
    # Helper variables.
    detuning = resonator_frequency_Hz - qubit_frequency_Hz
    single_chi = dispersive_shift_Hz/2
    
    # Expression.
    anharmonicity = -1 * (2*detuning * single_chi) / (lamb_shift_Hz + single_chi)
    
    # Done!
    return anharmonicity
    