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

import numpy as np
import re
import math

def expo_calc(expr: str) -> str:
    ''' Expected input format: 5236.024512284495 * $-6.26(44)$e--8
	'''
    # Parse with regex
    # Example: 11779.126043616085 * $1.92(4)$e--3
    m = re.match(r"\s*([\d\.\-]+)\s*\*\s*\$\s*([\-]?\d+(?:\.\d+)?)(?:\((\d+)\))?\$e--(\d+)", expr)
    if not m:
        raise ValueError("Input string not in expected format")

    prefactor = float(m.group(1))
    central = float(m.group(2))
    error_digits = m.group(3)
    exp = -int(m.group(4))  # the --N means exponent is -N

    # Compute uncertainty magnitude
    if error_digits is not None:
        # Scale the error according to decimal places of central value
        decimals = len(m.group(2).split(".")[1]) if "." in m.group(2) else 0
        error = int(error_digits) * 10**(-decimals)
    else:
        error = 0.0

    # Value with error
    value = central * (10**exp)
    err_val = error * (10**exp)

    # Multiply prefactor
    result_val = prefactor * value
    result_err = abs(prefactor) * err_val

    # --- Formatting ---
    # round error to 1–2 significant digits
    err_exp = math.floor(math.log10(result_err)) if result_err > 0 else 0
    # Use 1 digit for error if first digit >= 3, otherwise 2
    err_digits = 1 if int(str(int(result_err / 10**err_exp))[0]) >= 3 else 2
    rounded_err = round(result_err, -err_exp + (err_digits - 1))
    rounded_val = round(result_val, -err_exp + (err_digits - 1))

    # Now express in the $(val)(err)$ format
    # Need to align digits: error refers to last digits of value
    fmt_str = f"{{:.{-err_exp + (err_digits - 1)}f}}"
    val_str = fmt_str.format(rounded_val)
    err_str = str(int(rounded_err * 10**(-err_exp + (err_digits - 1))))
    return f"${val_str}({err_str})$"

