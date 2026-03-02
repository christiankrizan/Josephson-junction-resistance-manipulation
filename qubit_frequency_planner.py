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

import itertools
import numpy as np
import matplotlib.pyplot as plt
from superconductivity_tools import calculate_f01_from_RT_resistance_and_anharmonicity

def get_lowest_and_highest_frequencies_for_qubits(
    list_of_room_temperature_resistance,
    list_of_anharmonicity_Hz,
    list_of_Delta_cold_eV,
    list_fabrication_error_R_percent,
    difference_between_RT_and_cold_resistance = 1.1375,
    T = 0.010,
    verbose = False
    ):
    ''' Based on the known room-temperature resistances of two coupled
        qubits, and known transmon anharmonicities for the two qubits,
        and a known error in percent on the resistance, plot the
        spectrum where the CZ₀₂, CZ₂₀, and iSWAP gates are expected.
        Parametric gates, mind you.
        
        For list_fabrication_error_R_percent:
            "2.0" means you have a ±2.0% error on your R, from fabrication.
            Meaning that R_true ∈ [R_nominal*0.980, R_nominal*1.020]
    '''
    
    # Safety check
    n_qubits = len(list_of_room_temperature_resistance)
    if not (
        len(list_of_anharmonicity_Hz) == 
        len(list_of_Delta_cold_eV) == 
        len(list_fabrication_error_R_percent) == 
        n_qubits
    ):
        raise ValueError("Error! All input lists must have the same length.")
    
    lowest_frequencies_Hz  = [0.0] * n_qubits
    highest_frequencies_Hz = [0.0] * n_qubits
    
    # For all qubits,
    for qubit_ii in range(n_qubits):
        
        # Resistance values.
        R_nominal = list_of_room_temperature_resistance[qubit_ii]
        fabrication_error = list_fabrication_error_R_percent[qubit_ii]
        
        # Compute resistance values expected for the
        # low and high frequency outcomes.
        R_low  = R_nominal * (1 - 0.01 * fabrication_error)
        R_high = R_nominal * (1 + 0.01 * fabrication_error)
        
        # Low frequency (from low R)
        lowest_frequencies_Hz[qubit_ii] = calculate_f01_from_RT_resistance_and_anharmonicity(
            room_temperature_resistance = R_high,
            anharmonicity_Hz = list_of_anharmonicity_Hz[qubit_ii],
            Delta_cold_eV = list_of_Delta_cold_eV[qubit_ii],
            difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
            T = T,
            verbose = verbose
        )
        
        # High frequency (from high R)
        highest_frequencies_Hz[qubit_ii] = calculate_f01_from_RT_resistance_and_anharmonicity(
            room_temperature_resistance = R_low,
            anharmonicity_Hz = list_of_anharmonicity_Hz[qubit_ii],
            Delta_cold_eV = list_of_Delta_cold_eV[qubit_ii],
            difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
            T = T,
            verbose = verbose
        )
    
    # Report!
    return lowest_frequencies_Hz, highest_frequencies_Hz

def calculate_cz20_cz02_iswap(
    f_q1,
    f_q2,
    anharmonicity_q1,
    anharmonicity_q2
    ):
    ''' Given two qubit frequencies, and two qubit anharmonicities,
        calculate where the CZ₂₀, CZ₀₂, and iSWAP gates are expected to be.
        Parametric, mind you.
        
        All arguments are in units of Hz.
    '''
    
    # Calculate gates!
    f_cz20  = np.abs((2*f_q1 + anharmonicity_q1) - (f_q1 + f_q2))
    f_cz02  = np.abs((2*f_q2 + anharmonicity_q2) - (f_q1 + f_q2))
    f_iswap = np.abs(f_q1 - f_q2)
    
    # Return!
    return f_cz20, f_cz02, f_iswap

def compute_min_spacing(interval_objects):
    ''' Figures out the minimum spacing between 2 two-qubit-gate
        bands considered.
    '''
    
    # Catch bad input.
    if len(interval_objects) < 2:
        return None, None, []

    # Sort by lower bound.
    sorted_objs = sorted(
        interval_objects,
        key=lambda obj: obj["interval"][0]
    )
    
    min_spacing = float("inf")
    min_pair = None
    overlapping_pairs = []

    for i in range(len(sorted_objs) - 1):
        obj1 = sorted_objs[i]
        obj2 = sorted_objs[i + 1]

        l1, h1 = obj1["interval"]
        l2, h2 = obj2["interval"]

        spacing = l2 - h1

        # Track overlaps.
        if spacing < 0:
            overlapping_pairs.append((obj1, obj2))

        # Track smallest spacing.
        if spacing < min_spacing:
            min_spacing = spacing
            min_pair = (obj1, obj2)

    return min_spacing, min_pair, overlapping_pairs


def draw_frequency_crowding_7q(
    savepath = ''
    ):
    ''' For a set of seven qubit resistances and their corresponding transmon
        anharmonicities, draw the resulting frequency crowding that occurs
        in a three-qubit coupler design with 7 qubits.
        
        Parametric gates, mind you.
    '''
    
    # Set parameter data.
    R_list = [5275, 6000, 6475, 7200, 7800, 8375, 9000]
    anharmonicity_list = [-280e6, -280e6, -280e6, -280e6, -280e6, -280e6, -280e6]
    Delta_list = [174.3e-6, 174.3e-6, 174.3e-6, 174.3e-6, 174.3e-6, 174.3e-6, 174.3e-6] # With a superconducting energy gap as found in Križan2026.
    fab_error_list = [2, 2, 2, 2, 2, 2, 2]  # In percent.
    
    # Figure out what are the highest and lowest qubit frequency values
    # for the resistances and harmonicites as given above.
    f_low, f_high = get_lowest_and_highest_frequencies_for_qubits(
        R_list, anharmonicity_list, Delta_list, fab_error_list
    )

    # Define legal qubit pairs and their couplers.
    ## Assume a design that looks like this:
    ##
    ##     q1  q2
    ##      |  |
    ##     c1  c2
    ##     /\  /\
    ##   q3  q4  q5
    ##       |
    ##       c3
    ##       /\
    ##     q6  q7
    ##                      // Christian Križan
    
    coupler_map = {
        "Coupler1": [(0, 2), (0, 3), (2, 3)],  # q1q3, q1q4, q3q4
        "Coupler2": [(1, 3), (1, 4), (3, 4)],  # q2q4, q2q5, q4q5
        "Coupler3": [(3, 5), (3, 6), (5, 6)],  # q4q6, q4q7, q6q7
    }

    # Storage dictionaries and lists.
    gate_ranges = {}
    all_intervals = []
    
    # Go through all combinations.
    for coupler_name, pairs in coupler_map.items():
        gate_ranges[coupler_name] = {}

        for (i, j) in pairs:

            # Low frequencies.
            cz20_low, cz02_low, iswap_low = calculate_cz20_cz02_iswap(
                f_low[i], f_low[j],
                anharmonicity_list[i], anharmonicity_list[j]
            )

            # High frequencies.
            cz20_high, cz02_high, iswap_high = calculate_cz20_cz02_iswap(
                f_high[i], f_high[j],
                anharmonicity_list[i], anharmonicity_list[j]
            )
            
            # Set pair name.
            pair_name = f"q{i+1}q{j+1}"

            # Store intervals for analysis later. Include metadata, that is,
            # which is the actual gate + between which qubits,
            # that has this band?
            all_intervals.append({
                "interval": (cz20_low, cz20_high),
                "gate": "CZ₂₀",
                "pair": pair_name,
                "coupler": coupler_name
            })
            all_intervals.append({
                "interval": (cz02_low, cz02_high),
                "gate": "CZ₀₂",
                "pair": pair_name,
                "coupler": coupler_name
            })
            all_intervals.append({
                "interval": (iswap_low, iswap_high),
                "gate": "iSWAP",
                "pair": pair_name,
                "coupler": coupler_name
            })

            gate_ranges[coupler_name][pair_name] = {
                "CZ₂₀": (cz20_low, cz20_high),
                "CZ₀₂": (cz02_low, cz02_high),
                "iSWAP": (iswap_low, iswap_high),
            }
    
    # Is there spectral overlap? Then print this to the user.
    min_spacing, min_pair, overlaps = compute_min_spacing(all_intervals)
    if overlaps:
        print("Overlaps detected!\n")
        for obj1, obj2 in overlaps:
            print(
                f"{obj1['gate']} {obj1['pair']} ({obj1['coupler']}) "
                f"overlaps with "
                f"{obj2['gate']} {obj2['pair']} ({obj2['coupler']})"
            )
    else:
        print("No spectral overlap detected.")
    
    # Check band spacing.
    if min_pair is not None:
        obj1, obj2 = min_pair
        print("\nTightest spectral spacing occurs between:")
        print(
            f"{obj1['gate']} {obj1['pair']} on {obj1['coupler'].replace('Coupler', 'Coupler ')}"
            "  and  "
            f"{obj2['gate']} {obj2['pair']} on {obj2['coupler'].replace('Coupler', 'Coupler ')}."
        )
        print(f"Minimum spacing: {min_spacing/1e6:.3f} MHz")
    
    # Plotting time!
    plt.figure(figsize=(20.31, 8))
    plt.title("Expected CZ₂₀, CZ₀₂, and iSWAP frequency ranges", fontsize=33)
    plt.xlabel("Coupler frequency [MHz]", fontsize=33)
    plt.ylabel("Coupler", fontsize=33)
    plt.xlim(-10, 1410)

    # Assign vertical positions automatically.
    coupler_positions = {
        "Coupler1": 0.1,
        "Coupler2": 0.3,
        "Coupler3": 0.5,
    }
    
    # Define colours.
    colors = {
        "CZ₂₀": "#1C70EE",
        "CZ₀₂": "#1CEE70",
        "iSWAP": "#EE1C1C",
    }
    
    # Define heigh of the bars.
    bar_height = 0.04

    for coupler_name, pairs in gate_ranges.items():
        base_y = coupler_positions[coupler_name]

        for idx, (pair_name, gates) in enumerate(pairs.items()):
            y_offset = base_y + idx * bar_height
            
            for gate_type, (low, high) in gates.items():
                
                low_mhz  = low  / 1e6
                high_mhz = high / 1e6
                width    = high_mhz - low_mhz

                plt.barh(
                    y=y_offset,
                    width=width,
                    left=low_mhz,
                    height=bar_height,
                    color=colors[gate_type],
                    alpha=0.9,
                    label=gate_type if base_y == 0.1 and idx == 0 else ""
                )

                # --- Add text label ---
                label_text = f"{gate_type} {pair_name}"
                
                # Is the high_mhz value actually higher than the low_mhz value?
                if high_mhz > low_mhz:
                    place_here = high_mhz + 5
                else:
                    place_here = low_mhz + 5
                    
                plt.text(
                    x=place_here,
                    y=y_offset,
                    s=label_text,
                    va="center",
                    ha="left",
                    fontsize=16,
                    color="black"
                )

    ## Massage the plot formatting:
    plt.xticks(fontsize=30) # Solve x-ticks here, already.
    
    # Define y-tick positions (center of each coupler block)
    ytick_positions = []
    ytick_labels = []
    for coupler_name, base_y in coupler_positions.items():
        center_position = base_y + bar_height
        ytick_positions.append(center_position)
        ytick_labels.append(coupler_name.replace("Coupler", "C"))

    plt.yticks(ytick_positions, ytick_labels, fontsize=30)
    plt.legend(fontsize=26)
    plt.tight_layout()
    
    # Save plot?
    if savepath != '':
        plt.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show stuff.
    plt.show()


def intervals_overlap(interval1, interval2):
    """
    Returns True if two intervals (low, high) overlap.
    """
    (l1, h1) = interval1
    (l2, h2) = interval2
    return not (h1 <= l2 or h2 <= l1)

def any_overlap(interval_list, min_bandwidth=0.01e6, min_spacing=0.01e6):

    # Reject tiny intervals
    for (low, high) in interval_list:
        if (high - low) < min_bandwidth:
            return True

    # Sort
    sorted_intervals = sorted(interval_list, key=lambda x: x[0])

    # Check spacing
    for i in range(len(sorted_intervals) - 1):
        l1, h1 = sorted_intervals[i]
        l2, h2 = sorted_intervals[i+1]

        if (l2 - h1) < min_spacing:
            return True

    return False

def brute_force_search():
    ''' Throw hands in the air and brute-force the resistances and
        anharmonicities that allow for a collision-free spectrum.
    '''
    
    # Search ranges.
    R_search = np.linspace(5000, 17000, 26)  # Initiate a coarse sweep first.
    anharmonicity_search = np.linspace(-300e6, -150e6, 15)

    Delta_list = [174.3e-6] * 7
    fab_error_list = [2] * 7

    coupler_map = {
        "Coupler1": [(0, 2), (0, 3)],
        "Coupler2": [(1, 3), (1, 4)],
        "Coupler3": [(3, 5), (3, 6)],
    }

    # Sweep q4 first, that is, index 3.
    for R4 in R_search:
        for anharmonicity4 in anharmonicity_search:
            
            # Update the user.
            print(f"Trying q4: R={R4:.1f}, anharmonicity={anharmonicity4/1e6:.1f} MHz")

            # Sweep remaining qubits.
            for candidate in itertools.product(R_search, anharmonicity_search, repeat=6):

                R_list = [None]*7
                anharmonicity_list = [None]*7

                # Insert q4 fixed
                R_list[3] = R4
                anharmonicity_list[3] = anharmonicity4

                idx = 0
                for q in range(7):
                    if q == 3:
                        continue
                    R_list[q] = candidate[idx]
                    anharmonicity_list[q] = candidate[idx+1]
                    idx += 2

                # --- Compute frequency bounds ---
                f_low, f_high = get_lowest_and_highest_frequencies_for_qubits(
                    R_list, anharmonicity_list, Delta_list, fab_error_list
                )

                # --- Collect ALL intervals ---
                all_intervals = []

                for pairs in coupler_map.values():
                    for (i, j) in pairs:

                        cz20_low, cz02_low, iswap_low = calculate_cz20_cz02_iswap(
                            f_low[i], f_low[j],
                            anharmonicity_list[i], anharmonicity_list[j]
                        )

                        cz20_high, cz02_high, iswap_high = calculate_cz20_cz02_iswap(
                            f_high[i], f_high[j],
                            anharmonicity_list[i], anharmonicity_list[j]
                        )

                        all_intervals.append((cz20_low, cz20_high))
                        all_intervals.append((cz02_low, cz02_high))
                        all_intervals.append((iswap_low, iswap_high))

                # --- Check legality ---
                if not any_overlap(all_intervals):
                    print("\nLegal solution found!\n")
                    return R_list, anharmonicity_list

    print("No legal solution found.")
    return None
    
def draw_frequency_crowding_3qb():
    
    # --- Step 1: Get qubit frequency ranges ---
    R_list = [12571.16, 13418.01, 12997.81]
    anharmonicity_list = [-256.07e6, -207.49e6, -210e6]
    Delta_list = [174.3e-6, 174.3e-6, 174.3e-6]
    fab_error_list = [2, 2, 2]

    f_low, f_high = get_lowest_and_highest_frequencies_for_qubits(
        R_list, anharmonicity_list, Delta_list, fab_error_list
    )

    # --- Step 2: Calculate gate ranges ---
    cz20_low_q1q2,  cz02_low_q1q2,  iswap_low_q1q2  = calculate_cz20_cz02_iswap(f_low[0],  f_low[1],  anharmonicity_list[0], anharmonicity_list[1])
    cz20_high_q1q2, cz02_high_q1q2, iswap_high_q1q2 = calculate_cz20_cz02_iswap(f_high[0], f_high[1], anharmonicity_list[0], anharmonicity_list[1])
    cz20_low_q2q3,  cz02_low_q2q3,  iswap_low_q2q3  = calculate_cz20_cz02_iswap(f_low[1],  f_low[2],  anharmonicity_list[1], anharmonicity_list[2])
    cz20_high_q2q3, cz02_high_q2q3, iswap_high_q2q3 = calculate_cz20_cz02_iswap(f_high[1], f_high[2], anharmonicity_list[1], anharmonicity_list[2])
    
    # --- Step 3: Plot intervals horizontally ---
    plt.figure(figsize=(10, 2))
    plt.title("Expected CZ₂₀, CZ₀₂, and iSWAP frequency ranges")
    plt.xlabel("Frequency [MHz]")
    plt.xlim(-10, 1210)

    # Y positions for bars
    y_positions = {
        "CZ02_q1q2":    0.05,
        "CZ20_q1q2":    0.05,
        "iSWAP_q1q2":   0.05,
        "CZ02_q2q3":    0.1,
        "CZ20_q2q3":    0.1,
        "iSWAP_q2q3":   0.1
    }

    # Plot horizontal bars
    plt.barh(y=y_positions["CZ02_q1q2"],
             width=(cz02_high_q1q2 - cz02_low_q1q2)/1e6,
             left=cz02_low_q1q2/1e6,
             height=0.05,
             color="#1CEE70",
             alpha=0.9,
             label="CZ₀₂_q1q2")

    plt.barh(y=y_positions["CZ20_q1q2"],
             width=(cz20_high_q1q2 - cz20_low_q1q2)/1e6,
             left=cz20_low_q1q2/1e6,
             height=0.05,
             color="#1C70EE",
             alpha=0.9,
             label="CZ₂₀_q1q2")

    plt.barh(y=y_positions["iSWAP_q1q2"],
             width=(iswap_high_q1q2 - iswap_low_q1q2)/1e6,
             left=iswap_low_q1q2/1e6,
             height=0.05,
             color="#EE1C1C",
             alpha=0.9,
             label="iSWAP_q1q2")
    
    plt.barh(y=y_positions["CZ02_q2q3"],
             width=(cz02_high_q2q3 - cz02_low_q2q3)/1e6,
             left=cz02_low_q2q3/1e6,
             height=0.05,
             color="#1CEE70",
             label="CZ₀₂_q2q3")
    
    plt.barh(y=y_positions["CZ20_q2q3"],
             width=(cz20_high_q2q3 - cz20_low_q2q3)/1e6,
             left=cz20_low_q2q3/1e6,
             height=0.05,
             color="#1C70EE",
             label="CZ₂₀_q2q3")

    plt.barh(y=y_positions["iSWAP_q2q3"],
             width=(iswap_high_q2q3 - iswap_low_q2q3)/1e6,
             left=iswap_low_q2q3/1e6,
             height=0.05,
             color="#EE1C1C",
             label="iSWAP_q2q3")

    plt.yticks([])  # hide y-axis
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()