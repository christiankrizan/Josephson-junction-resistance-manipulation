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

from random import randint
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import sys
import os
import re
import colorsys # Used for generating curve colours.
import time as time_module
from scipy.optimize import curve_fit
from scipy.stats import ttest_rel
from scipy.stats import moment
from scipy.stats import linregress
from scipy.interpolate import interp1d
from datetime import datetime
from collections import deque

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16)/255.0 for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(round(c * 255)) for c in rgb))

def interpolate_hsv_colours(start_hex, end_hex, n):
    start_rgb = hex_to_rgb(start_hex)
    end_rgb = hex_to_rgb(end_hex)

    start_hsv = colorsys.rgb_to_hsv(*start_rgb)
    end_hsv = colorsys.rgb_to_hsv(*end_rgb)

    # Handle hue circular interpolation
    h1, h2 = start_hsv[0], end_hsv[0]
    if abs(h1 - h2) > 0.5:
        if h1 > h2:
            h2 += 1
        else:
            h1 += 1

    colours = []
    for i in range(n):
        t = i / (n - 1) if n > 1 else 0
        h = (h1 + t * (h2 - h1)) % 1.0
        s = start_hsv[1] + t * (end_hsv[1] - start_hsv[1])
        v = start_hsv[2] + t * (end_hsv[2] - start_hsv[2])
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colours.append(rgb_to_hex(rgb))

    return colours

def get_colourise(
    colourised_counter
    ):
    ''' If using patterned colours, return an appropriate colour.
       -2.x:  White
       -1.x:  Regular Black
        0.x:  Kite Blue
        1.x:  Sunny Yellow
        2.x:  Raspberry Red
        3.x:  Frog Green
    '''
    
    # WHITE TONES
    if ((colourised_counter <= -2) and (colourised_counter > -3)):
        if   colourised_counter == -2.1:
            return "#F2F1ED"
        elif colourised_counter == -2.2:
            return "#F2F1ED"
        elif colourised_counter == -2.3:
            return "#F2F1ED"
        elif colourised_counter == -2.4:
            return "#F2F1ED"
        else:
            # Covers the .0 case too.
            return "#F2F1ED"
    
    # REGULAR BLACK TONES
    if ((colourised_counter <= -1) and (colourised_counter > -2)):
        if   colourised_counter == -1.1:
            return "#4D4A49"
        elif colourised_counter == -1.2:
            return "#777373"
        elif colourised_counter == -1.3:
            return "#9F9E9B"
        elif colourised_counter == -1.4:
            return "#C9C7C5"
        else:
            # Covers the .0 case too.
            return "#242021"
    
    # KITE BLUE TONES
    elif ((colourised_counter >= 0) and (colourised_counter < 1)):
        if   colourised_counter == 0.1:
            return "#6EBAE0"
        elif colourised_counter == 0.2:
            return "#90C8E4"
        elif colourised_counter == 0.3:
            return "#B0D6E6"
        elif colourised_counter == 0.4:
            return "#D2E4EA"
        else:
            # Covers the .0 case too.
            return "#4EADDD"
    
    # SUNNY YELLOW TONES
    elif ((colourised_counter >= 1) and (colourised_counter < 2)):
        if   colourised_counter == 1.1:
            return "#F6D330"
        elif colourised_counter == 1.2:
            return "#F5DA60"
        elif colourised_counter == 1.3:
            return "#F4E38E"
        elif colourised_counter == 1.4:
            return "#F3EABE"
        else:
            # Covers the .0 case too.
            return "#F7CC01"
    
    # RASPBERRY RED TONES
    elif ((colourised_counter >= 2) and (colourised_counter < 3)):
        if   colourised_counter == 2.1:
            return "#EE635B"
        elif colourised_counter == 2.2:
            return "#F08680"
        elif colourised_counter == 2.3:
            return "#F0ABA4"
        elif colourised_counter == 2.4:
            return "#F2CEC9"
        else:
            # Covers the .0 case too.
            return "#EE4037"
    
    # FROG GREEN TONES
    elif ((colourised_counter >= 3) and (colourised_counter < 4)):
        if   colourised_counter == 3.1:
            return "#6EBA30"
        elif colourised_counter == 3.2:
            return "#90C860"
        elif colourised_counter == 3.3:
            return "#B0D68E"
        elif colourised_counter == 3.4:
            return "#D2E4BE"
        else:
            # Covers the .0 case too.
            return "#4EAD01"
    
    # ERROR
    else:
        # Default!
        return '#000000'

def calculate_f01_from_RT_resistance(
    room_temperature_resistance,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    verbose = True
    ):
    ''' Given a room temperature resistance, calculate the resulting f_01.
        For difference_between_RT_and_cold_resistance, a value of 1.1385
        means that a cold junction is 13.85 % more resistive than a room
        temperature one. This number is the average of the two junctions
        that were measured in Fig. 2.12 by A. Osman's thesis.
        
        The thesis is OF COURSE not uploaded to Chalmers ODR archive as of
        2025-01-31, that would require somebody to actually know anything
        about archiving practices and university rules at that department's
        division. Lol, ngh. Find the thesis here:
        https://research.chalmers.se/en/publication/543784
        
        E_C is the transmon's charging energy.
        Delta_cold is the superconducting gap at millikelvin temperatures.
        T is the temperature of operation, typically dilution fridge
          temperatures. For instance, 10 mK.
    '''
    
    # Physical constants
    h = 6.62607015e-34       # Planck's constant [J/Hz]
    h_bar = h / (2 * np.pi)  # Reduced Planck's constant [J/Hz]
    e = 1.602176634e-19      # Elementary charge [C]
    k_B = 1.380649e-23       # Boltzmann's constant [J/K]
    
    # User-set values
    ## Calculate the normal state resistance of the S-I-S junction.
    R_N = room_temperature_resistance * difference_between_RT_and_cold_resistance
    Delta_cold = Delta_cold_eV * e # Superconducting gap at mK temperature [J]
    E_C = E_C_in_Hz * h # Charging energy [J]
    
    # Calculate I_c using the Ambegaokar-Baratoff relation
    I_c = (np.pi * Delta_cold)/(2*e*R_N) * np.tanh(Delta_cold / (2 * k_B * T))
    
    # Calculate E_J
    E_J = (h_bar / (2*e)) * I_c
    
    # Print E_C and E_J?
    if verbose:
    
        ## Print E_C
        if (E_C/h) > 1e9:
            print("E_C is [GHz]: "+str((E_C/h)/1e9))
        elif (E_C/h) > 1e6:
            print("E_C is [MHz]: "+str((E_C/h)/1e6))
        elif (E_C/h) > 1e3:
            print("E_C is [kHz]: "+str((E_C/h)/1e3))
        else:
            print("E_C is [Hz]: "+str(E_C/h))
        
        ## Print E_J
        if (E_J/h) > 1e9:
            print("E_J is [GHz]: "+str((E_J/h)/1e9))
        elif (E_J/h) > 1e6:
            print("E_J is [MHz]: "+str((E_J/h)/1e6))
        elif (E_J/h) > 1e3:
            print("E_J is [kHz]: "+str((E_J/h)/1e3))
        else:
            print("E_J is [Hz]: "+str(E_J/h))
        
        ## Print E_J / E_C
        print("E_J / E_C is: "+str(E_J/E_C))
    
    # Calculate f_01
    '''## Koch 2007 equation regarding transmon f_01, precision to second order.
    ## https://doi.org/10.1103/PhysRevB.77.180502
    second_order_correction_factor = -(E_C / 2) * (E_C / (8*E_J))'''
    # Updated calculation from
    xi = np.sqrt(2 * E_C / E_J)
    second_order_correction_factor = (1 + (1/4)*xi + (21/128)*(xi**2))
    f_01 = (np.sqrt(8 * E_J * E_C) - E_C * second_order_correction_factor)/h
    
    # Return value!
    return f_01

def calculate_RT_resistance_from_target_f01(
    target_f_01,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    R_N_initial_guess = 15000,
    acceptable_frequency_offset = 100,
    verbose = True
    ):
    ''' Given a target |0⟩ → |1⟩ transition of a transmon,
        calculate what room-temperature resistance the Josephson junction
        should have.
        
        First, read the description for the function
        "calculate_f01_from_RT_resistance" above.
        
        Now, instead of fighting quartic functions in finding out the
        inverse of the f_01 equation, this function uses a different
        approach. As in, guessing different resistances until a
        match is found.
        
        acceptable_frequency_offset [Hz] is the difference after which
        the function will stop.
        
        R_N_initial_guess [Ω] is the initial resistance guess where we'll
        begin.
    '''
    
    done = False
    r_rt = R_N_initial_guess
    while(not done):
        
        # Try!
        result_freq = calculate_f01_from_RT_resistance(
            room_temperature_resistance = r_rt,
            E_C_in_Hz = E_C_in_Hz,
            Delta_cold_eV = Delta_cold_eV,
            difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
            T = T,
            verbose = False
        )
        
        # Find difference.
        ## Positive difference: the target freq is above your calculated value.
        ## Negative difference: the target freq is below your calculated value.
        difference = target_f_01 - result_freq
        
        # Finished?
        if (np.abs(difference) < np.abs(acceptable_frequency_offset)):
            done = True
        else:
            if difference > 0:
                r_rt *= 0.99
            elif difference < 0:
                r_rt *= 1.01
    
    # Print some values.
    if verbose:
        calculate_f01_from_RT_resistance(
            room_temperature_resistance = r_rt,
            E_C_in_Hz = E_C_in_Hz,
            Delta_cold_eV = Delta_cold_eV,
            difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
            T = T,
            verbose = True
        )
    
    # Return value!
    return r_rt

def calculate_resistance_to_manipulate_to(
    target_f_01,
    E_C_in_Hz,
    Delta_cold_eV,
    original_resistance_of_junction = 0,
    expected_aging = 0,
    expected_resistance_relaxation = 0,
    verbose = True
    ):
    ''' Given a target frequency, calculate what resistance should be
        manipulated to, including resistance relaxation effects.
        
        The units of expected_aging and expected_resistance_relaxation are linear.
        I.e., 1.05 means that you expect an additional 5 % extra resistance.
    '''
    
    # Get the room temperature resistance that
    # this target frequency corresponds to.
    room_temperature_resistance_to_hit = calculate_RT_resistance_from_target_f01(
        target_f_01 = target_f_01,
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        verbose = verbose
    )
    
    # Print to user?
    if verbose:
        # Original junction resistance known?
        if original_resistance_of_junction > 0:
            increase = room_temperature_resistance_to_hit / original_resistance_of_junction
            assert increase >= 1.0, "Error! The resistance can only go up (for now). Unable to tune junction to the target frequency. Needed tuning: "+str((increase-1)*100)+" %"
            print(f"The target room-temperature resistance is: {(room_temperature_resistance_to_hit):.3f} [Ω], which corresponds to {((increase-1)*100):.3f} %")
        else:
            print(f"The target room-temperature resistance is: {(room_temperature_resistance_to_hit):.3f} [Ω]")
    
    # Does the sample have some aging left to do?
    ## TODO
    if expected_aging > 0:
        raise NotImplementedError("Not finished.")
    
    # TODO current, our only knowledge of the relaxation is that it is a fixed offset compared to the initial resistance.
    ## Well, to be picky, we know that the end number is pretty much
    ## a fixed resistance offset.
    if expected_resistance_relaxation >= 0:
        if original_resistance_of_junction == 0:
            raise ValueError("Halted! If assuming relaxation, then the original resistance of the junction must be known. Check your arguments.")
        
        print(f"Expecting {((expected_resistance_relaxation-1)*100):.3f} % worth of resistance relaxation.")
        
        # Figure out the relaxation.
        relaxation_in_ohms = (original_resistance_of_junction * expected_resistance_relaxation - original_resistance_of_junction)
        
        # Subtract!
        room_temperature_resistance_to_hit -= relaxation_in_ohms
        increase = room_temperature_resistance_to_hit / original_resistance_of_junction
        print(f"Excluding resistance relaxation, expect to hit: {(room_temperature_resistance_to_hit):.3f} [Ω], which is {((increase-1)*100):.3f} %")
        
    elif expected_resistance_relaxation < 0:
        raise ValueError("Error! The resistance relaxation is expected to be a positive number; the resistance is expected to increase post-manipulation.")
    
    return room_temperature_resistance_to_hit

def fit_ambegaokar_baratoff_josephson_koch_to_resistance(
    measured_junction_resistances,
    measured_qubit_frequencies,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    colourise = False,
    verbose = True
    ):
    ''' Given measured frequency and resistance values,
        fit the equation that maps frequency to resistance.
    '''
    
    # User argument sanitation:
    if not (len(measured_qubit_frequencies) == len(measured_junction_resistances)):
        raise ValueError("Halted! The list measured_qubit_frequencies must have an equal number of entries as the list measured_junction_resistances.")
    
    # Sort the lists. Sort by the first list, and unzip them
    sorted_pairs = sorted(zip(measured_junction_resistances, measured_qubit_frequencies))
    sorted_resistances, sorted_frequencies = zip(*sorted_pairs)
    measured_junction_resistances = (list(sorted_resistances)).copy()
    measured_qubit_frequencies    = (list(sorted_frequencies)).copy()
    
    # Define the equation to fit to.
    def ambegaokar_baratoff_josephson_koch(
        room_temperature_resistance,
        E_C_in_Hz,
        Delta_cold_eV,
        ##difference_between_RT_and_cold_resistance,
        ##T
        ):
        # Physical constants
        h = 6.62607015e-34       # Planck's constant [J/Hz]
        h_bar = h / (2 * np.pi)  # Reduced Planck's constant [J/Hz]
        e = 1.602176634e-19      # Elementary charge [C]
        k_B = 1.380649e-23       # Boltzmann's constant [J/K]
        
        # User-set values
        ## Calculate the normal state resistance of the S-I-S junction.
        R_N = room_temperature_resistance * difference_between_RT_and_cold_resistance
        Delta_cold = Delta_cold_eV * e # Superconducting gap at mK temperature [J]
        E_C = E_C_in_Hz * h # Charging energy [J]
        
        # Calculate I_c using the Ambegaokar-Baratoff relation
        I_c = (np.pi * Delta_cold)/(2*e*R_N) ## * np.tanh(Delta_cold / (2 * k_B * T))
        
        # Calculate E_J
        E_J = (h_bar / (2*e)) * I_c
        
        # Calculate f_01
        ## Koch 2007 equation regarding transmon f_01, precision to second order.
        ## https://doi.org/10.1103/PhysRevB.77.180502
        second_order_correction_factor = -(E_C / 2) * (E_C / (8*E_J))
        
        # Return answer.
        return (np.sqrt(8 * E_J * E_C) -E_C + second_order_correction_factor)/h
    
    # Create figure for plotting.
    if verbose:
        if colourise:
            fig, ax = plt.subplots(figsize=(12, 10), facecolor=get_colourise(-2))
        else:
            fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot the data, and its fitted-to curve,
    ## Get the fit and its data.
    optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
        f     = ambegaokar_baratoff_josephson_koch,
        xdata = measured_junction_resistances,
        ydata = measured_qubit_frequencies,
        p0    = (E_C_in_Hz, Delta_cold_eV)
    )
    ###(E_C_in_Hz, Delta_cold_eV, difference_between_RT_and_cold_resistance, T)
    fit_x_values = np.linspace(measured_junction_resistances[0]*0.90, measured_junction_resistances[-1]*1.10, 100)
    fitted_curve = ambegaokar_baratoff_josephson_koch(
        room_temperature_resistance = fit_x_values,
        E_C_in_Hz = optimal_vals[0],
        Delta_cold_eV = optimal_vals[1],
        ##difference_between_RT_and_cold_resistance = optimal_vals[2],
        ##T = optimal_vals[3],
    )
    
    # Get the fit error.
    fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
    err_E_C_in_Hz = fit_err[0]
    err_Delta_cold_eV = fit_err[1]
    ##err_difference_between_RT_and_cold_resistance = fit_err[2]
    ##err_T = fit_err[3]
    
    # Do teh plottings!
    if verbose:
        if colourise:
            plt.scatter(measured_junction_resistances, measured_qubit_frequencies, color=get_colourise(colourise_counter), label=f"Measured data.")
            colourise_counter += 1
            ## Plot the ideal curve.
            plt.plot(fit_x_values, fitted_curve, color=get_colourise(colourise_counter))
            colourise_counter += 1
        else:
            plt.scatter(measured_junction_resistances, measured_qubit_frequencies, color="#34D2D6", label=f"Measured data")
            plt.plot(fit_x_values, fitted_curve, '--', color="#D63834")
    
    # Labels and such.
    if verbose:
        plt.grid()
        if colourise:
            fig.patch.set_alpha(0)
            ax.grid(color=get_colourise(-1))
            ax.set_facecolor(get_colourise(-2))
            ax.spines['bottom'].set_color(get_colourise(-1))
            ax.spines['top'].set_color(get_colourise(-1))
            ax.spines['left'].set_color(get_colourise(-1))
            ax.spines['right'].set_color(get_colourise(-1))
            ax.tick_params(axis='both', colors=get_colourise(-1))
    
        # Bump up the size of the ticks' numbers on the axes.
        ax.tick_params(axis='both', labelsize=23)
    
        # Fancy colours?
        if (not colourise):
            plt.xlabel("Resistance [Ω]", fontsize=33)
            plt.ylabel("Qubit plasma frequency [Hz]", fontsize=33)
            plt.title(f"Qubit frequency vs. resistance", fontsize=38)
        else:
            plt.xlabel("Resistance [Ω]", color=get_colourise(-1), fontsize=33)
            plt.ylabel("Qubit plasma frequency [Hz]", color=get_colourise(-1), fontsize=33)
            plt.title(f"Qubit frequency vs. resistance", color=get_colourise(-1), fontsize=38)
        
        # Show shits.
        plt.legend(fontsize=26)
        plt.show()
    
    # Print shits.
    if verbose:
        print("E_C: "+str(optimal_vals[0])+" ±"+str(fit_err[0])+" Hz")
        print("Delta: "+str(optimal_vals[1])+" ±"+str(fit_err[1])+" eV")
        ##print("Diff. R vs. R_N: "+str(optimal_vals[2])+" ±"+str(fit_err[2])+" %")
        ##print("T: "+str(optimal_vals[3])+" ±"+str(fit_err[3])+" K")
        
    # Calculate frequency differences.
    diff_list = []
    for fif in range(len(measured_qubit_frequencies)):
        current_measured_frequency  = measured_qubit_frequencies[fif]
        current_measured_resistance = measured_junction_resistances[fif]
        
        current_predicted_frequency = calculate_f01_from_RT_resistance(
            room_temperature_resistance = current_measured_resistance,
            E_C_in_Hz = 195e6,#optimal_vals[0],
            Delta_cold_eV = 172.48e-6,#optimal_vals[1],
            difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
            T = T,
            verbose = verbose
        )
        
        # Positive = "Your value was above the prediction"
        # Negative = "Your value was below the prediction"
        curr_difference = current_measured_frequency - current_predicted_frequency
        
        # Append to list of stats.
        diff_list.append(curr_difference)
    
    return optimal_vals, fit_err, diff_list

def run_fits_of_ambegaokar_baratoff_josephson_koch(
    measured_junction_resistances,
    measured_qubit_frequencies,
    E_C_in_Hz_guess_list,
    Delta_cold_eV_guess_list,
    ##difference_between_RT_and_cold_resistance_guess_list,
    acceptable_limits = [(100e6, 500e6), (105e-6, 285e-6)]##, (0.95, 1.25)]
    ##T = 0.010,
    ):
    ''' Try different values until the fit works out.
    '''
    
    # Set flags.
    success = False
    attempts = 0
    try:
        total_attempts_to_do = len(E_C_in_Hz_guess_list) * len(Delta_cold_eV_guess_list) * len(difference_between_RT_and_cold_resistance_guess_list)
    except NameError:
        total_attempts_to_do = len(E_C_in_Hz_guess_list) * len(Delta_cold_eV_guess_list)
    
    for E_C_in_Hz_current in E_C_in_Hz_guess_list:
        for Delta_cold_eV_current in Delta_cold_eV_guess_list:
            ##for difference_between_RT_and_cold_resistance_current in difference_between_RT_and_cold_resistance_guess_list:
            try:
                optimal_vals, fit_err = fit_ambegaokar_baratoff_josephson_koch_to_resistance(
                    measured_junction_resistances = measured_junction_resistances,
                    measured_qubit_frequencies = measured_qubit_frequencies,
                    E_C_in_Hz = E_C_in_Hz_current,
                    Delta_cold_eV = Delta_cold_eV_current,
                    ##difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance_current,
                    ##T = T,
                    colourise = False,
                    verbose = False
                )
                
                # Check limits:
                if (optimal_vals[0] >= acceptable_limits[0][0]) and (optimal_vals[0] <= acceptable_limits[0][1]):
                    # E_C was fine!
                    if (optimal_vals[1] >= acceptable_limits[1][0]) and (optimal_vals[1] <= acceptable_limits[1][1]):
                        # Delta was fine!
                        ##if (optimal_vals[2] >= acceptable_limits[2][0]) and (optimal_vals[2] <= acceptable_limits[2][1]):
                        ## # Resistance mapping was fine!
                        
                        # Jolly! Report!
                        print("--------------------------------------------------\nLegal values found!")
                        print("E_C: "+str(optimal_vals[0])+" Hz ±"+str(fit_err[0]))
                        print("Delta: "+str(optimal_vals[1])+" eV ±"+str(fit_err[1]))
                        ##print("R_RT to R_N: "+str(optimal_vals[2])+" ±"+str(fit_err[2]))
                        
                        success = True
            except RuntimeError:
                pass
            
            attempts += 1
            if ((attempts/total_attempts_to_do) % 0.025) == 0:
                print("Attempts made: "+str(attempts)+", "+str(attempts/total_attempts_to_do)+"% done.")
    
    if (not success):
        print("Failed to find working set of parameters.")
    
def plot_fourier_transform_of_resistance_relaxation(
    filepath,
    normalise_resistances = 0,
    normalise_time_to_relaxation_effect = False,
    attempt_to_color_plots_from_file_name = False,
    plot_no_junction_resistance_under_ohm = 0
    ):
    ''' Take the resistance-over-time data, and plot the FFT.
    '''
    
    # User input formatting.
    if isinstance(filepath, str):
        filepath = [filepath]
    elif isinstance(filepath, (tuple, set)):
        filepath = list(filepath)
    elif isinstance(filepath, dict):
        filepath = list(filepath.keys())
    elif not isinstance(filepath, list):
        # Wrap it.
        filepath = [filepath]
    
    # Create figure for plotting.
    plt.figure(figsize=(10, 5))
    
    # Create list that will keep track of where the "time = 0" points are
    # in the files.
    zero_points = np.zeros_like(filepath)
    zz = 0
    
    # Go through the files and add them to the plot.
    for jj in range(len(filepath)):
        filepath_item = filepath[jj]
    
        # Initialise values.
        zero_points[zz] = 0
        times = []
        resistances = []
        first_time_value_has_been_checked = False
        zero_point_was_found = False
        time_offset_due_to_appended_data = 0
        si_unit_prefix_scaler = 1.0
        resistance_at_relaxation_start = 0
        add_this_time_offset_too = 0
        
        with open(os.path.abspath(filepath_item), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            rows = list(reader)  # Convert to list for indexing options
            
            # Go through the file.
            for i in range(len(rows)):
                if i % 6 == 3:
                    
                    # Every sixth row +3 contains a resistance value
                    current_resistance = float(rows[i][1])
                    
                    # Get the SI prefix for this data.
                    ## TODO append more options, like MOhm.
                    if '[kOhm]' in str(rows[i][0]):
                        si_unit_prefix_scaler = 1000
                    else:
                        si_unit_prefix_scaler = 1
                    
                    # Scale to Ohm
                    current_resistance *= si_unit_prefix_scaler
                    
                    # Plot junction? (i.e., plot broken junctions?)
                    if current_resistance > plot_no_junction_resistance_under_ohm:
                        # Super, append junction and continue.
                        resistances.append(current_resistance)
                        
                        # Every sixth row +4 contains a time value
                        time_value = float(rows[i+1][1])
                        
                        # Check whether a new measurement appended data onto the old one.
                        if time_value == 0:
                            if not first_time_value_has_been_checked:
                                # The very first entry is time = 0, do not append.
                                first_time_value_has_been_checked = True
                            else:
                                if time_offset_due_to_appended_data > 0:
                                    print("WARNING: file \""+str(filepath_item)+"\" contains multiple \"time = 0\" points.")
                                    add_this_time_offset_too = times[-1]
                                else:
                                    # Set zero point, i.e., start of relaxation.
                                    zero_points[zz] = time_offset_due_to_appended_data
                                    zero_point_was_found = True
                                    
                                    # Log the resistance at the relaxation's start
                                    resistance_at_relaxation_start = current_resistance
                                    
                                # Log time for the appended data.
                                ## I.e., time_value == 0, and it's not the first
                                ## data value in the time series.
                                time_offset_due_to_appended_data = times[-1]
                                
                        # Append time value.
                        times.append( time_value + time_offset_due_to_appended_data + add_this_time_offset_too )
        
        # Ensure lists are the same length
        min_length = min(len(times), len(resistances))
        times = times[:min_length]
        resistances = resistances[:min_length]
        
        # Normalise resistance axis?
        resistances = np.array(resistances, dtype=np.float64)
        if normalise_resistances == 2:
            # Check whether the start of the relaxation was found.
            if resistance_at_relaxation_start > 0:
                resistances = (resistances / resistance_at_relaxation_start) - 1
                plt.ylabel("Resistance normalised to relaxation effect start [-]")
            else:
                # In that case, just take the last value and normalise to that.
                resistances = (resistances / resistances[-1]) - 1
        elif normalise_resistances == 1:
            resistances = (resistances / resistances[0]) - 1
            plt.ylabel("Resistance normalised to starting value [-]")
        else:
            plt.ylabel("Resistance [Ω]")
        
        # Normalise time axis?
        if normalise_time_to_relaxation_effect:
            # Then scale the time axis accordingly.
            times = np.array(times, dtype=np.float64)
            times = times - time_offset_due_to_appended_data
            if not zero_point_was_found:
                # In that case, the measurement was likely aborted.
                # Subtract the highest time value.
                times -= times[-1]
        
        # Iterate zero_points index counter.
        zz += 1
        
        # Add item to plot!
        ## Get the file label name.
        file_label = str(os.path.splitext(os.path.basename(filepath_item))[0])
        
        ## Attempt to find the chip identity and the junction position.
        ## Extract channel number (ChXX)
        ch_match = re.search(r'Ch(\d+)_', filepath_item)
        chip_number = ch_match.group(1) if ch_match else None

        ## Extract TR/BL prefix (TRX or BLX)
        tr_bl_match = re.search(r'_(tr|bl)\d+', filepath_item.lower())
        tr_bl = tr_bl_match.group(1) if tr_bl_match else None
        
        # Determine color for trace?
        if ((chip_number is not None) and (tr_bl is not None)) and (attempt_to_color_plots_from_file_name):
            raise NotImplementedError("Halted. Not implemented.")
            '''hex_color_string = '#'
            
            # Red
            ##hex_color_string += f"{int(((int(chip_number) - 1) / 26) * 255):02X}"
            if int(chip_number)-1 <= 13:
                hex_color_string += f"{randint(0,30):02X}"
            else:
                hex_color_string += f"{randint(204,225):02X}"
            
            # Green
            if tr_bl == "tr":
                hex_color_string += f"{randint(0,30):02X}"
            elif tr_bl == "bl":
                hex_color_string += f"{randint(204,225):02X}"
            else:
                raise ValueError("ERROR: Could not determine TL/BR for this measurement data file, even though the file name seemed to still match expectations.")
            
            # Blue
            hex_color_string += f"{randint(10,225):02X}"
            
            # Plot!
            plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=hex_color_string)'''
        else:
            # Just plot from a map.
            num_items_to_colour = len(filepath)
            colours = plt.cm.get_cmap('tab20', num_items_to_colour)
            
            ## Perform Fourier transform things.
            # Define zero-padding factor
            zero_padding_factor = 64  # Increase this for even finer resolution

            # Compute the next power of two for zero-padding
            n_fft = len(resistances) * zero_padding_factor  

            # Compute FFT with zero-padding
            fft_values = np.fft.fft(resistances, n=n_fft)  
            freqs = np.fft.fftfreq(n_fft, d=(times[1] - times[0]))  # Proper frequency scaling

            # Plot the magnitude spectrum
            plt.plot(freqs[:n_fft // 2], np.abs(fft_values[:n_fft // 2]))  # Keep positive frequencies

            
            # Plot the magnitude spectrum
            ###plt.figure(figsize=(8, 4))
            ###plt.plot(freqs, np.abs(fft_values))
            plt.xlabel("Frequency")
            plt.ylabel("Magnitude")
            plt.title("FFT: resistances over time")
            plt.grid()
            plt.show()
            
            ##plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colours[jj])## TODO!! colours(jj))
        
    ##plt.xlabel("Duration [s]")
    ##plt.title("Resistance vs. Time")
    ##plt.grid()
    ##plt.legend()
    ##plt.show()
    
def plot_josephson_junction_resistance_manipulation_and_relaxation(
    filepath,
    normalise_resistances = 2,
    normalise_time = 0,
    attempt_to_color_plots_from_file_name = False,
    plot_no_junction_resistance_under_ohm = 0,
    colourise = False,
    savepath = '',
    ):
    ''' Plot the data from the resistance manipulation and ensueing
        resistance relaxation.
        
        normalise_resistances will:
            0:  do not normalise resistances,
            1:  set the first measured resistance value as the initial
                resistance of the device; all subsequent values
                will be reported as a percentage of this initial value.
            2:  same as 1, but the resistance relaxation's datapoint_0
                will be the resistance that is normalised to.
        
        normalise_time will:
            0: plot the UNIX timestamp on the x axis.
            1: normalise to the relaxation effect.
            2: normalise to the very beginning of the measurement.
    '''
    
    # Colourise counter, keeping track of the colour formatting.
    colourised_counter = 0
    
    # User input formatting.
    if isinstance(filepath, str):
        filepath = [filepath]
    elif isinstance(filepath, (tuple, set)):
        filepath = list(filepath)
    elif isinstance(filepath, dict):
        filepath = list(filepath.keys())
    elif not isinstance(filepath, list):
        # Wrap it.
        filepath = [filepath]
    
    # Create figure for plotting.
    if colourise:
        fig, ax = plt.subplots(figsize=(12.8, 13), facecolor=get_colourise(-2))
        #plt.figure(figsize=(10, 5), facecolor=get_colourise(-2))
    else:
        # Used for methods example demonstration.
        ##fig, ax = plt.subplots(figsize=(13.58, 9.78)) #### (b) ####
        ##fig, ax = plt.subplots(figsize=(14.045, 9.75)) #### (d) ####
        
        # Used for the stepped_active_manipulation data (sub)plots.
        fig, ax = plt.subplots(figsize=(12.34, 13))
    
    # Create list that will keep track of where the "time = 0" points are
    # in the files.
    zero_points = np.zeros_like(filepath)
    zz = 0
    
    # Go through the files and add them to the plot.
    for jj in range(len(filepath)):
        filepath_item = filepath[jj]
        
        # Initialise values.
        zero_points[zz] = 0
        times = []
        resistances = []
        first_time_value_has_been_checked = False
        zero_point_was_found = False
        time_offset_due_to_appended_data = 0
        first_time_value_detected = -1.0
        si_unit_prefix_scaler = 1.0
        resistance_at_relaxation_start = 0
        add_this_time_offset_too = 0
        obvious_short = 100 # [Ω]  --  Define a resistance that defines "a short."
        lowest_non_short_resistance_in_set = 1000000000
        
        with open(os.path.abspath(filepath_item), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            rows = list(reader)  # Convert to list for indexing options
            
            # Go through the file.
            for i in range(len(rows)):
                if i % 6 == 3:
                    
                    # Every sixth row +3 contains a resistance value
                    current_resistance = float(rows[i][1])
                    
                    # Get the SI prefix for this data.
                    ## TODO append more options, like MOhm.
                    if '[kOhm]' in str(rows[i][0]):
                        si_unit_prefix_scaler = 1000
                    else:
                        si_unit_prefix_scaler = 1
                    
                    # Scale to Ohm
                    current_resistance *= si_unit_prefix_scaler
                    
                    # Update the lowest resistance found!
                    if (current_resistance < lowest_non_short_resistance_in_set) and (current_resistance > obvious_short):
                        lowest_non_short_resistance_in_set = current_resistance
                    
                    # Plot junction? (i.e., plot broken junctions?)
                    if current_resistance > plot_no_junction_resistance_under_ohm:
                        # Super, append junction and continue.
                        resistances.append(current_resistance)
                        
                        # Every sixth row +4 contains a time value
                        time_value = float(rows[i+1][1])
                        
                        ## Was this a UNIX timestamp that we should offset for?
                        if first_time_value_detected == -1.0:
                            first_time_value_detected = time_value
                        
                        # Check whether a new measurement appended data onto the old one.
                        if time_value == 0:
                            if not first_time_value_has_been_checked:
                                # The very first entry is time = 0, do not append.
                                first_time_value_has_been_checked = True
                            else:
                                if time_offset_due_to_appended_data > 0:
                                    print("WARNING: file \""+str(filepath_item)+"\" contains multiple \"time = 0\" points.")
                                    add_this_time_offset_too = times[-1]
                                else:
                                    # Set zero point, i.e., start of relaxation.
                                    zero_points[zz] = time_offset_due_to_appended_data
                                    zero_point_was_found = True
                                    
                                    # Log the resistance at the relaxation's start
                                    resistance_at_relaxation_start = current_resistance
                                    
                                # Log time for the appended data.
                                ## I.e., time_value == 0, and it's not the first
                                ## data value in the time series.
                                time_offset_due_to_appended_data = times[-1]
                                
                        # Append time value.
                        times.append( time_value + time_offset_due_to_appended_data + add_this_time_offset_too )
        
        # Ensure lists are the same length
        min_length = min(len(times), len(resistances))
        times = times[:min_length]
        resistances = resistances[:min_length]
        
        # Normalise resistance axis?
        resistances = np.array(resistances, dtype=np.float64)
        
        if normalise_resistances == 3:
            # In this case, use both y axes of normalise_resistances 1 and 2.
            resistances_ohm = resistances.copy()
            resistances_pct = ((resistances / resistances[0]) - 1) * 100
        else:
            if normalise_resistances == 2:
                # Check whether the start of the relaxation was found.
                if resistance_at_relaxation_start > 0:
                    resistances = (resistances / resistance_at_relaxation_start) - 1
                    y_label_text = "Resistance normalised to relaxation effect start [-]"
                else:
                    # In that case, just take the last value and normalise to that.
                    resistances = (resistances / resistances[-1]) - 1
                    y_label_text = "Resistance increase [%]"
            elif normalise_resistances == 1:
                resistances = ((resistances / resistances[0]) - 1) * 100
                y_label_text = "Resistance increase [%]"
            else:
                y_label_text = "Resistance [Ω]"
        
        # Normalise time axis to relaxation effect?
        if (normalise_time == 1):
            # Then scale the time axis accordingly.
            times = np.array(times, dtype=np.float64)
            times = times - time_offset_due_to_appended_data
            if not zero_point_was_found:
                # In that case, the measurement was likely aborted.
                # Subtract the highest time value.
                times -= times[-1]
        elif (normalise_time == 2):
            # Then cut away the UNIX timestamp taken at datapoint 0.
            times = np.array(times, dtype=np.float64)
            times -= first_time_value_detected
        
        # Iterate zero_points index counter.
        zz += 1
        
        # Add item to plot!
        ## Get the file label name.
        file_label = str(os.path.splitext(os.path.basename(filepath_item))[0])
        
        ## Attempt to find the chip identity and the junction position.
        ## Extract channel number (ChXX)
        ch_match = re.search(r'Ch(\d+)_', filepath_item)
        chip_number = ch_match.group(1) if ch_match else None

        ## Extract TR/BL prefix (TRX or BLX)
        tr_bl_match = re.search(r'_(tr|bl)\d+', filepath_item.lower())
        tr_bl = tr_bl_match.group(1) if tr_bl_match else None
        
        # Determine color for trace?
        if (not colourise):
            if ((chip_number is not None) and (tr_bl is not None)) and (attempt_to_color_plots_from_file_name):
                hex_color_string = '#'
                
                # Red
                ##hex_color_string += f"{int(((int(chip_number) - 1) / 26) * 255):02X}"
                if int(chip_number)-1 <= 13:
                    hex_color_string += f"{randint(0,30):02X}"
                else:
                    hex_color_string += f"{randint(204,225):02X}"
                
                # Green
                if tr_bl == "tr":
                    hex_color_string += f"{randint(0,30):02X}"
                elif tr_bl == "bl":
                    hex_color_string += f"{randint(204,225):02X}"
                else:
                    raise ValueError("ERROR: Could not determine TL/BR for this measurement data file, even though the file name seemed to still match expectations.")
                
                # Blue
                hex_color_string += f"{randint(10,225):02X}"
                
                # Plot!
                plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=hex_color_string)
            else:
                # Just plot from a map.
                num_items_to_colour = len(filepath)
                colours = plt.cm.get_cmap('tab20', num_items_to_colour)
                if savepath == '':
                    ## TODO remove this try-catch bs.
                    try:
                        plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colours[jj])## TODO!! colours(jj))
                    except TypeError:
                        plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=colours(jj)) ## TODO!!
                else:
                    dot_color = "#C4EE1C" ## TODO!
                    #dot_color = "#1CEE70" ## TODO!
                    #dot_color = "#EE1C1C" ## TODO!
                    if normalise_resistances == 3:
                        ##ax.plot(times, resistances_ohm, marker='o', linestyle='-', label=file_label + " [Ω]", color=dot_color)
                        print("TODO: dividing the time axis by 3600 to put it in hours instead of seconds.")
                        ax.plot(times/3600, resistances_ohm, marker='o', linestyle='-', label="Low-dose 1", color=dot_color) ##label="200x200 nm low_dose-oxide", color=dot_color)
                        #ax.plot(times/3600, resistances_ohm, marker='o', linestyle='-', label="Medium-dose 1", color=dot_color) ## label="350x350 nm medium_dose-oxide", color=dot_color)
                        #ax.plot(times/3600, resistances_ohm, marker='o', linestyle='-', label="High-dose 1", color=dot_color) ## label="318x318 nm high_dose-oxide", color=dot_color)
                        
                        if 'ax2' not in locals():
                            ax2 = ax.twinx()
                            ax2.set_ylabel("Resistance increase [%]", fontsize=33)
                            
                            # Set ylim limits!
                            
                            # Used for the stepped active manipulation examples.
                            bottom_percent = -5
                            top_percent = 280
                            
                            # Used for the (b) illustration example.
                            ##bottom_percent = -0.9
                            ##top_percent = 6.1
                            # Used for the (d) illustration example.
                            ##bottom_percent = -5
                            ##top_percent = 58
                            
                            # Begin with the percent axis. Simple, just set the numbers.
                            ax2.set_ylim(bottom_percent,top_percent)
                            
                            ## Ok, but for resistances, then this gymnastics
                            ## right here is a bit tricky. Say that
                            ## bottom_percent = -5, meaning, -5% is the bottom.
                            ## And, say that top_percent = 280, i.e.,
                            ## that the resist. axis top corresponds to +280%.
                            ## The solution is to count backwards.
                            ## (res/res[0] - 1)*100 = -5  # [%]
                            ## ⇒ res = (((-5)/100)+1)*res[0]  # [%]
                            ## ... where you replace -5% with bottom_percent
                            ##     or top_percent, as needed. res then goes
                            ##     into .ylim for the resistance axis in Ω.
                            
                            # Set the ordinary resistance axis!
                            ax.set_ylim(((bottom_percent/100)+1)*resistances_ohm[0], ((top_percent/100)+1)*resistances_ohm[0])
                        
                        print("TODO: the percent-x axis has been divided into hours, instead of seconds.")
                        ax2.plot(times/3600, resistances_pct, marker='o', linestyle='--', color=dot_color)
                        
                    else:
                        plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=dot_color)
        else:
            # Use patterned colour.
            plt.plot(times, resistances, marker='o', linestyle='-', label=file_label, color=get_colourise((colourised_counter // 4) + ((colourised_counter % 4) + 1) / 10))
            colourised_counter += 1
        
    
    # Set axes' colour? Title colour? And so on.
    if colourise:
        fig.patch.set_alpha(0)
        ax.grid(color=get_colourise(-1))
        ax.set_facecolor(get_colourise(-2))
        ax.spines['bottom'].set_color(get_colourise(-1))
        ax.spines['top'].set_color(get_colourise(-1))
        ax.spines['left'].set_color(get_colourise(-1))
        ax.spines['right'].set_color(get_colourise(-1))
        #ax.set_xlabel('X Label', color=get_colourise(-1))
        #ax.set_ylabel('Y Label', color=get_colourise(-1))
        ax.tick_params(axis='both', colors=get_colourise(-1))
        plt.grid()
    
    
    if normalise_resistances == 3:
        ax.set_ylabel("Resistance [Ω]", fontsize=33)
        ax.grid()
        if colourise:
            ax2.tick_params(axis='y', colors=get_colourise(-1), labelsize=23)
            ax2.spines['right'].set_color(get_colourise(-1))
            ax2.yaxis.label.set_color(get_colourise(-1))
        ax.tick_params(axis='y', labelsize=26)
        if 'ax2' in locals():
            ax2.tick_params(axis='y', labelsize=26)
    else:
        ax.set_ylabel(y_label_text, fontsize=33)
        ax.grid()
    
    # Set x-axis text.
    if normalise_time == 0:
        ax.set_xlabel("Time [s since epoch]", fontsize=33)
    elif normalise_time == 1:
        ax.set_xlabel("Time since resistance relaxation onset [s]", fontsize=33)
    elif normalise_time == 2:
        print("TODO: the x axis may have been divided into hours, instead of seconds. Check the unit in the plot if unsure.")
        ax.set_xlabel("Duration [h]", fontsize=33)
    else:
        ax.set_xlabel("Time [s]", fontsize=33)
    
    # Bump up the size of the ticks' numbers on the axes.
    ax.tick_params(axis='both', labelsize=26)
    
    plt.xlabel("Duration [s]", color=get_colourise(-1), fontsize=33)
    ##plt.ylabel(y_label_text, color=get_colourise(-1), fontsize=33)
    if savepath == '':
        plt.title("Resistance vs. Time", color=get_colourise(-1), fontsize=38)
    
    # Legend and layout.
    if normalise_resistances == 3:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=26)
    else:
        ax.legend()
    
    # Tight layout.
    plt.tight_layout()
    
    # Adjust x-axis?
    ## Normally, leave this commented-out.
    ## For the (b) figure in the resistance manipulation curves methods
    ## example, use the following settings to reproduce the plot:
    #ax.set_xlim(-0.86, 5.90)
    
    # Save plots?
    if savepath != '':
    
        # Fix path name?
        thin = ('JJAnneal01C' in filepath[0]) and ('stepped' in filepath[0])
        thick = ('MPW005A' in filepath[0]) and ('stepped' in filepath[0])
        base, _ = os.path.splitext(savepath)
        if thin:
            new_savepath = f"{base}_soft_stepped.png"
        elif thick:
            new_savepath = f"{base}_hard_stepped.png"
        else:
            new_savepath = f"{base}.png"
        plt.savefig(new_savepath, dpi=164, bbox_inches='tight')
    
    # Show shits.
    plt.show()

def simulate_frequency_accuracy_of_model_from_RT_resistance(
    no_junctions,
    resistance,
    resistance_measurement_error_std_deviation,
    E_C_mean_in_Hz,
    E_C_error_std_deviation_in_Hz,
    Delta_mean_eV,
    Delta_error_std_deviation_eV,
    temperature_mean,
    temperature_std_deviation,
    difference_between_RT_and_cold_resistance_mean,
    difference_between_RT_and_cold_resistance_std_dev,
    plot = True
    ):
    ''' Virtually manufactures no_junctions worth of junctions,
        and produces a distribution, showing what the frequency
        accuracy actually is.
        
        resistance_measurement_error_mean is the error bar
        of the resistance measurement itself.
    '''
    
    # Given a chip's RT resistance, what is the accuracy of the model?
    ## First, we have a resistance measurement error.
    ## Create some distribution around a mean, which is our measured value.
    resistance_with_meas_error = np.random.normal(resistance, resistance_measurement_error_std_deviation, no_junctions)
    
    # Given some known E_C, and some known deviation onto it,
    # randomly sample some E_C values.
    E_C_with_error = np.random.normal(E_C_mean_in_Hz, E_C_error_std_deviation_in_Hz, no_junctions)
    
    # Given some known superconducting energy gap, and its error, get values.
    Delta_with_error_eV = np.random.normal(Delta_mean_eV, Delta_error_std_deviation_eV, no_junctions)
    error_Delta_in_text = str(f"{(Delta_error_std_deviation_eV / Delta_mean_eV)*100:.3f}")
    
    # Given some known temperature, and its error, get values.
    temperature_with_error = np.random.normal(temperature_mean, temperature_std_deviation, no_junctions)
    ##error_temperature_in_text = str(f"{(temperature_std_deviation / temperature_mean)*100:.3f}")
    error_temperature_in_text = str(f"{(temperature_std_deviation*1000):.1f}")
    
    # Given some known difference between room temperature resistance
    # and the normal state resistance at mK, get values.
    diff_rt_R_with_error = np.random.normal(difference_between_RT_and_cold_resistance_mean, difference_between_RT_and_cold_resistance_std_dev, no_junctions)
    error_diff_R_in_text = str(f"{(difference_between_RT_and_cold_resistance_std_dev / difference_between_RT_and_cold_resistance_mean)*100:.3f}")
    
    # What frequency does those resistances correspond to?
    frequencies_calculated = []
    for jj in range(no_junctions):
        frequencies_calculated.append(
            calculate_f01_from_RT_resistance(
                room_temperature_resistance = resistance_with_meas_error[jj],
                E_C_in_Hz = E_C_with_error[jj],
                Delta_cold_eV = Delta_with_error_eV[jj],
                difference_between_RT_and_cold_resistance = diff_rt_R_with_error[jj],
                T = temperature_with_error[jj],
                verbose = False
            )
        )
    
    # Get the standard deviation of the calculated frequencies.
    frequencies_calculated_standard_deviation = np.std(frequencies_calculated, ddof=0)
    ## Here, I am not using the sample standard deviation.
    ## Isn't it so that I know the full population? TODO
    
    # Plot the expected normal distribution curve!
    ''' This code snippet was implemented from Christian Križan's
        research work in https://arxiv.org/abs/2412.15022 '''
    
    ## Here, Sturge's formula along with Doane's correction factor is used
    ## for getting a decent number of bins.
    no_entries = len(frequencies_calculated)
    third_moment_skewness_of_distribution = moment(frequencies_calculated, moment = 3) # Get the assymetry of the distribution
    sigma_g1 = np.sqrt( (6*(no_entries - 2))/((no_entries + 1)*(no_entries + 3)) )
    doane_correction_factor_Ke = np.log2(1 + np.abs(third_moment_skewness_of_distribution)/sigma_g1)
    bins_calculated = int(np.ceil(1 + np.log2( no_entries ) + doane_correction_factor_Ke))
    
    # Plot histogram of the calculated frequency values
    if plot:
        plt.figure(figsize=(18,6))
        plt.hist(frequencies_calculated, bins=bins_calculated, density=True, alpha=0.6, color="#1C70EE", edgecolor='black', rwidth = 0.9)
        num_sigmas_in_expected_pdf = 5
        
        # Create trace for the expected probability distribution
        x = np.linspace(
            np.mean(frequencies_calculated) - float(num_sigmas_in_expected_pdf*frequencies_calculated_standard_deviation),
            np.mean(frequencies_calculated) + float(num_sigmas_in_expected_pdf*frequencies_calculated_standard_deviation),
            100
        )
        pdf = (1 / (frequencies_calculated_standard_deviation * np.sqrt(2 * np.pi))) * \
            np.exp(-0.5 * ((x - np.mean(frequencies_calculated)) / frequencies_calculated_standard_deviation) ** 2)
        plt.plot(x, pdf, '-', color="#EE1C1C", label="Expected normal\ndistribution")
        
        # Labels and title
        plt.xlabel("Calculated frequencies [GHz]", fontsize=33)
        plt.ylabel("Probability density", fontsize=33)
        plt.title("Distribution about frequency target:\n±"+str(resistance_measurement_error_std_deviation)+" Ω measurement error, ±"+str(E_C_error_std_deviation_in_Hz/1e6)+" MHz E_C,\n±"+str(error_Delta_in_text)+"% Δ, ±"+str(error_temperature_in_text)+" mK T, ±"+str(error_diff_R_in_text)+"% R vs R_T", fontsize=18)
        plt.tick_params(axis='both', labelsize=24)
        plt.legend(fontsize=26)
        plt.xlim(2.5e9, 5.0e9)
        plt.ylim(0, 1.25e-8)
        plt.show()
    
    # Print some things.
    print("Mean frequency is: "+str(np.mean(frequencies_calculated))+" [Hz]")
    print("Standard deviation for the frequency is: "+str(frequencies_calculated_standard_deviation)+" [Hz]")
    
    # Get values for the return, and return.
    final_mean = np.mean(frequencies_calculated)
    final_std = frequencies_calculated_standard_deviation
    return (final_mean, final_std)

def plot_trend_for_changing_superconducting_gap(
    list_of_doubles_of_Delta_eV_and_Delta_std_eV,
    no_junctions,
    resistance,
    resistance_measurement_error_std_deviation,
    E_C_mean_in_Hz,
    E_C_error_std_deviation_in_Hz,
    Delta_mean_eV,
    Delta_error_std_deviation_eV,
    temperature_mean,
    temperature_std_deviation,
    difference_between_RT_and_cold_resistance_mean,
    difference_between_RT_and_cold_resistance_std_dev,
    ):
    ''' Given a list of doubles, containing (Delta_mean, Delta_std),
        calculate the resulting frequencies and their standard deviation.
        Finally, plot.
    '''
    
    # Get calculated frequencies.
    output_values = []
    inserted_std_values = []
    for ii in range(len(list_of_doubles_of_Delta_eV_and_Delta_std_eV)):
        inserted_std_values.append( list_of_doubles_of_Delta_eV_and_Delta_std_eV[ii][1] )
        output_values.append(
            simulate_frequency_accuracy_of_model_from_RT_resistance(
                no_junctions = no_junctions,
                resistance = resistance,
                resistance_measurement_error_std_deviation = resistance_measurement_error_std_deviation,
                E_C_mean_in_Hz = E_C_mean_in_Hz,
                E_C_error_std_deviation_in_Hz = E_C_error_std_deviation_in_Hz,
                Delta_mean_eV = list_of_doublers_of_Delta_eV_and_Delta_std_eV[ii][0],
                Delta_error_std_deviation_eV = inserted_std_values[ii],
                temperature_mean = temperature_mean,
                temperature_std_deviation = temperature_std_deviation,
                difference_between_RT_and_cold_resistance_mean = difference_between_RT_and_cold_resistance_mean,
                difference_between_RT_and_cold_resistance_std_dev = difference_between_RT_and_cold_resistance_std_dev,
                plot = False
            )
        )
    
    # Unpack data.
    means, error_bars = zip(*output_values)
    
    # Plot!
    plt.figure(figsize=(8, 5))
    #plt.errorbar(range(len(means)), means, yerr=error_bars, fmt='o', linestyle='-', color='orange', label='Simulated frequency')
    plt.plot(inserted_std_values, error_bars, marker='o', linestyle='-', color='orange')
    plt.xlabel("std_dev of Δ [eV]")
    plt.ylabel("Simulated frequency std_dev [Hz]")
    plt.title("Improving accuracy of Δ")
    #plt.legend()
    plt.grid(True)
    plt.show()

def plot_manipulation_plan(
    expected_resistance_relaxation = 1.0229,
    E_C_in_Hz = 195e6,
    Delta_cold_eV = 174.28e-6,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    verbose = False
    ):
    ''' Illustrate how to manipulate qubits on a chip.
    '''
    
    # TODO user-settable    
    original_resistances = [
        5.749e3, 6.045e3, 6.334e3, 6.653e3,
        7.411e3, 7.479e3, 7.541e3, 7.979e3
    ]
    original_frequencies = []
    for jj in range(len(original_resistances)):
        resistance_item = original_resistances[jj]
        original_frequencies.append(
            calculate_f01_from_RT_resistance(
                room_temperature_resistance = resistance_item,
                E_C_in_Hz = E_C_in_Hz,
                Delta_cold_eV = Delta_cold_eV,
                difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
                T = T,
                verbose = False
            )
        )
    
    # Target frequencies.
    '''## Find the top and bottom values first.
    top_freq = calculate_f01_from_RT_resistance(
        room_temperature_resistance = original_resistances[0] * 1.143, # Statistically survivable
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = False
    )
    bot_freq = calculate_f01_from_RT_resistance(
        room_temperature_resistance = original_resistances[-1] * 1.0143, # Statistically survivable
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = False
    )'''
    target_frequencies = np.linspace(original_frequencies[0], original_frequencies[-1], len(original_frequencies))
    
    res_of_m = original_resistances[jj]
    min_res_increased_for_m = res_of_m * 1.0238 # 75% survival for n=8
    new_m = calculate_f01_from_RT_resistance(
        room_temperature_resistance = min_res_increased_for_m,
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = True
    )
    target_frequencies[4] = new_m
    slope_of_old_line = (original_frequencies[-1] - original_frequencies[0]) / len(original_frequencies)
    
    target_frequencies[5] = slope_of_old_line * 1 + new_m
    target_frequencies[6] = slope_of_old_line * 2 + new_m
    target_frequencies[7] = slope_of_old_line * 3 + new_m
    
    target_frequencies[3] = slope_of_old_line * -1 + new_m
    target_frequencies[2] = slope_of_old_line * -2 + new_m
    target_frequencies[1] = slope_of_old_line * -3 + new_m
    target_frequencies[0] = slope_of_old_line * -4 + new_m
    
    calcualted_res_to_manip = []
    for jj in range(len(target_frequencies)):
        frequency_item = target_frequencies[jj]
        calcualted_res_to_manip.append(
            calculate_resistance_to_manipulate_to(
                target_f_01 = frequency_item,
                E_C_in_Hz = E_C_in_Hz,
                Delta_cold_eV = Delta_cold_eV,
                original_resistance_of_junction = original_resistances[jj],
                expected_aging = 0,
                expected_resistance_relaxation = expected_resistance_relaxation,
                verbose = True
            )
        )
    
    ##data_points = [
    ##    5.43108914, 5.31263521, 5.19418128, 5.07572735,
    ##    4.95727342, 4.83881949, 4.72036556, 4.60191163
    ##]
    
    # Plot frequencies
    plt.figure(figsize=(8, 5))
    qubit_axis = np.linspace(1, len(target_frequencies), len(target_frequencies))
    plt.plot(qubit_axis, original_frequencies, marker='s', linestyle='--', color='r', label='Original frequencies')
    plt.plot(qubit_axis, target_frequencies, marker='o', linestyle='-', color='b', label='Target frequencies')
    
    # Labels and title
    plt.xlabel('Qubit')
    plt.ylabel('Frequency [Hz]')
    plt.title('Frequency manipulation plan')
    plt.legend()
    plt.grid(True)
    
    # Show the plot
    plt.show()

def plot_active_manipulation(
    filepath,
    normalise_resistances = 0,
    normalise_time = True,
    plot_no_junction_resistance_under_ohm = 0,
    fitter = 'none',
    skip_initial_drop = False,
    plot_fit_parameters_in_legend = False,
    colourise = False,
    title_label = None,
    enable_mask = False,
    savepath = ''
    ):
    ''' Plot soledly only the active manipulation.
        
        normalise_resistances will:
            0:  do not normalise resistances,
            1:  set the first measured resistance value as the initial
                resistance of the device; all subsequent values
                will be reported as a percentage of this initial value.
        
        normalise_time:
            True: the x-axis will be in seconds after the measurement started.
            False: the x-axis will be given in UNIX time.
        
        fitter:
            'none':         Perform no fitting.
            'second_order': Attempt fit to R(t) = R_0 + alpha·t + beta·t^2
            'third_order':  Attempt fit to R(t) = R_0 + alpha·t + beta·t^2 + delta·t^3
            'exponential':  Attempt fit to R(t) = R_0 + epsilon·e( t/t_0 · gamma)
            'power':        Attempt fit to R(t) = R_0 + A·t^B
        
        skip_initial_drop:
            If false, look for the string "START_MANIPULATION" in column 3
            of the .csv format.
    '''
    
    ## Already up here, let's define a few lists to be used for the
    ## return statement.
    list_of_traces_in_plot = []
    list_of_error_bars_of_traces_in_plot = []
    list_of_fit_parameter_labels = []
    
    # Initially, let's define some functions for the fitting.
    '''def second_order_func(t, t_0, R_0, alpha, beta):
        return R_0 + (alpha * (t-t_0)) + (beta * (t-t_0)**2)'''
    def second_order_func(t, alpha, beta):
        return (alpha * t) + (beta * t**2)
    
    '''def third_order_func(t, t_0, R_0, alpha, beta, delta):
        return R_0 + (alpha * (t-t_0)) + (beta * (t-t_0)**2) + (delta * (t-t_0)**3)'''
    def third_order_func(t, alpha, beta, delta):
        return (alpha * t) + (beta * t**2) + (delta * t**3)
    
    '''def exponential_func(t, t_0, R_0, epsilon, gamma, tau):
        ##return R_0 + epsilon * (np.e)**((t-t_0) * gamma)
        return R_0 + epsilon * (1 - (np.e)**((t-t_0)/tau * -gamma))'''
    def exponential_func(t, epsilon, gamma):
        return epsilon * (1 - (np.e)**(t * -gamma))
    
    '''def power_func(t, t_0, R_0, A, B):
        return R_0 + A * ((t-t_0)**B)'''
    def power_func(t, A, B):
        return A * (t**B)
    
    
    def active_increase_fitter(
        resistances,
        time,
        fitter
        ):
        ''' fitter:
            'second_order': Attempt fit to R(t) = R_0 + alpha·t + beta·t^2
            'third_order':  Attempt fit to R(t) = R_0 + alpha·t + beta·t^2 + delta·t^3
            'exponential':  Attempt fit to R(t) = R_0 + epsilon·e( t/t_0 · gamma)
            'power':        Attempt fit to R(t) = R_0 + A·t^B
        '''
        ## Let's guess initial guessing values.
        ## To guess the polynomial values, let's do something a bit
        ## interesting, and run np.polyfit, to fit a 2nd and 3rd order
        ## polynomial.
        ## Then, fit.
        
        t_0_guess = time[0]
        R_0_guess = resistances[0] # Should be R_0, will be overwritten if better numbers exist below.
        if fitter == 'second_order':
            coeffs_2nd = np.polyfit(time, resistances, 2) # [beta, alpha, R_0]
            R_0_guess, alpha_guess, beta_guess = coeffs_2nd[::-1]  ## Reverse order for readability
        elif fitter == 'third_order':
            coeffs_3rd = np.polyfit(time, resistances, 3) # [delta, beta, alpha, R_0]
            R_0_guess, alpha_guess, beta_guess, delta_guess = coeffs_3rd[::-1]
        elif fitter == 'exponential':
            gamma_guess   = 0.010  # Exponential rise (so, minus +gamma_guess). 1/100 s is a rough guess.
            epsilon_guess = 0.10   # Expect flattening off at ~10% increase.
            tau_guess     = 30     # Rough estimate of one experimental exp-slope I saw.
        elif fitter == 'power':
            A_guess = 0.007 # Pro-tip: check t=1 for this value, since A*(t=1)^borp = A
            B_guess = 0.9   # Must be between 0 and 1 for the expected flattening behaviour, right?
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Fit!
        if fitter == 'second_order':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = second_order_func,
                xdata = time,
                ydata = resistances,
                p0    = (alpha_guess, beta_guess)
            )
            """p0    = (t_0_guess, R_0_guess, alpha_guess, beta_guess)"""
        elif fitter == 'third_order':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = third_order_func,
                xdata = time,
                ydata = resistances,
                p0    = (alpha_guess, beta_guess, delta_guess)
            )
            """p0    = (t_0_guess, R_0_guess, alpha_guess, beta_guess, delta_guess)"""
        elif fitter == 'exponential':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = exponential_func,
                xdata = time,
                ydata = resistances,
                p0    = (epsilon_guess, gamma_guess)
            )
            """p0    = (t_0_guess, R_0_guess, epsilon_guess, gamma_guess, tau_guess)"""
        elif fitter == 'power':
            optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
                f     = power_func,
                xdata = time,
                ydata = resistances,
                p0    = (A_guess, B_guess)
            )
            """p0    = (t_0_guess, R_0_guess, A_guess, B_guess)"""
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Extract parameters.
        """optimal_t_0    = optimal_vals[0]
        optimal_R_0    = optimal_vals[1]
        if fitter == 'second_order':
            optimal_alpha = optimal_vals[2]
            optimal_beta  = optimal_vals[3]
        elif fitter == 'third_order':
            optimal_alpha = optimal_vals[2]
            optimal_beta  = optimal_vals[3]
            optimal_delta = optimal_vals[4]
        elif fitter == 'exponential':
            optimal_epsilon = optimal_vals[2]
            optimal_gamma   = optimal_vals[3]
            optimal_tau     = optimal_vals[4]
        elif fitter == 'power':
            optimal_A = optimal_vals[2]
            optimal_B = optimal_vals[3]
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))"""
        
        # Updated version without R_0 and without t_0, to try and force
        # the error bar down on the fits. because, it's larger than
        # the fitted parameter, which is a sign that the number of parameters
        # is too big.
        if fitter == 'second_order':
            optimal_alpha = optimal_vals[0]
            optimal_beta  = optimal_vals[1]
        elif fitter == 'third_order':
            optimal_alpha = optimal_vals[0]
            optimal_beta  = optimal_vals[1]
            optimal_delta = optimal_vals[2]
        elif fitter == 'exponential':
            optimal_epsilon = optimal_vals[0]
            optimal_gamma   = optimal_vals[1]
            optimal_tau     = optimal_vals[2]
        elif fitter == 'power':
            optimal_A = optimal_vals[0]
            optimal_B = optimal_vals[1]
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Get the fit errors.
        fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
        """err_t_0 = fit_err[0]
        err_R_0 = fit_err[1]
        if fitter == 'second_order':
            err_alpha = fit_err[2]
            err_beta  = fit_err[3]
        elif fitter == 'third_order':
            err_alpha = fit_err[2]
            err_beta  = fit_err[3]
            err_delta = fit_err[4]
        elif fitter == 'exponential':
            err_epsilon = fit_err[2]
            err_gamma   = fit_err[3]
            err_tau     = fit_err[4]
        elif fitter == 'power':
            err_A = fit_err[2]
            err_B = fit_err[3]
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))"""
        # See comment above regarding the error bars being larger than
        # the fitted parameters if R_0 and t_0 is included.
        if fitter == 'second_order':
            err_alpha = fit_err[0]
            err_beta  = fit_err[1]
        elif fitter == 'third_order':
            err_alpha = fit_err[0]
            err_beta  = fit_err[1]
            err_delta = fit_err[2]
        elif fitter == 'exponential':
            err_epsilon = fit_err[0]
            err_gamma   = fit_err[1]
            err_tau     = fit_err[2]
        elif fitter == 'power':
            err_A = fit_err[0]
            err_B = fit_err[1]
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Get a fit curve!
        if fitter == 'second_order':
            fitted_curve = second_order_func(
                t     = times,
                alpha = optimal_alpha,
                beta  = optimal_beta
            )
            """t_0   = optimal_t_0,
                R_0   = optimal_R_0,"""
        elif fitter == 'third_order':
            fitted_curve = third_order_func(
                t     = times,
                alpha = optimal_alpha,
                beta  = optimal_beta,
                delta = optimal_delta
            )
            """t_0   = optimal_t_0,
                R_0   = optimal_R_0,"""
        elif fitter == 'exponential':
            fitted_curve = exponential_func(
                t       = times,
                epsilon = optimal_epsilon,
                gamma   = optimal_gamma,
                tau     = optimal_tau
            )
            """t_0     = optimal_t_0,
                R_0     = optimal_R_0,"""
        elif fitter == 'power':
            fitted_curve = exponential_func(
                t   = times,
                A   = optimal_A,
                B   = optimal_B,
            )
            """t_0 = optimal_t_0,
                R_0 = optimal_R_0,"""
        else:
            raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
        
        # Return!
        return fitted_curve, optimal_vals, fit_err
    
    # User input formatting.
    if isinstance(filepath, str):
        filepath = [filepath]
    elif isinstance(filepath, (tuple, set)):
        filepath = list(filepath)
    elif isinstance(filepath, dict):
        filepath = list(filepath.keys())
    elif not isinstance(filepath, list):
        # Wrap it.
        filepath = [filepath]
    
    # Create figure for plotting.
    if colourise:
        fig1, ax1 = plt.subplots(figsize=(12.8, 10), facecolor=get_colourise(-2))
        fig2, ax2 = plt.subplots(figsize=(9, 11), facecolor=get_colourise(-2))
    else:
        ##fig1, ax1 = plt.subplots(figsize=(12.8, 10))
        fig1, ax1 = plt.subplots(figsize=(12.8, 9.14))
        fig2, ax2 = plt.subplots(figsize=(9, 11))
    
    # Go through the files and add them to the plot.
    ## Also, store fit values for the return statement.
    fitted_values_to_be_returned = []
    fitted_errors_to_be_returned = []
    lowest_non_short_resistance_of_all = 1000000000
    highest_number_on_the_y_axis = 0.0
    for jj in range(len(filepath)):
        filepath_item = filepath[jj]
        
        # Set initial parameters.
        times = []
        resistances = []
        obvious_short = 160 # [Ω]  --  Define a resistance that defines "a short."
        lowest_non_short_resistance_in_set = 1000000000
        
        ## The new time format assumes that all data is set with reference
        ## to some initial time.
        time_0 = -1 # [s] -- Which would be 23:59:59 on December 31st 1969
        time_at_start = 0.0
        
        # Check whether to begin to store data from this file, i.e., whether
        # the user is requesting to skip the initial resistance drop.
        do_not_save_data_yet = skip_initial_drop
        
        # Open file.
        with open(os.path.abspath(filepath_item), newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            rows = list(reader)  # Convert to list for indexing options
            
            ## In case of old data files, we need to catch whether there is
            ## no tag telling us where the manipulation sequence started.
            ## So, we actually go through the file once first, looking
            ## for tags.
            old_file = True
            for i in range(len(rows)):
                try:
                    if 'START_MANIPULATION' in str(rows[i+1][2]):
                        old_file = False
                except IndexError:
                    # In this case, we didn't find the tag either.
                    pass
            if old_file:
                # Thus, we (TODO: currently) have no reliable way of
                # determining where the initial drop ends, if this happens.
                print("Data file '"+str(filepath_item)+"' could not be used to identify where the initial drop ends. Ignoring argument.")
                do_not_save_data_yet = False
            
            # Old file or not, we take the timestamp in cell [4,1] (0 indx.)
            # in order to calculate how much initial time was removed.
            time_value_at_first_entry = float(rows[4][1])
            time_at_start = -1.0
            
            # Go through the file.
            for i in range(len(rows)):
                if i % 6 == 3:
                    
                    if do_not_save_data_yet:
                        ## This portion will create an IndexError in case
                        ## the datafile is old and does not contain
                        ## any 'START_MANIPULATION' tag.
                        ## This error case was handled above.
                        if 'START_MANIPULATION' in str(rows[i+1][2]):
                            # Then signal that we may commence.
                            do_not_save_data_yet = False
                            
                            # Report how much time passed before this tag
                            # appeared.
                            time_at_start = float(rows[i+1][1])
                            print(filepath_item)
                            print("Detected that the initial drop was "+str(time_at_start - time_value_at_first_entry)+" [s] long.")
                    
                    # The reason this if-if case is written this way,
                    # is to catch the data in the same data storage event
                    # in the file, that also contained the START_* keyword.
                    # Which, could have been the zeroth data storage event.
                    
                    if (not do_not_save_data_yet):
                        ## In that case, continue!
                        
                        # Every sixth row +3 contains a resistance value
                        current_resistance = float(rows[i][1])
                        
                        # Get the SI prefix for this data.
                        ## TODO append more options, like MOhm.
                        if '[kOhm]' in str(rows[i][0]):
                            si_unit_prefix_scaler = 1000
                        else:
                            si_unit_prefix_scaler = 1
                        
                        # Scale to Ohm
                        current_resistance *= si_unit_prefix_scaler
                        
                        # Update the lowest resistance found!
                        if (current_resistance < lowest_non_short_resistance_in_set) and (current_resistance > obvious_short):
                            lowest_non_short_resistance_in_set = current_resistance
                            if lowest_non_short_resistance_in_set < lowest_non_short_resistance_of_all:
                                lowest_non_short_resistance_of_all = lowest_non_short_resistance_in_set
                        
                        # Plot junction? (i.e., plot broken junctions?)
                        if current_resistance > plot_no_junction_resistance_under_ohm:
                            
                            # Super, append junction and continue.
                            resistances.append(current_resistance)
                            
                            # Every sixth row +4 contains a time value
                            time_value = float(rows[i+1][1])
                            ## Did we define the starting time?
                            if time_0 == -1:
                                time_0 = time_value
                            
                            # Calculate what number to be put as the time_value.
                            ## UNIX time or seconds relative to start?
                            if normalise_time:
                                time_value -= time_0
                            times.append(time_value)
                            
                            ## The new format assumes that any and all times
                            ## report the UNIX timestamp of the data itself.
                            ## This way, there is less b/s here regarding
                            ## relative offsets and hatmatilka.
        
        # Ensure lists are the same length
        min_length = min(len(times), len(resistances))
        times = times[:min_length]
        resistances = resistances[:min_length]
        
        # At this point we may just as well numpy-ify the lists.
        times = np.array(times, dtype=np.float64)
        resistances = np.array(resistances, dtype=np.float64)
        
        # Report detected R(t=0).
        print("R₀ of trace: "+str(resistances[0]))
        
        # Normalise resistance axis?
        if normalise_resistances == 1:
            resistances = ((resistances / resistances[0]) - 1) * 100
            y_label_text = "Resistance increase [%]"
        else:
            y_label_text = "Resistance [Ω]"
        
        # At this point, we can identify how much initial drop was removed,
        # vs. the maximum resistance change.
        if normalise_resistances == 1:
            print("Drop removed: "+str(time_at_start - time_value_at_first_entry)+", max change: "+str(resistances[-1]))
        
        # Add item to plot!
        ## First, let's try to fit the data too.
        fit_results = None
        if (fitter != 'none'):
            # The user has requested a fit.
            fit_results = active_increase_fitter(
                resistances = resistances, ## Note that this axis has been normalised by now, if so was requested by the user.
                time = times,
                fitter = fitter
            )
        
        ## Get the file label name.
        file_label = str(os.path.splitext(os.path.basename(filepath_item))[0])
        
        # Determine color and marker for trace?
        ## Select marker symbol. Also, this marker symbol is used later, too.
        if   (jj % 5) == 0:
            marker_symbol = 'o'
        elif (jj % 5) == 1:
            marker_symbol = 's'
        elif (jj % 5) == 2:
            marker_symbol = '^'
        elif (jj % 5) == 3:
            marker_symbol = '*'
        elif (jj % 5) == 4:
            marker_symbol = 'D'
        if (not colourise):
            
            # Just plot from a map.
            if 'thin' in file_label.lower():
                colour_label = 'custom_thin'
                colours = ['#1cee70', '#1cd89c', '#1cc1c3', '#1ca5da', '#1c88e8', '#1c70ee']
                
                num_items_to_colour = len(filepath)
                if num_items_to_colour < 6:
                    colors = colours[:num_items_to_colour]
                elif num_items_to_colour == 6:
                    colors = colours
                else:
                    raise NotImplementedError("Colouring error.")
                
                # Get the last number from the filename
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None) + ' mV'
            
            elif 'thick' in file_label.lower():
                colour_label = 'custom_thick'
                colours = ['#ee1c1c', '#f47e1c', '#f6c11c', '#e6df1c', '#d1eb1c', '#c4ee1c']
                
                num_items_to_colour = len(filepath)
                if num_items_to_colour < 6:
                    colors = colours[:num_items_to_colour]
                elif num_items_to_colour == 6:
                    colors = colours
                else:
                    raise NotImplementedError("Colouring error.")
                
                # Get the last number from the filename
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None) + ' mV'
            
            elif 'thinLEGACY' in file_label.lower():
                ## Catch legacy thin.
                colour_label = 'winter'
                num_items_to_colour = len(filepath)
                cmap = plt.cm.get_cmap(colour_label, num_items_to_colour)
                colours = [cmap(i) for i in range(num_items_to_colour)]
                
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
            
            elif 'thickLEGACY' in file_label.lower():
                ## Catch legacy thick.
                colour_label = 'autumn'
                num_items_to_colour = len(filepath)
                cmap = plt.cm.get_cmap(colour_label, num_items_to_colour)
                colours = [cmap(i) for i in range(num_items_to_colour)]
                
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
            
            else:
                # Fallback to default colormap if not 'thin' or 'thick'
                colour_label = 'tab20'
                num_items_to_colour = len(filepath)
                cmap = plt.cm.get_cmap(colour_label, num_items_to_colour)
                colours = [cmap(i) for i in range(num_items_to_colour)]
            
            # Mask the scatter plot?
            if enable_mask:
                mask = (times >= 0) & (times <= 300)
                times_to_plot = np.array(times)[mask]
                resistances_to_plot = np.array(resistances)[mask]
            else:
                times_to_plot = times.copy()
                resistances_to_plot = resistances.copy()
            
            plt.figure(1) # Set figure 1 as active.
            plt.scatter(times_to_plot, resistances_to_plot, marker=marker_symbol, label=file_label, color=colours[jj])## TODO!! colours[jj])## TODO!! colours(jj))
            
            # Update the largest value present on the y-axis?
            if np.max(resistances) > highest_number_on_the_y_axis:
                highest_number_on_the_y_axis = np.max(resistances)
            
            ##plt.figure(3) # Set figure 3 as active.
            ##plt.scatter(times_to_plot, np.log10(resistances_to_plot), marker=marker_symbol, label=file_label, color=colours[jj])## TODO!! colours(jj))
        else:
            
            # Get pseudolegacy filename labels?
            if   'thin'  in file_label.lower():
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
                
            elif 'thick' in file_label.lower():
                # Get the last number.
                match = re.search(r'(\d+)$', file_label)
                file_label = str(int(match.group(1)) if match else None)+' mV'
            
            # Mask the scatter plot?
            if enable_mask:
                mask = (times >= 0) & (times <= 300)
                times_to_plot = np.array(times)[mask]
                resistances_to_plot = np.array(resistances)[mask]
            else:
                times_to_plot = times.copy()
                resistances_to_plot = resistances.copy()
        
            # Then follow the schema.
            plt.figure(1) # Set figure 1 as active.
            plt.scatter(times_to_plot, resistances_to_plot, marker=marker_symbol, label=file_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
            
            ##plt.figure(3) # Set figure 3 as active.
            ##plt.plot(times_to_plot, np.log10(resistances_to_plot), marker=marker_symbol, label=file_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
            
            # Update the largest value present on the y-axis?
            if np.max(resistances) > highest_number_on_the_y_axis:
                highest_number_on_the_y_axis = np.max(resistances)
        
        # Plot the fit curve.
        fit_label = ''
        
        ## At this point, store the fit values and errors, for the return
        ## statement that happens later.
        if fitter != 'none':
            fitted_values_to_be_returned.append(fit_results[1])
            fitted_errors_to_be_returned.append(fit_results[2])
        
            for kk in range(len(fit_results[1])):
                fitted_values = (fit_results[1])[kk]
                fitted_errors = (fit_results[2])[kk]
                prefix = '?'
                
                # See comment above regarding the error bars being larger
                # than the fitted values if R_0 and t_0 are included.
                """if   kk == 0:
                    prefix = 'R₀'
                    fit_label += prefix+': '+(f"{fitted_values:.3f} ±{fitted_errors:.3f}")+'\n'
                elif kk == 1:
                    prefix = 't₀'
                    fit_label += prefix+': '+(f"{fitted_values:.3f} ±{fitted_errors:.3f}")+'\n'
                else:"""
                if fitter == 'second_order':
                    """if   kk == 2:"""
                    if   kk == 0:
                        prefix = 'α'
                        """elif kk == 3:"""
                    elif kk == 1:
                        prefix = 'β'
                elif fitter == 'third_order':
                    """if   kk == 2:"""
                    if   kk == 0:
                        prefix = 'α'
                        """elif kk == 3:"""
                    elif kk == 1:
                        prefix = 'β'
                        """elif kk == 4:"""
                    elif kk == 2:
                        prefix = 'δ'
                elif fitter == 'exponential':
                    """if   kk == 2:"""
                    if   kk == 0:
                        prefix = 'ε'
                        """elif kk == 3:"""
                    elif kk == 1:
                        prefix = 'V₀'
                        """elif kk == 4:"""
                    elif kk == 2:
                        prefix = 'τ'
                elif fitter == 'power':
                    """if   kk == 2:"""
                    if   kk == 0:
                        prefix = 'A'
                        """elif kk == 3:"""
                    elif kk == 1:
                        prefix = 'B'
                else:
                    raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
                
                # Find a proper exponent of the number.
                exponent       = np.floor(np.log10(np.abs( fitted_values )))
                error_exponent = np.floor(np.log10(np.abs( fitted_errors )))
                fit_label += prefix+': '+(f"{(fitted_values * (10**(-exponent))):.3f}·10^{exponent} ±{(fitted_errors * (10**(-error_exponent))):.3f}·10^{error_exponent}")+'\n'
            
            # Mask the fit pot?
            if enable_mask:
                mask = (times >= 0) & (times <= 300)
                times_to_plot = np.array(times)[mask]
                fit_to_plot   = np.array(fit_results[0])[mask]
            else:
                times_to_plot = times
                fit_to_plot   = fit_results[0]
            
            if (not colourise):
                plt.figure(1) # Set figure 1 as active.
                if plot_fit_parameters_in_legend:
                    plt.plot(times_to_plot, fit_to_plot, linestyle='--', label='Fit '+str(jj)+': '+fit_label, color=colours[jj])## TODO!! colours(jj))
                else:
                    plt.plot(times_to_plot, fit_to_plot, linestyle='--', color=colours[jj])## TODO!! colours(jj))
            else:
                plt.figure(1) # Set figure 1 as active.
                if plot_fit_parameters_in_legend:
                    plt.plot(times_to_plot, fit_to_plot, linestyle='--', label='Fit '+str(jj)+': '+fit_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
                else:
                    plt.plot(times_to_plot, fit_to_plot, linestyle='--', color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
        else:
            # Fitter == 'none'.
            fitted_values_to_be_returned.append(None)
            fitted_errors_to_be_returned.append(None)
        
        # Get residuals plot?
        if fitter != 'none':
            ## fit_results[0]: the fit curve.
            ##    resistances: the actual numbers measured.
            ##       residual: actual_y - predicted_y
            residuals = resistances - fit_results[0]
            
            MSE  = np.mean(residuals**2)
            RMSE = np.sqrt(MSE)
            print("RMSE is: "+str(RMSE))
            
            if (not colourise):
                plt.figure(2) # Set figure 2 as active.
                plt.scatter(times, residuals, marker=marker_symbol, label='Residuals, '+file_label, color=colours[jj])## TODO!! colours(jj))
            else:
                plt.figure(2) # Set figure 2 as active.
                plt.scatter(times, residuals, marker=marker_symbol, label='Residuals, '+file_label, color=get_colourise((jj // 4) + ((jj % 4) + 1) / 10))
    
    # Set axes' colour? Title colour? And so on.
    for mm in range(2):
        plt.figure(mm+1)
        plt.grid()
    
    if colourise:
        fig1.patch.set_alpha(0)
        fig2.patch.set_alpha(0)
        
        ax1.grid(color=get_colourise(-1))
        ax1.set_facecolor(get_colourise(-2))
        ax1.spines['bottom'].set_color(get_colourise(-1))
        ax1.spines['top'].set_color(get_colourise(-1))
        ax1.spines['left'].set_color(get_colourise(-1))
        ax1.spines['right'].set_color(get_colourise(-1))
        ax1.tick_params(axis='both', colors=get_colourise(-1))
        
        ax2.grid(color=get_colourise(-1))
        ax2.set_facecolor(get_colourise(-2))
        ax2.spines['bottom'].set_color(get_colourise(-1))
        ax2.spines['top'].set_color(get_colourise(-1))
        ax2.spines['left'].set_color(get_colourise(-1))
        ax2.spines['right'].set_color(get_colourise(-1))
        ax2.tick_params(axis='both', colors=get_colourise(-1))
    
    # Bump up the size of the ticks' numbers on the axes.
    ax1.tick_params(axis='both', labelsize=23)
    ax2.tick_params(axis='both', labelsize=23)
    
    # Extend axes to include the origin?
    ## Do not extend the x-axis if trying to plot the UNIX time.
    if (np.all(times > 0)) and (normalise_time):
        ax1.set_xlim(xmin=0)
    #if np.all(lowest_non_short_resistance_of_all >= -5):
    #    ax1.set_ylim(ymin=-5)
    
    # Other figure formatting.
    ## Figure out the label padding.
    if (normalise_resistances == 1):
        ax1.set_xlim(-10,310)
        if highest_number_on_the_y_axis < 8:
            ax1.set_ylim(-1,9)
            label_padding = 68
        elif highest_number_on_the_y_axis < 30:
            ax1.set_ylim(-1,25)
            label_padding = 49
        else:
            ax1.set_ylim(-8,105)
            label_padding = 30
    else:
        label_padding = 30
    
    
    if (not colourise):
        plt.figure(1) # Set figure 1 as active.
        plt.xlabel("Duration [s]", fontsize=33)
        plt.ylabel(y_label_text, fontsize=33, labelpad=label_padding)
        
        # Set title for this figure.
        plt.title(title_label, fontsize=38)
        
        plt.figure(2) # Set figure 2 as active.
        plt.xlabel("Duration [s]", fontsize=33)
        plt.ylabel(y_label_text, fontsize=33)
        if fitter != 'none':
            if fitter == 'second_order':
                plt.title("Residuals, 2nd-ord-polyn.", fontsize=38)
            elif fitter == 'third_order':
                plt.title("Residuals, 3rd-ord-polyn.", fontsize=38)
            elif fitter == 'exponential':
                plt.title("Residuals, exponential func.", fontsize=38)
            elif fitter == 'power':
                plt.title("Residuals, power-law", fontsize=38)
    else:
        plt.figure(1) # Set figure 1 as active.
        plt.xlabel("Duration [s]", color=get_colourise(-1), fontsize=33)
        plt.ylabel(y_label_text, color=get_colourise(-1), fontsize=33)
        plt.title("Resistance vs. Time", color=get_colourise(-1), fontsize=38)
    
    # Show shits.
    for ll in range(2):
        plt.figure(ll+1)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        if (not plot_fit_parameters_in_legend):
            plt.legend(fontsize=26)
        else:
            plt.legend()
        # Save plot?
        if savepath != '':
            plt.tight_layout()
            fig1.savefig(savepath, dpi=164, bbox_inches='tight')
    plt.show()
    
    # Return stuffs.
    return fitted_values_to_be_returned, fitted_errors_to_be_returned
    
def analyse_fitted_polynomial_factors(
    filepath,
    voltage_list_mV = ['auto'],
    normalise_resistances = 0,
    normalise_time = True,
    plot_no_junction_resistance_under_ohm = 0,
    fitter = 'second_order',
    skip_initial_drop = False,
    plot_fit_parameters_in_legend = False,
    colourise = False,
    savepath = '',
    ):
    ''' For a list of files, given as a list of filepath (strings),
        perform a fit for the whole file, and get the fit values
        back.
        
        Then, plot these fitted values versus a user-supplied voltage list.
        The value 'auto' vill analyse the file name in order to try to find
        XpYY, which defines X.YY volt.
    '''
    
    # Create a list, that will be filled with traces and their properties.
    # This list will be returned as the function ends.
    list_of_traces_in_plot = []
    
    # User input formatting.
    if isinstance(filepath, str):
        filepath = [filepath]
    elif isinstance(filepath, (tuple, set)):
        filepath = list(filepath)
    elif isinstance(filepath, dict):
        filepath = list(filepath.keys())
    elif not isinstance(filepath, list):
        # Wrap it.
        filepath = [filepath]
    
    # Try to make voltage list?
    if voltage_list_mV[0] == 'auto':
        
        # User said yes.
        ## Clear out the list.
        voltage_list_mV = []
        for mm in range(len(filepath)):
            voltage_list_mV.append(0)
        
        # Go through files.
        for ii in range(len(filepath)):
            item = filepath[ii]
            
            # Pattern to match "_XpYY_" format, [V]
            match_volts = re.search(r'(\d+)p(\d+)', item)
            if match_volts:
                volts = float(f"{match_volts.group(1)}.{match_volts.group(2)}")
                voltage_list_mV[ii] = int(volts*1000)
            
            else:
                # Alternative: pattern to match the last number before ".csv"
                # Which, is in units of [mV]
                match_millivolts = re.search(r'(\d+)(?=\.csv$)', item)
                if match_millivolts:
                    voltage_list_mV[ii] = int(match_millivolts.group(1))
                
                else:
                    # At this point, the filepath could not be used to determine the voltage.
                    raise ValueError("Halted! Could not determine voltage used automatically from the file: '"+str(item)+"'")
    
    ## At this point, the voltage list is known.
    
    # Perform fits.
    assert fitter != 'none', "Halted! This function requires a fitter to be active, i.e., fitter != 'none'."
    (fitted_values, fitted_errors) = plot_active_manipulation(
        filepath = filepath,
        normalise_resistances = normalise_resistances,
        normalise_time = normalise_time,
        plot_no_junction_resistance_under_ohm = plot_no_junction_resistance_under_ohm,
        fitter = fitter,
        skip_initial_drop = skip_initial_drop,
        plot_fit_parameters_in_legend = plot_fit_parameters_in_legend,
        colourise = colourise,
        savepath = savepath
    )
    
    ## The data format here is weird:
    ## Each new ROW of fitted_values, contains information about the next
    ## datapoint on the Y axis. Each COLUMN of fitted values, contains
    ## this datapoint for a new TRACE in the plot. And, each value
    ## in fitted_errors, is the error bar.
    
    num_traces = max(len(arr) for arr in fitted_values)  # Max number of parameters
    num_points = len(fitted_values)  # Number of voltage points
    
    # Organise data by parameter index
    y_values = [[] for _ in range(num_traces)]
    y_errors = [[] for _ in range(num_traces)]
    for i in range(num_points):
        for j in range(len(fitted_values[i])):  # Iterate over parameters in each fit
            y_values[j].append(fitted_values[i][j])
            y_errors[j].append(fitted_errors[i][j])
    
    # Prepare labels for the plot.
    if fitter == 'second_order':
        fit_label_list = ['α', 'β']
    elif fitter == 'third_order':
        fit_label_list = ['α', 'β', 'δ']
    elif fitter == 'exponential':
        fit_label_list = ['γ', 'τ']
    elif fitter == 'power':
        fit_label_list = ['A', 'B']
    else:
        raise ValueError("Halted! Unknown value provided for agument 'fitter': "+str(fitter))
    
    ## At this point, voltage_list_mV is the X axis.
    ## Similarly, y_values[i] is the Y axis.
    
    # If fitter is either 'second_order' or 'third order', then the first
    # parameter reveals the dependency on applied voltage.
    # Let's try to fit this dependency.    
    polynomial_fit_successful = False
    if (fitter == 'second_order') or (fitter == 'third_order'):
        polynomial_fit_successful = True
        try:
            ''' Plotting it as a log-lin diagram, reveals a straight line for the
            alpha fit parameter, so we have reason to suspect that the first-order
            resistance dependency to the applied voltage is exponential.
            // Christian 2025-03-09'''
            
            def exponential_func_for_alpha(V_mV, alpha_0, V_0_mV):
                return alpha_0 * ((np.e)**(V_mV / V_0_mV))
                ##return alpha_0 * ((np.e)**(v_mV * gamma))  ## Works well, but gamma is not super-intuitive.
                #return alpha_0 * ((np.e)**((v_mV - v_mott_mV) * gamma))
            
            # Grab the α values and β values.
            alpha_values = y_values[0]
            beta_values  = y_values[1]
            
            # Here, sort the alpha_values versus the applied voltages.
            ##if voltage_list_mV == sorted(voltage_list_mV):
            ##    # The list is ordered, do nothing.
            ##    pass
            ##elif
            if voltage_list_mV == sorted(voltage_list_mV, reverse = True):
                # The list is reversed, sort it.
                ## IMPORTANT: remember that the beta_values
                ##            also must be reversed here.
                voltage_list_mV.reverse()
                alpha_values.reverse()
                beta_values.reverse()
            else:
                # The list is a mess.
                sorted_triples = sorted(zip(voltage_list_mV, alpha_values, beta_values))
                voltage_list_mV, alpha_values, beta_values = zip(*sorted_triples)
                voltage_list_mV = list(voltage_list_mV)
                alpha_values = list(alpha_values)
                beta_values = list(beta_values)
            
            ## I don't really know how to make a good guess for the scalar alpha_0.
            alpha_0_guess = 1.0
            
            ## For the gamma guess, take the slope of the ln curve.
            gamma_guess_vector_y = np.log(alpha_values)
            gamma_guess_alpha_fit = (gamma_guess_vector_y[-1] - gamma_guess_vector_y[0])/(voltage_list_mV[-1] - voltage_list_mV[0])
            
            # When instead making a characteristic voltage guess,
            # this would be 1/gamma.
            V_0_guess_vector_mV = 1/gamma_guess_alpha_fit
            
            ## For the V_mott guess, it's about 0.5 V for aliminium.
            v_mott_guess_mV = 500 # mV
            
            # Fit!
            optimal_vals_alpha_fit, covariance_mtx_of_opt_vals_alpha_fit = curve_fit(
                f     = exponential_func_for_alpha,
                xdata = voltage_list_mV,
                ydata = alpha_values,
                p0    = (alpha_0_guess, V_0_guess_vector_mV) ## gamma_guess_alpha_fit)#, v_mott_guess_mV)
            )
            # Get fit errors.
            fit_err_alphas = np.sqrt(np.diag(covariance_mtx_of_opt_vals_alpha_fit))
            # Get fit curve for later.
            fit_curve_x_mV = np.linspace(0, np.max(voltage_list_mV), 100)
            fitted_curve_alphas = exponential_func_for_alpha(
                V_mV = fit_curve_x_mV,
                alpha_0 = optimal_vals_alpha_fit[0],
                V_0_mV = optimal_vals_alpha_fit[1],
                ##gamma = optimal_vals_alpha_fit[1], ## Works well, but gamma is not very intuitive.
                ##v_mott_mV = optimal_vals_alpha_fit[2]
            )
        except RuntimeError:
            # Signal failed fit.
            polynomial_fit_successful = False
    
    # Plot each parameter trace.
    ## Create figure for plotting. Note that this is the plot with
    ## α₀ and V₀ fitted in it, that is, not the α(V) fit.
    if colourise:
        fig1, ax1 = plt.subplots(figsize=(12.8, 12.8), facecolor=get_colourise(-2))
    else:
        fig1, ax1 = plt.subplots(figsize=(12.8, 12.8))
    
    for i in range(num_traces):
        if y_values[i]:
            if fit_label_list[i] == 'α':
                label_string = f'{fit_label_list[i]} [s⁻¹]'
            elif fit_label_list[i] == 'β':
                label_string = f'{fit_label_list[i]} [s⁻²]'
            else:
                label_string = f'Parameter {fit_label_list[i]}'
            if colourise:
                plt.errorbar(voltage_list_mV, y_values[i], yerr=y_errors[i], marker='o', linestyle='-', capsize=3, label=label_string, color=get_colourise(i))
            else:
                plt.errorbar(voltage_list_mV, y_values[i], yerr=y_errors[i], marker='o', linestyle='-', capsize=3, label=label_string)
            
            # Append to list of traces to be returned.
            list_of_traces_in_plot += [[voltage_list_mV, y_values[i], label_string, y_errors[i]]]
    
    # Plot fit of fit?
    ## polynomial_fit_successful will be False if fitter
    ## was not set to 'second_order' or 'third_order'.
    if polynomial_fit_successful:
        fit_of_fit_label = 'f(V) = α₀·e^(V/V₀)\n'
        for item in range(len(optimal_vals_alpha_fit)):
            # Find a proper exponent of the number.
            exponent       = np.floor(np.log10(np.abs( optimal_vals_alpha_fit[item] )))
            error_exponent = np.floor(np.log10(np.abs( fit_err_alphas[item] )))
            if item == 0:
                prefix = 'α₀'
            else:
                prefix = 'V₀'
            fit_of_fit_label += prefix+': '+(f"{(optimal_vals_alpha_fit[item] * (10**(-exponent))):.3f}·10^{exponent} ±{(fit_err_alphas[item] * (10**(-error_exponent))):.3f}·10^{error_exponent}")
            if item != (len(optimal_vals_alpha_fit)-1):
                fit_of_fit_label += '\n'
        plt.plot(fit_curve_x_mV, fitted_curve_alphas, label = fit_of_fit_label)
        
        # Append to list that will be returned.
        list_of_traces_in_plot += [[fit_curve_x_mV, fitted_curve_alphas, fit_of_fit_label, None]]
    
    if colourise:
        plt.xlabel("Voltage [mV]", fontsize=33, color=get_colourise(-1))
        plt.ylabel("Fit parameters", fontsize=33, color=get_colourise(-1))
        plt.title("Fit parameter trends vs. voltage", fontsize=38, color=get_colourise(-1))
    else:
        plt.xlabel("Voltage [mV]", fontsize=33)
        plt.ylabel("Fit parameters", fontsize=33)
        plt.title("Fit parameter trends vs. voltage", fontsize=38)
    
    # Colourise axes, set axis limits, and such?
    plt.grid()
    ax1.set_xlim(xmin=0.0, xmax=1100)   # Include the zero for the voltage.
    ax1.set_ylim(ymax=0.6, ymin=-0.050)
    if colourise:
        fig1.patch.set_alpha(0)
        
        ax1.grid(color=get_colourise(-1))
        ax1.set_facecolor(get_colourise(-2))
        ax1.spines['bottom'].set_color(get_colourise(-1))
        ax1.spines['top'].set_color(get_colourise(-1))
        ax1.spines['left'].set_color(get_colourise(-1))
        ax1.spines['right'].set_color(get_colourise(-1))
        ax1.tick_params(axis='both', colors=get_colourise(-1))
    
    # Show and save shits.
    plt.legend(fontsize=26)
    plt.show()
    
    # Return a whole bunch of stuff, so that this function
    # can be used in the meta-function that calls it several times.
    return list_of_traces_in_plot

def analyse_multiple_sets_of_fitted_polynomial_factors(
    list_of_filepath_lists,
    voltage_list_mV = ['auto'],
    normalise_resistances = 0,
    normalise_time = True,
    plot_no_junction_resistance_under_ohm = 0,
    fitter = 'second_order',
    skip_initial_drop = False,
    plot_fit_parameters_in_legend = False,
    set_labels = [],
    colourise = False,
    savepath = '',
    ):
    ''' Analyse multiple sets of active resistance manipulation data.
        
        The list_of_filepath_lists argument should be a list, containing lists
        of filepaths. These filepaths correspond to resistance-vs-time
        manipulations done on Josephson junctions.
    '''
    
    # User argument sanitation.
    if fitter.lower() == 'none':
        raise ValueError("Halted! This function requires a fitter to be active; you provided 'none' as an argument here.")

    # Did the user label the sets?
    if (len(set_labels) != 0) and (not (len(list_of_filepath_lists) == len(set_labels))):
        raise ValueError("Halted! Unable to determine set labels, ensure that you did provide a set label for each filepath in your arguments.") 
    
    def insert_counter_before_extension(filepath, counter, default_ext=".png"):
        directory, filename = os.path.split(filepath)
        name, ext = os.path.splitext(filename)

        if not ext:
            ext = default_ext

        new_filename = f"{name}{counter}{ext}"
        return os.path.join(directory, new_filename)
    
    # Collect all of the data.
    results_of_sets = []
    repeat_counter = 1
    for i in range(len(list_of_filepath_lists)):
        current_set = list_of_filepath_lists[i]
        
        # Fit!
        results_of_sets.append(
            analyse_fitted_polynomial_factors(
                filepath = current_set,
                voltage_list_mV = voltage_list_mV,
                normalise_resistances = normalise_resistances,
                normalise_time = normalise_time,
                plot_no_junction_resistance_under_ohm = plot_no_junction_resistance_under_ohm,
                fitter = fitter,
                skip_initial_drop = skip_initial_drop,
                plot_fit_parameters_in_legend = plot_fit_parameters_in_legend,
                colourise = colourise,
                savepath = insert_counter_before_extension(savepath, i),
            )
        )
    
    # Plot the parameters in separate plots.
    ## How many parameters do we expect to see?
    if   fitter == 'second_order':
        expected_number_of_fit_parameters = 2
    elif fitter == 'third_order':
        expected_number_of_fit_parameters = 3
    elif fitter == 'exponential':
        expected_number_of_fit_parameters = 2
    elif fitter == 'power':
        expected_number_of_fit_parameters = 2
    
    # Make plots.
    ## Create figures for plotting. First, clear all of the previous junk.
    plt.close('all')
    plt.ioff()
    if colourise:
        fig1, ax1 = plt.subplots(figsize=(15, 11), facecolor=get_colourise(-2))
        fig2, ax2 = plt.subplots(figsize=(15, 11), facecolor=get_colourise(-2))
        if expected_number_of_fit_parameters == 3:
            fig3, ax3 = plt.subplots(figsize=(15, 11), facecolor=get_colourise(-2))
    else:
        #fig1, ax1 = plt.subplots(figsize=(14.6, 9))
        fig1, ax1 = plt.subplots(figsize=(11.605, 9))
        #fig2, ax2 = plt.subplots(figsize=(12.5, 9))
        fig2, ax2 = plt.subplots(figsize=(12.60, 9))
        if expected_number_of_fit_parameters == 3:
            fig3, ax3 = plt.subplots(figsize=(15, 11))
    
    # Figure out colours.
    if not colourise:
        num_items_to_colour = len(list_of_filepath_lists)
        if num_items_to_colour != 5:
            colour_label = 'tab20'
            colours = plt.cm.get_cmap(colour_label, num_items_to_colour)
        else:
            colours = ["#EE1C1C", "#C4EE1C", "#1CEE70", "#1C70EE", "#C41CEE"]
    
    # Loop through the data and create plotty things.
    axes = [ax1, ax2]
    figs = [fig1, fig2]

    for curr_figure in range(expected_number_of_fit_parameters):
        ax = axes[curr_figure]
        fig = figs[curr_figure]

        # Go through the data.
        for sets in range(len(list_of_filepath_lists)):
            # Get the fitted parameter's voltages, data, label, and error bars.
            voltage_list_mV = results_of_sets[sets][curr_figure][0]
            parameter_data = results_of_sets[sets][curr_figure][1]

            if len(set_labels) > 0:
                label_string = str(set_labels[sets])
            else:
                label_string = str(sets + 1)

            errors = results_of_sets[sets][curr_figure][3]

            # Choose linestyle
            linestyle = '' if curr_figure == 0 else '-'

            # Plot data points
            if colourise:
                ax.errorbar(
                    voltage_list_mV, parameter_data, yerr=errors,
                    marker='o', linestyle=linestyle, capsize=3,
                    label=label_string, color=get_colourise(sets)
                )
            else:
                if num_items_to_colour != 5:
                    ax.errorbar(
                        voltage_list_mV, parameter_data, yerr=errors,
                        marker='o', linestyle=linestyle, capsize=3,
                        label=label_string, color=colours(sets)
                    )
                else:
                    ax.errorbar(
                        voltage_list_mV, parameter_data, yerr=errors,
                        marker='o', linestyle=linestyle, capsize=3,
                        label=label_string, color=colours[sets]
                    )

            # Plot fit trace in the alpha plot (first parameter only)
            if curr_figure == 0:
                try:
                    fit_x_axis = results_of_sets[sets][curr_figure + expected_number_of_fit_parameters][0]
                    fit_y_axis = results_of_sets[sets][curr_figure + expected_number_of_fit_parameters][1]
                    if plot_fit_parameters_in_legend:
                        fit_label = results_of_sets[sets][curr_figure + expected_number_of_fit_parameters][2]
                    else:
                        fit_label = None

                    if colourise:
                        ax.plot(fit_x_axis, fit_y_axis, label=fit_label, color=get_colourise(sets + 0.1))
                    else:
                        if num_items_to_colour != 5:
                            ax.plot(fit_x_axis, fit_y_axis, label=fit_label, color=colours(sets))
                        else:
                            ax.plot(fit_x_axis, fit_y_axis, label=fit_label, color=colours[sets])
                except IndexError:
                    print("No fit data found in set " + str(sets) + ".")
    
    # Include the origin and such.
    ax1.set_xlim(xmin=0.0, xmax=1100)
    ax2.set_xlim(xmin=0.0, xmax=1100)
    
    # Apply axis labels, titles, grid, and legend using explicit axes
    axes = [ax1, ax2]
    figs = [fig1, fig2]

    for ll in range(expected_number_of_fit_parameters):
        ax = axes[ll]
        fig = figs[ll]

        # Grid
        ax.grid(True)

        # Title colour
        title_colour = "#000000" if not colourise else get_colourise(-1)

        # Axis labels
        ax.set_xlabel("Voltage [mV]", fontsize=33)
        if fitter == 'second_order':
            if ll == 0:
                ax.set_ylabel("α [s⁻¹]", fontsize=33)
            else:
                ax.set_ylabel("β [s⁻²]", fontsize=33)
        else:
            print("WARNING: Unable to select appropriate Y-axis labels at this time.")
            ax.set_ylabel("Parameter value", fontsize=33)

        # Tick sizes
        ax.tick_params(axis='both', labelsize=26)

        # Titles
        if fitter != 'none':
            if savepath == '':
                if fitter == 'second_order':
                    title_text = "Fit parameters, 2nd order polynomial"
                elif fitter == 'third_order':
                    title_text = "Fit parameters, 3rd order polynomial"
                elif fitter == 'exponential':
                    title_text = "Fit parameters, exponential function"
                elif fitter == 'power':
                    title_text = "Fit parameters, power-law"
                else:
                    title_text = "Fit parameters"
                ax.set_title(title_text, color=title_colour, fontsize=38)
            else:
                ax.set_title("", color=title_colour, fontsize=38)
        else:
            raise ValueError(f"Error! Could not understand argument provided to 'fitter': {fitter}")

        # Legend
        if (ll == 1) and (fitter == 'second_order'):
            ax.legend(fontsize=26, loc='lower left')
        else:
            ax.legend(fontsize=26)
    
    # Grid colourisation?
    if colourise:
        fig1.patch.set_alpha(0)
        ax1.grid(color=get_colourise(-1))
        ax1.set_facecolor(get_colourise(-2))
        ax1.spines['bottom'].set_color(get_colourise(-1))
        ax1.spines['top'].set_color(get_colourise(-1))
        ax1.spines['left'].set_color(get_colourise(-1))
        ax1.spines['right'].set_color(get_colourise(-1))
        ax1.tick_params(axis='both', colors=get_colourise(-1))
    
        fig2.patch.set_alpha(0)
        ax2.grid(color=get_colourise(-1))
        ax2.set_facecolor(get_colourise(-2))
        ax2.spines['bottom'].set_color(get_colourise(-1))
        ax2.spines['top'].set_color(get_colourise(-1))
        ax2.spines['left'].set_color(get_colourise(-1))
        ax2.spines['right'].set_color(get_colourise(-1))
        ax2.tick_params(axis='both', colors=get_colourise(-1))
    
    # Save plots?
    """if savepath != '':
        plt.tight_layout()"""
    for fig in figs:
        fig.tight_layout()
        fig1.savefig(insert_counter_before_extension(savepath, 10000), dpi=164, bbox_inches='tight')
        fig2.savefig(insert_counter_before_extension(savepath, 20000), dpi=164, bbox_inches='tight')
    
    # Show shits.        
    plt.show()
    
    return results_of_sets

def calculate_delta_f01(
    initial_resistance,
    final_resistance,
    E_C_in_Hz,
    Delta_cold_eV,
    difference_between_RT_and_cold_resistance = 1.1385,
    T = 0.010,
    verbose = True
    ):
    ''' Return the difference in frequency that the qubit has shifted,
        given the before and after resistances of the junction.
    '''
    
    # Get final frequency.
    final_frequency = calculate_f01_from_RT_resistance(
        room_temperature_resistance = final_resistance,
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = verbose
    )
    
    # Get initial frequency.
    initial_frequency = calculate_f01_from_RT_resistance(
        room_temperature_resistance = initial_resistance,
        E_C_in_Hz = E_C_in_Hz,
        Delta_cold_eV = Delta_cold_eV,
        difference_between_RT_and_cold_resistance = difference_between_RT_and_cold_resistance,
        T = T,
        verbose = verbose
    )
    
    # Return difference.
    return (final_frequency - initial_frequency)

def acquire_relaxation_data_from_folder(
    folder_path,
    take_relaxation_data_at_this_time_s,
    filename_tags = [],
    verbose = True,
    ):
    ''' From a supplied folder path, acquire the resistance relaxation of the
        files held within.
    '''
    
    # Make lists to store the active and active+passive resistance increase.
    active_gain_percent  = []
    total_gain_percent   = []
    
    # User input sanitisation.
    if not os.path.isdir(folder_path):
        raise ValueError("Halted! Invalid path: "+str(folder_path))
    
    # Process all files yo.
    for filename in os.listdir(folder_path):
        # Ensure that all filenames are present.
        if all(keyword in filename for keyword in filename_tags):
            full_path = os.path.join(folder_path, filename)
            # Catch whether there are nasty subfolders.
            if os.path.isfile(full_path):
                
                ## TODO: Below, there is no catch to see whether
                ##       the user simply inserted a very very long time
                ##       for relaxation, and then simply let the sample run
                ##       for this very very long time. A "STOP_CREEP" is
                ##       inserted by the measurement apparatur.
                ##       Then, typically there is a new resistance
                ##       manipulation that triggers. So the file gets
                ##       'corrupted' into containing two manipulations.
                ##       In one case, such a file was discovered.
                
                # We only want .csv files.
                if filename.endswith(".csv"):
                    ## Set some process flags.
                    reference_resistance = 0.0
                    resistance_at_manipulation_finished = 0.0
                    relaxation_began_at_time = -1.0
                    relaxation_analysed_at_this_time = -1.0
                    resistance_at_relaxation_point = 0.0
                    shorted = False
                    
                    # Open file, do the thing.                    
                    with open(os.path.abspath(full_path), newline='', encoding='utf-8') as csvfile:
                        reader = csv.reader(csvfile, delimiter=';')
                        rows = list(reader)  # Convert to list for indexing options
                        
                        # Go through the file.
                        for i in range(len(rows)):
                            # Every sixth row +3 contains a resistance value.
                            if (i % 6 == 3) and (not shorted):
                                
                                # We will constantly be on the lookout for
                                # broken junctions in the measurement.
                                try:
                                    if 'SHORTED' in rows[i+1][2]:
                                        # Broken junction. If we got our value,
                                        # so be it. Otherwise, just abort anyhow.
                                        shorted = True
                                except:
                                    # Then there is no such cell.
                                    pass
                                
                                # Grab some resistance as the starting value
                                # for the resistance manipulation?
                                if reference_resistance == 0.0:
                                    try:
                                        # Was this the start of the
                                        # resistance manipulation?
                                        if 'START_MANIPULATION' in rows[i+1][2]:
                                            # Oh shit, it was. Grab resistance.
                                            reference_resistance = float(rows[i][1])
                                            
                                            # Get the SI prefix for this data.
                                            ## TODO append more options, like MOhm.
                                            if '[kOhm]' in str(rows[i][0]):
                                                si_unit_prefix_scaler = 1000
                                            else:
                                                si_unit_prefix_scaler = 1
                                            
                                            # Scale to Ohm
                                            reference_resistance *= si_unit_prefix_scaler
                                            
                                    except IndexError:
                                        # In this case, there is simply nothing
                                        # written in such a cell.
                                        pass
                                
                                else:
                                    ## At this point, there is a reference
                                    ## resistance that we can work from.
                                    ## We want to know whether the upcoming
                                    ## resistance value is the one that the
                                    ## user wants.
                                    
                                    # Look for the tag signalling
                                    # that the manipulation is done.
                                    ## There are situations where
                                    ## this tag never appears.
                                    ## If that happens, simply assume
                                    ## that the measurement before
                                    ## the START_CREEP tag is the end
                                    ## of the manipulation.
                                    if 'STOP_MANIPULATION' in rows[i+1][2]:
                                        # Manipulation finished!
                                        # Grab the current resistance.
                                        resistance_at_manipulation_finished = float(rows[i][1])
                                        
                                        # Get the SI prefix for this data.
                                        ## TODO append more options, like MOhm.
                                        if '[kOhm]' in str(rows[i][0]):
                                            si_unit_prefix_scaler = 1000
                                        else:
                                            si_unit_prefix_scaler = 1
                                        
                                        # Scale to Ohm
                                        resistance_at_manipulation_finished *= si_unit_prefix_scaler
                                        
                                        # Grab the time at which this happened.
                                        # Every sixth row +4 contains a time value
                                        relaxation_began_at_time = float(rows[i+1][1])
                                    
                                    elif 'START_CREEP' in rows[i+1][2]:
                                        # Check whether this is a merged file,
                                        # that is, the resistance manipulation
                                        # simply stopped, and the operator
                                        # stopped the manipulation portion.
                                        if resistance_at_manipulation_finished == 0:
                                            # This happened, meaning that
                                            # the manipulation didn't reach
                                            # its target. Select the previous
                                            # resistance as the 'end datapoint'
                                            resistance_at_manipulation_finished = float(rows[i-6][1])
                                        
                                            # Get the SI prefix for this data.
                                            ## TODO append more options, like MOhm.
                                            if '[kOhm]' in str(rows[i-6][0]):
                                                si_unit_prefix_scaler = 1000
                                            else:
                                                si_unit_prefix_scaler = 1
                                            
                                            # Scale to Ohm
                                            resistance_at_manipulation_finished *= si_unit_prefix_scaler
                                            
                                            # Grab the time at which the relaxation started.
                                            try:
                                                relaxation_began_at_time = float(rows[i+1-6][1])
                                            except IndexError:
                                                relaxation_began_at_time = float(rows[i+1][1])
                                        
                                        # In either case, we continue
                                        # by looking for the user-set
                                        # time at which the resistance
                                        # of interest is located.
                                        relaxation_analysed_at_this_time = \
                                            relaxation_began_at_time + \
                                            take_relaxation_data_at_this_time_s
                                    
                                    # Now, we are merely waiting for the
                                    # resistance point taken at the time that
                                    # the user is interested in.
                                    if relaxation_analysed_at_this_time != -1:
                                        
                                        # Then, let's look for times.
                                        current_time = float(rows[i+1][1])
                                        if (current_time >= relaxation_analysed_at_this_time) and (resistance_at_relaxation_point == 0.0):
                                            
                                            ## Here, resistance_at_relaxation_point == 0.0 ensures that
                                            ## we only perform this part if the sample
                                            ## has not had its relaxation point resistance identified.
                                            
                                            # Fantastic, this resistance should probably be our final datapoint.
                                            # Let's just check whether the latest datapoint were
                                            # closer in time to that data, first.
                                            previous_time = float(rows[i+1-6][1])
                                            diff_current  = relaxation_began_at_time - current_time
                                            diff_previous = relaxation_began_at_time - previous_time
                                            if diff_current < diff_previous:
                                                # The datapoint that passed the
                                                # timestamp where the user
                                                # would have wanted the data,
                                                # is closer.
                                                ## VERIFY that the sample
                                                ## didn't die at precisely
                                                ## this point in time!
                                                try:
                                                    if 'SHORTED' in rows[i+1][2]:
                                                        # Junction dead at the
                                                        # finish line.
                                                        shorted = True
                                                except:
                                                    # Then there is no such cell.
                                                    pass
                                                
                                                if not shorted:
                                                    # Success!
                                                    resistance_at_relaxation_point = float(rows[i][1])
                                            else:
                                                # The previous datapoint was
                                                # closer in time to the user-
                                                # requested time.
                                                ## Here, we do not have to
                                                ## verify whether this
                                                ## resistance is taken at a
                                                ## point that is a short.
                                                ## Since, this fact would
                                                ## have been discovered by now.
                                                resistance_at_relaxation_point = float(rows[i-6][1])
                                            
                                            # Double-check whether we shorted
                                            # at the finish line.
                                            if not shorted:
                                                # Get the SI prefix for this data.
                                                ## TODO append more options, like MOhm.
                                                if '[kOhm]' in str(rows[i-6][0]):
                                                    si_unit_prefix_scaler = 1000
                                                else:
                                                    si_unit_prefix_scaler = 1
                                                
                                                # Scale to Ohm
                                                resistance_at_relaxation_point *= si_unit_prefix_scaler
                    
                    # At this point, we are done with the file.
                    # Collect values.
                    if reference_resistance != 0.0:
                        if resistance_at_manipulation_finished != 0.0:
                            if resistance_at_relaxation_point != 0.0:
                                if verbose:
                                    print("Ref res: "+str(reference_resistance)+", Res at manip finished: "+str(resistance_at_manipulation_finished)+", Res total: "+str(resistance_at_relaxation_point))
                                active_gain_percent.append((resistance_at_manipulation_finished / reference_resistance -1)*100)
                                total_gain_percent.append((resistance_at_relaxation_point / reference_resistance -1)*100)
                                
                                ## Removing this for now; there is no guarantee that the resistance hasn't
                                ## crept down below zero.
                                ##assert (resistance_at_relaxation_point / reference_resistance -1) > 0, "ERROR: "+str(filename)
                                
                                if (shorted) and (verbose):
                                    print(">> Shorted during relaxation, but after the sought-for data was found: "+str(filename))
                            else:
                                if verbose:
                                    print(">> Failed during relaxation: "+str(filename))
                        else:
                            if verbose:
                                print(">> Failed during manipulation: "+str(filename))
                    else:
                        if verbose:
                            print(">> Manipulation failed to start: "+str(filename))
                else:
                    print(">> Can't read file '"+str(filename)+"'")
            else:
                print(">> Can't process '"+str(filename)+"'")
    
    # Return things.
    if len(active_gain_percent) != len(total_gain_percent):
        raise RuntimeError("Error! The number of entries for the active manipulations does not match the number of successful manipulations. This is a bug. No. manipulations was: "+str(active_gain_percent)+", No. successes was: "+str(total_gain_percent))
    return (active_gain_percent, total_gain_percent)
    

def plot_trend_active_vs_total_resistance_gain(
    title_voltage_V,
    title_junction_size_nm,
    folder_path,
    take_relaxation_data_at_this_time_interval_s,
    filename_tags = [],
    outlier_threshold_in_std_devs = 2.0,
    colourise = False,
    savepath = '',
    plot = True,
    enforce_n = -1,
    plot_RMS_deviation = False,
    reproduce_paper_200_low = False,
    ):
    ''' For an interval of times to be observed, get the slope and offset
        data for the linear fits of the active-vs-total-manipulation
        experiments.
        
        enforce_n:  Set the minimum number of samples taken at some datapoint
                    to quality as fittable.
        
        plot_RMS_deviation: during experimental development, it is useful to
                            know the fit deviation. This argument shows this
                            deviation.
        
        reproduce_paper_200_low:    a shortcut argument. If set to true,
                                    then a 'cached'/'finished' set of numpy
                                    arrays will be used, instead of reloading
                                    all of the 200 nm low-dose device data.
                                    This data corresponds to the plot showing
                                    the active vs total resistance gain fit
                                    over time, as shown in the 2026 paper by
                                    C. Križan et all. See Zenodo for raw data.
                                    This argument saves you about
                                    85 minutes of computer runtime, if you
                                    simply want to see the plot from the paper.
    '''
    list_of_slopes              = []
    list_of_slopes_err          = []
    list_of_offsets             = []
    list_of_offsets_err         = []
    list_of_n_samples           = []
    list_of_rms_deviations      = []
    list_of_rms_standard_errors = []
    latest_progress = 0.0
    last_progress_printed = -100.0
    
    # Keep last 10 timing intervals between progress prints
    progress_timestamps = deque(maxlen=15)
    
    # Use cached values for the 2026 low-dose device by C.Križan et al.?
    if not reproduce_paper_200_low:
        for ii in range(len(take_relaxation_data_at_this_time_interval_s)):
            # Get progress.
            latest_progress = int(np.round((100*(ii/len(take_relaxation_data_at_this_time_interval_s)))))
            if not (latest_progress == last_progress_printed):
                ## #print(f"Progress: {100*(ii/len(take_relaxation_data_at_this_time_interval_s)):.1f} % done.")
                ## print("Progress: "+str(latest_progress)+" % done.")
                ## last_progress_printed = latest_progress
                
                now = time_module.time()
                
                # Calculate estimated remaining time if we have enough data
                eta_str = ""
                if last_progress_printed != -100.0:
                    progress_timestamps.append(now)
                    if len(progress_timestamps) >= 2:
                        # Compute average seconds per progress step
                        intervals = [
                            progress_timestamps[i] - progress_timestamps[i-1]
                            for i in range(1, len(progress_timestamps))
                        ]
                        avg_interval = sum(intervals) / len(intervals)
                        
                        remaining_steps = 100 - latest_progress
                        remaining_seconds = int(avg_interval * remaining_steps)
                        
                        # Convert to h, m, s
                        h, rem = divmod(remaining_seconds, 3600)
                        m, s = divmod(rem, 60)
                        eta_str = f", {h}h {m}' {s}\" remaining."
                
                print(f"{latest_progress} %{eta_str}")
                last_progress_printed = latest_progress
            
            # Get time!
            curr_investigated_time = take_relaxation_data_at_this_time_interval_s[ii]
            
            # Get data!
            curr_k, curr_m, curr_k_err, curr_m_err, n_samples, rms_deviation, rms_se = plot_active_vs_total_resistance_gain(
                title_voltage_V = title_voltage_V,
                title_junction_size_nm = title_junction_size_nm,
                folder_path = folder_path,
                take_relaxation_data_at_this_time_s = curr_investigated_time,
                filename_tags = filename_tags,
                outlier_threshold_in_std_devs = outlier_threshold_in_std_devs,
                highlight_outliers = False,
                plot_ideal_curve = False,
                colourise = colourise,
                savepath = '',
                plot = False,
            )
            
            # Append to lists!
            list_of_slopes.append(curr_k)
            list_of_offsets.append(curr_m)
            list_of_slopes_err.append(curr_k_err)
            list_of_offsets_err.append(curr_m_err)
            list_of_n_samples.append(n_samples)
            list_of_rms_deviations.append(rms_deviation)
            list_of_rms_standard_errors.append(rms_se)
    else:
        print("NOTE: THE ARGUMENT reproduce_paper_200_low IS SET TO TRUE - USING CALCULATED/CACHED VALUES MATCHING THE 200 low-dose DEVICE IN THE 2026 PAPER BY C.KRIŽAN ET AL.") # 2025-12-28
        
        list_of_slopes = [np.float64(1.0083145019665645), np.float64(1.018504610444493), np.float64(1.032310754019432), np.float64(1.0426702731835238), np.float64(1.0496541722695292), np.float64(1.0579980831758922), np.float64(1.0662054585739842), np.float64(1.0704139510098172), np.float64(1.0771908492539408), np.float64(1.0800818158253722), np.float64(1.0865114441588775), np.float64(1.0901246016746138), np.float64(1.0926033095379728), np.float64(1.0967132827047432), np.float64(1.0992111888812608), np.float64(1.1028614910843897), np.float64(1.1039818350935453), np.float64(1.1106531742222083), np.float64(1.110760431493953), np.float64(1.1139881841638362), np.float64(1.118169621278399), np.float64(1.1180400790276306), np.float64(1.1222396569695872), np.float64(1.121704801573073), np.float64(1.120724032825599), np.float64(1.1256495296789384), np.float64(1.1243210262550107), np.float64(1.1260168813308715), np.float64(1.1290460791351717), np.float64(1.1308754314911391), np.float64(1.13047928622566), np.float64(1.1322568516942353), np.float64(1.1340658369071215), np.float64(1.133884804952088), np.float64(1.1367634851138881), np.float64(1.1376153774012108), np.float64(1.1405218173774414), np.float64(1.1384590404884853), np.float64(1.1409076349993073), np.float64(1.1419921383097063), np.float64(1.142732674188685), np.float64(1.1452213516623473), np.float64(1.1467952712522047), np.float64(1.1463039599985505), np.float64(1.1464524812325154), np.float64(1.147234355006555), np.float64(1.1477264244101377), np.float64(1.149939990118938), np.float64(1.1506399404559529), np.float64(1.1495696176482193), np.float64(1.149030703519166), np.float64(1.151146879227402), np.float64(1.1520294306017649), np.float64(1.1523939168156885), np.float64(1.1545453361960845), np.float64(1.1536948625667574), np.float64(1.1565014311568655), np.float64(1.1534684840809561), np.float64(1.1567997514843902), np.float64(1.1601799936700257), np.float64(1.1599333601857904), np.float64(1.1623365534670984), np.float64(1.1625705220572615), np.float64(1.1634868445249464), np.float64(1.1663472644944968), np.float64(1.1651967955369444), np.float64(1.1660505481454593), np.float64(1.1673947471794939), np.float64(1.1657685582516257), np.float64(1.1664888898804069), np.float64(1.1672464618723346), np.float64(1.1689754210505732), np.float64(1.167429738385836), np.float64(1.1683531125459987), np.float64(1.1692446949978297), np.float64(1.1699353804928603), np.float64(1.1679577734660365), np.float64(1.171830556782237), np.float64(1.1708686922709788), np.float64(1.169392217726654), np.float64(1.1748551064259445), np.float64(1.1732271486510493), np.float64(1.1737665665723707), np.float64(1.1732639101266706), np.float64(1.1746268989328637), np.float64(1.1764574573500959), np.float64(1.1753197066713104), np.float64(1.1769498821608346), np.float64(1.1767540898485704), np.float64(1.1790255850943183), np.float64(1.1755726956790795), np.float64(1.1752816877283874), np.float64(1.1770276888251028), np.float64(1.176797057113224), np.float64(1.1766354348063481), np.float64(1.1798936423565691), np.float64(1.1796979591077845), np.float64(1.1791128072412167), np.float64(1.1792338726942013), np.float64(1.1815445455197942), np.float64(1.1788028424252335), np.float64(1.1814866082395918), np.float64(1.1792956358221505), np.float64(1.1824975831259261), np.float64(1.1826506853362195), np.float64(1.1814338292625761), np.float64(1.184939876013666), np.float64(1.1834306164312514), np.float64(1.181626914579461), np.float64(1.184273053406331), np.float64(1.1843809414657314), np.float64(1.1835612040129628), np.float64(1.1836261615170602), np.float64(1.1816745914967617), np.float64(1.1829897489651342), np.float64(1.1817805921073425), np.float64(1.1826836461558965), np.float64(1.1867423113530722), np.float64(1.183838036744731), np.float64(1.1846434589489843), np.float64(1.1840123728517817), np.float64(1.188008017015301), np.float64(1.1847458817621337), np.float64(1.1838151045108067), np.float64(1.184664848158491), np.float64(1.184889802843933), np.float64(1.1855514243989873), np.float64(1.1854718565274325), np.float64(1.1860291525822027), np.float64(1.1872172045307121), np.float64(1.1861419295062334), np.float64(1.1875004932514377), np.float64(1.1897370853523956), np.float64(1.1901680795712344), np.float64(1.1904035822089205), np.float64(1.1894431941696677), np.float64(1.1887150974155885), np.float64(1.1875048896549023), np.float64(1.1921471064930937), np.float64(1.1891338270665084), np.float64(1.190475432310319), np.float64(1.1874389626019048), np.float64(1.1878422630730598), np.float64(1.1895154574692923), np.float64(1.187860588202764), np.float64(1.1867392724723183), np.float64(1.1911485872632288), np.float64(1.1902818293182171), np.float64(1.190400438336175), np.float64(1.18796319803227), np.float64(1.1872510962789455), np.float64(1.1904559077640795), np.float64(1.1877177595638), np.float64(1.1890894505904008), np.float64(1.190332698169083), np.float64(1.1916788188271532), np.float64(1.1899193590360027), np.float64(1.1900798010251215), np.float64(1.1930252926065508), np.float64(1.1936912840082838), np.float64(1.191923369499633), np.float64(1.1944424852690367), np.float64(1.1939565683387081), np.float64(1.1921418224991445), np.float64(1.1916947957090767), np.float64(1.191964547805021), np.float64(1.1943241766788304), np.float64(1.19186003512542), np.float64(1.1915960154991327), np.float64(1.1918588208199108), np.float64(1.1911430712517275), np.float64(1.193120311118272), np.float64(1.1935001810050225), np.float64(1.194178100821411), np.float64(1.194459447375542), np.float64(1.1978087748333155), np.float64(1.195128266862652), np.float64(1.1957962404776548), np.float64(1.1967598605314733), np.float64(1.1970212083642195), np.float64(1.1990966315571034), np.float64(1.1996470508889163), np.float64(1.1979843506913928), np.float64(1.1968953320574334), np.float64(1.1990688817590354), np.float64(1.201524465976986), np.float64(1.1980832112399085), np.float64(1.1991097691631956), np.float64(1.200399395138325), np.float64(1.2011794113176437), np.float64(1.1999493231689922), np.float64(1.2009515507372106), np.float64(1.2010035709827684), np.float64(1.201374575186365), np.float64(1.1989816103322628), np.float64(1.198310260758706), np.float64(1.2014510165578316), np.float64(1.1998063245027242), np.float64(1.2009350117526911), np.float64(1.2021102017960763), np.float64(1.2021682869264967), np.float64(1.200566431991772), np.float64(1.2041482336856606), np.float64(1.2021260498743365), np.float64(1.2041987257173772), np.float64(1.2050822783438062), np.float64(1.20162249285605), np.float64(1.2046428107902019), np.float64(1.2028226814598635), np.float64(1.1993991681070602), np.float64(1.19723332302755), np.float64(1.2008756009420012), np.float64(1.199688710714154), np.float64(1.1992865058144817), np.float64(1.199579294062468), np.float64(1.1983665061268107), np.float64(1.1998558301946085), np.float64(1.1977849193797414), np.float64(1.2006326229280961), np.float64(1.1978311267287995), np.float64(1.2003711828083883), np.float64(1.2015331244180272), np.float64(1.196171762991452), np.float64(1.199145061942525), np.float64(1.1999860375788352), np.float64(1.2008456052989431), np.float64(1.202089357992279), np.float64(1.2015769624449668), np.float64(1.2013690993898483), np.float64(1.2020588386388238), np.float64(1.199736588203388), np.float64(1.2007607822080055), np.float64(1.1998020602206065), np.float64(1.201807102390488), np.float64(1.2026467555318032), np.float64(1.2038426385429801), np.float64(1.1984223021168403), np.float64(1.2003603449216445), np.float64(1.2012211267503434), np.float64(1.2019464398452095), np.float64(1.2023721482390555), np.float64(1.2023650009770752), np.float64(1.2014839217753719), np.float64(1.2022718990309686), np.float64(1.2001616383457305), np.float64(1.201041941897938), np.float64(1.2026114197776088), np.float64(1.200380282202539), np.float64(1.2013164953023943), np.float64(1.2015449430143603), np.float64(1.2042705115912224), np.float64(1.2022618539180185), np.float64(1.2024214571647063), np.float64(1.201881497527577), np.float64(1.2048695055330305), np.float64(1.204564591093936), np.float64(1.2028739574226652), np.float64(1.2047449471774698), np.float64(1.206283857612625), np.float64(1.2039064618906132), np.float64(1.2033841986227451), np.float64(1.2025961447549536), np.float64(1.2012143480618904), np.float64(1.203279164990808), np.float64(1.201551032030173), np.float64(1.2039766252683073), np.float64(1.2038519912927623), np.float64(1.205587434582935), np.float64(1.203777711180329), np.float64(1.204664437680951), np.float64(1.203207029218441), np.float64(1.2049213985862757), np.float64(1.2044054624623846), np.float64(1.2052068037386996), np.float64(1.2056218316766154), np.float64(1.2052700393912614), np.float64(1.2029328205344267), np.float64(1.202739785670959), np.float64(1.2054401159359125), np.float64(1.2055870449093187), np.float64(1.2072406543968113), np.float64(1.2057444261395325), np.float64(1.2040091485343576), np.float64(1.203591995493069), np.float64(1.2063980686432592), np.float64(1.2054618288798709), np.float64(1.2082297317837158), np.float64(1.2063238100924871), np.float64(1.205570738439649), np.float64(1.2058327347640068), np.float64(1.2046450737527632), np.float64(1.206210252400665), np.float64(1.205931851333767), np.float64(1.2100642434768998), np.float64(1.2074586180466669), np.float64(1.2066332632481522), np.float64(1.2060234961805856), np.float64(1.207107234886917), np.float64(1.2066965294407643), np.float64(1.2066396648082702), np.float64(1.2086798156208927), np.float64(1.2035566538180287), np.float64(1.2081884515491246), np.float64(1.2064243897036553), np.float64(1.2088891690856245), np.float64(1.2071565860799478), np.float64(1.2072284050183135), np.float64(1.206933669398424), np.float64(1.2053329699319695), np.float64(1.2088467856659295), np.float64(1.208056778438441), np.float64(1.2120393713861894), np.float64(1.2078876767829785), np.float64(1.2065284249287813), np.float64(1.208421119685373), np.float64(1.2095135414081608), np.float64(1.2067163653281252), np.float64(1.2082776907859487), np.float64(1.2080252890032115), np.float64(1.2079343225221195), np.float64(1.207836800039754), np.float64(1.2094069345809366), np.float64(1.2064124951839408), np.float64(1.2072104366701055), np.float64(1.2083196860801326), np.float64(1.2095454934672443), np.float64(1.2084963742409578), np.float64(1.210018202341004), np.float64(1.211574833501421), np.float64(1.2117115214080274), np.float64(1.2094019452116087), np.float64(1.2120745382447717), np.float64(1.2106927566541437), np.float64(1.2106302196936791), np.float64(1.2107239902462266), np.float64(1.208977530130831), np.float64(1.2102431729402887), np.float64(1.211952190537952), np.float64(1.2097538180418095), np.float64(1.2114840239879723), np.float64(1.2128910711832388), np.float64(1.2130452217443837), np.float64(1.2102128459689745), np.float64(1.2111652121819179), np.float64(1.2128574143858726), np.float64(1.212802989494654), np.float64(1.2152789775953687), np.float64(1.2121450026399634), np.float64(1.2132007833841445), np.float64(1.2105658514821465), np.float64(1.2126392227869132), np.float64(1.210226013043036), np.float64(1.2138110240063678), np.float64(1.2120497205173548), np.float64(1.2127952103193196), np.float64(1.2142529235519475), np.float64(1.215412691804955), np.float64(1.2126782871167687), np.float64(1.216293745027816), np.float64(1.2114262198014358), np.float64(1.2112567414752906), np.float64(1.2141499466734307), np.float64(1.2125374054823974), np.float64(1.2128853814337024), np.float64(1.2111979561268305), np.float64(1.2109196393012165), np.float64(1.2136733404221924), np.float64(1.2117893991553357), np.float64(1.215676210926588), np.float64(1.2129796233144938), np.float64(1.2136006188231674), np.float64(1.2128922545147265), np.float64(1.212850082515545), np.float64(1.214359791731056), np.float64(1.2132935094930006), np.float64(1.214858801397903), np.float64(1.2123010701150585), np.float64(1.2142995245022787), np.float64(1.21414864211363), np.float64(1.2157750434505243), np.float64(1.2146305491923983), np.float64(1.2135710317765214), np.float64(1.214056428121569), np.float64(1.2107206294600457), np.float64(1.2135228454278357), np.float64(1.2126088020611414), np.float64(1.2109329503992006), np.float64(1.2120523736829587), np.float64(1.2139028816346074), np.float64(1.2118887003707361), np.float64(1.2144990535084639), np.float64(1.213114310692445), np.float64(1.211504364483386), np.float64(1.2132615534408357), np.float64(1.2152134712796605), np.float64(1.210672067840577), np.float64(1.214412061609592), np.float64(1.2130982174166443), np.float64(1.2129916247788552), np.float64(1.2127841635765195), np.float64(1.2127562754263639), np.float64(1.212272290031006), np.float64(1.210030240464154), np.float64(1.2133985175070519), np.float64(1.213004684120588), np.float64(1.2134521949319097), np.float64(1.2162920537967157), np.float64(1.214030892508854), np.float64(1.2156572529633987), np.float64(1.214494909350495), np.float64(1.2157658123774566), np.float64(1.2153789261032597), np.float64(1.2150125837804246), np.float64(1.2138023017440671), np.float64(1.212395685606131), np.float64(1.2153688492232804), np.float64(1.2143970693159492), np.float64(1.2152582161054015), np.float64(1.2139545796846523), np.float64(1.2134906139552133), np.float64(1.214053256514907), np.float64(1.2168874939023782), np.float64(1.2131492565477506), np.float64(1.2175491300318668), np.float64(1.214414309025008), np.float64(1.2137407932934539), np.float64(1.2138371866642355), np.float64(1.2140033050204206), np.float64(1.2123368144014348), np.float64(1.21617655402699), np.float64(1.2177443074258423), np.float64(1.2184744705136517), np.float64(1.216816154591641), np.float64(1.215219137847481), np.float64(1.2179880772192875), np.float64(1.2136990904505653), np.float64(1.2157588128287893), np.float64(1.2142186332726845), np.float64(1.2141067780776864), np.float64(1.2148121317170857), np.float64(1.2161950818345133), np.float64(1.2138489293573291), np.float64(1.214838167398497), np.float64(1.217557765600698), np.float64(1.2186799896498934), np.float64(1.2160601526823063), np.float64(1.2196965815346639), np.float64(1.2173579633988523), np.float64(1.2187139269118872), np.float64(1.216218787452153), np.float64(1.2170300326267662), np.float64(1.2157960011350621), np.float64(1.2191249429011448), np.float64(1.2171314204535648), np.float64(1.2187166906136708), np.float64(1.2197581801283386), np.float64(1.2138711788511334), np.float64(1.2171653337749142), np.float64(1.2166007909206529), np.float64(1.2172525933904526), np.float64(1.2197404505802836), np.float64(1.2174435356993658), np.float64(1.219752077847734), np.float64(1.2155710737347036), np.float64(1.2168139058206502), np.float64(1.2182560135185805), np.float64(1.2188087839411033), np.float64(1.221067249566452), np.float64(1.221130971688252), np.float64(1.2202085967924245), np.float64(1.2190007205201454), np.float64(1.2196732773874666), np.float64(1.221030066253081), np.float64(1.2192543925692463), np.float64(1.2199198093556243), np.float64(1.220768648491369), np.float64(1.2195575093703857), np.float64(1.2212646778215726), np.float64(1.219469814006687), np.float64(1.2181733849699041), np.float64(1.219855954814718), np.float64(1.2206202024239978), np.float64(1.2212508839663272), np.float64(1.2194372012119974), np.float64(1.217512665807888), np.float64(1.2211321395241472), np.float64(1.219358311747335), np.float64(1.222259443046989), np.float64(1.2209916204246374), np.float64(1.221963628090141), np.float64(1.223118247597414), np.float64(1.2188001962577737), np.float64(1.2203546249463653), np.float64(1.2208775823721567), np.float64(1.2229982366901333), np.float64(1.223648870966905), np.float64(1.2229253035493408), np.float64(1.2200625603381334), np.float64(1.2226851287222837), np.float64(1.2228949862613872), np.float64(1.2231887131159043), np.float64(1.2204062254569144), np.float64(1.2199498571565435), np.float64(1.2219536074039548), np.float64(1.2203276928600115), np.float64(1.220520138646251), np.float64(1.2225106091078872), np.float64(1.2219772402715943), np.float64(1.2232933405900457), np.float64(1.2197952404560217), np.float64(1.220294258167572), np.float64(1.2220578940478726), np.float64(1.2229572027334352), np.float64(1.2210535973011174), np.float64(1.2229695805305392), np.float64(1.2216293397714304), np.float64(1.2184671549266148), np.float64(1.2202128535563574), np.float64(1.2211818695814958), np.float64(1.2216369390745712), np.float64(1.2190140441091488), np.float64(1.221183610235024), np.float64(1.2230471078715706), np.float64(1.2194781575617692), np.float64(1.2223416682492538), np.float64(1.2208609160080106), np.float64(1.2180639622064566), np.float64(1.2180866809732818), np.float64(1.2187438314851686), np.float64(1.220190190457532), np.float64(1.2196256179873817), np.float64(1.2215863474282573), np.float64(1.2212237989709076), np.float64(1.2231670781870652), np.float64(1.2214199488527346), np.float64(1.2217901253297125), np.float64(1.2210163004928765), np.float64(1.2213282498309763), np.float64(1.2211573794612354), np.float64(1.2215969653462122), np.float64(1.2232179749979017), np.float64(1.2227727128788755), np.float64(1.2209570914844352), np.float64(1.2197098296394806), np.float64(1.2210875133365064), np.float64(1.2216396227760014), np.float64(1.220444501565288), np.float64(1.2208362265038961), np.float64(1.22255337027239), np.float64(1.2183154452321183), np.float64(1.2189001731058429), np.float64(1.218989537333756), np.float64(1.2225980409951844), np.float64(1.2226833310857288), np.float64(1.2231316164006827), np.float64(1.2234400044416966), np.float64(1.2214832161948808), np.float64(1.223491843125433), np.float64(1.2224293520711649), np.float64(1.222507255783875), np.float64(1.2258452495861558), np.float64(1.2237699972155285), np.float64(1.2229046646277446), np.float64(1.222939886016079), np.float64(1.222435850790174), np.float64(1.221589441265563), np.float64(1.2216830426277083), np.float64(1.2228756919476351), np.float64(1.2216543818533268), np.float64(1.223752263002991), np.float64(1.2253178123709543), np.float64(1.2221502795956196), np.float64(1.2210643376005323), np.float64(1.225184351627304), np.float64(1.2232869136924283), np.float64(1.2238321863575985), np.float64(1.223182577935021), np.float64(1.2234107500381686), np.float64(1.2239165640310787), np.float64(1.2221267982421704), np.float64(1.2248828335293642), np.float64(1.2244900777889527), np.float64(1.223264123450028), np.float64(1.2239353818983634), np.float64(1.222293768069134), np.float64(1.2256685018002242), np.float64(1.2244279505252742), np.float64(1.2240571395284887), np.float64(1.2243931544882407), np.float64(1.2260012253093644), np.float64(1.2250552736058076), np.float64(1.2263281582811942), np.float64(1.2244389761644099), np.float64(1.2250489503714126), np.float64(1.2282016375201565), np.float64(1.2244574398338255), np.float64(1.2253442042031195), np.float64(1.2262156705645095), np.float64(1.2283827295613143), np.float64(1.2219968079357495), np.float64(1.2255842824188727), np.float64(1.2273578304659771), np.float64(1.2247046809054454), np.float64(1.2247801249385704), np.float64(1.2255979301402795), np.float64(1.2254786102290185), np.float64(1.227228107431678), np.float64(1.2237991847107004), np.float64(1.225768115495755), np.float64(1.2279461159234049), np.float64(1.2278343450255524), np.float64(1.2251354325987796), np.float64(1.2245124932159495), np.float64(1.2279998406575892), np.float64(1.225680720972529), np.float64(1.2245679640378069), np.float64(1.2283648501201871), np.float64(1.2287680324416304), np.float64(1.2273904446133306), np.float64(1.227205844431266), np.float64(1.228752020446907), np.float64(1.2299412605172277), np.float64(1.2264591242612517), np.float64(1.2285557948232295), np.float64(1.22799929171996), np.float64(1.2291039229733107), np.float64(1.2286174519560398), np.float64(1.2270019278706368), np.float64(1.2275107738560562), np.float64(1.2284102063163664), np.float64(1.2276208538177276), np.float64(1.2270009740522612), np.float64(1.2265045398717156), np.float64(1.2275425517762006), np.float64(1.2275060496911105), np.float64(1.2263935805180357), np.float64(1.2295966800225073), np.float64(1.226615370571852), np.float64(1.2283791177170709), np.float64(1.228074619595258), np.float64(1.2296172006376551), np.float64(1.2285772687633325), np.float64(1.2304323535800303), np.float64(1.2262789314465035), np.float64(1.2267988975955066), np.float64(1.2284892876896583), np.float64(1.227450603445059), np.float64(1.2282487948086545), np.float64(1.2290252122293792), np.float64(1.2308043070159198), np.float64(1.2300115295726801), np.float64(1.2309169748508457), np.float64(1.2298031383306194), np.float64(1.2292300963644565), np.float64(1.2300827616765302), np.float64(1.229894705525094), np.float64(1.2280230090761566), np.float64(1.2278150892384705), np.float64(1.229580695132711), np.float64(1.2267685024642718), np.float64(1.228450594683733), np.float64(1.2274524572048302), np.float64(1.2274853001438515), np.float64(1.2292020787891447), np.float64(1.2272077629942866), np.float64(1.2277517446502675), np.float64(1.2289686367688175), np.float64(1.2304784834674256), np.float64(1.2250965921446224), np.float64(1.2308482849302265), np.float64(1.228087905921302), np.float64(1.2280283508983734), np.float64(1.2285116284697084), np.float64(1.227569676969377), np.float64(1.2303579541577512), np.float64(1.2305530165911809), np.float64(1.2302945446083577), np.float64(1.2292230753236186), np.float64(1.2305944741321673), np.float64(1.2290696494248963), np.float64(1.227846810927196), np.float64(1.2287874056385928), np.float64(1.2298656328871698), np.float64(1.2302863009040852), np.float64(1.2292900983374249), np.float64(1.2310422897392588), np.float64(1.2290489548626184), np.float64(1.2297696308973995), np.float64(1.2273524930034674), np.float64(1.22712906130471), np.float64(1.226795392239661), np.float64(1.2272131767030043), np.float64(1.2311831907175703), np.float64(1.229385144935647), np.float64(1.2277646253206396), np.float64(1.227948664850614), np.float64(1.2307268737511503), np.float64(1.229541888655152), np.float64(1.2309087779613348), np.float64(1.230464347657033), np.float64(1.227724642529088), np.float64(1.2291627739533182), np.float64(1.2280519833489327), np.float64(1.2296374293354109), np.float64(1.229628887647799), np.float64(1.2293207841095422), np.float64(1.2331662737424458), np.float64(1.231793120978308), np.float64(1.2337768711302564), np.float64(1.2311231364557158), np.float64(1.23071490017016), np.float64(1.2333278647543107), np.float64(1.2301347065197257), np.float64(1.2313526029446247), np.float64(1.2303437375313626), np.float64(1.228808886616204), np.float64(1.2273224571341685), np.float64(1.228321460724857), np.float64(1.228642856615239), np.float64(1.230771260594265), np.float64(1.231766938518215), np.float64(1.2321789997072432), np.float64(1.2326563377601134), np.float64(1.2304611336520743), np.float64(1.230716991944624), np.float64(1.2316673911363871), np.float64(1.2287825095152853), np.float64(1.2284287204403859), np.float64(1.227420228785025), np.float64(1.2299637851583567), np.float64(1.2276471637625432), np.float64(1.2286884785621641), np.float64(1.2305631452945303), np.float64(1.2318140546468566), np.float64(1.2299263243819478), np.float64(1.2332278864794863), np.float64(1.2319157026790999), np.float64(1.2314382633167271), np.float64(1.2275967105215948), np.float64(1.227634791287678), np.float64(1.2292116859996918), np.float64(1.2257128236353423), np.float64(1.230645496027712), np.float64(1.2299503372515863), np.float64(1.2298258685211205), np.float64(1.2308637971107477), np.float64(1.2315656680738276), np.float64(1.2310676808389354), np.float64(1.230423684172191), np.float64(1.2310521279904716), np.float64(1.2307407394229148), np.float64(1.2302262338443126), np.float64(1.2342085855482807), np.float64(1.2308907020855575), np.float64(1.23127953039615), np.float64(1.2299064958378894), np.float64(1.2322243342756503), np.float64(1.2316041546211873), np.float64(1.2315245595614834), np.float64(1.2347034640505337), np.float64(1.23041021094031), np.float64(1.2309827689400596), np.float64(1.2328847623185233), np.float64(1.2343301362712529), np.float64(1.2334438541016035), np.float64(1.2329243233750107), np.float64(1.2307461300323628), np.float64(1.2350072634776181), np.float64(1.2333520277717505), np.float64(1.2323656915776584), np.float64(1.2337399840223382), np.float64(1.2322828338843492), np.float64(1.2306846133392937), np.float64(1.2310780032006474), np.float64(1.2350206807066204), np.float64(1.2360471041382652), np.float64(1.2352866930229123), np.float64(1.2326579154438788), np.float64(1.2346875065497918), np.float64(1.2329925922474743), np.float64(1.2324309033694676), np.float64(1.233790266714199), np.float64(1.233760231987932), np.float64(1.232632677322706), np.float64(1.2324545600726253), np.float64(1.2337467616928282), np.float64(1.2344822417171675), np.float64(1.232780931247305), np.float64(1.231761321757811), np.float64(1.232066573289568), np.float64(1.2324703732284248), np.float64(1.2342099173403558), np.float64(1.2313781064805016), np.float64(1.2310552289188503), np.float64(1.2334414318050744), np.float64(1.234770752554404), np.float64(1.2316098995201048), np.float64(1.2334587537535102), np.float64(1.2319800807045418), np.float64(1.2330067518763699), np.float64(1.232643989196572), np.float64(1.2303897113596465), np.float64(1.2290729070914843), np.float64(1.2308857082985163), np.float64(1.2300422984474773), np.float64(1.2293075942985916), np.float64(1.2320768892787903), np.float64(1.2307984112262984), np.float64(1.2293661828200777), np.float64(1.231636672343899), np.float64(1.2291731111459063), np.float64(1.229918162833321), np.float64(1.2295341828376671), np.float64(1.231814672208903), np.float64(1.2311200759667225), np.float64(1.2332084584952723), np.float64(1.2307353776411782), np.float64(1.2301255372046136), np.float64(1.232799883575061), np.float64(1.2324082220058212), np.float64(1.231631313835522), np.float64(1.2315455581836257), np.float64(1.2332190963564091), np.float64(1.2325467166827602), np.float64(1.2330281261849039), np.float64(1.23383860951125), np.float64(1.2332856815044129), np.float64(1.2340216398177632), np.float64(1.2333162460317015), np.float64(1.2326779457887886), np.float64(1.2336913410221892), np.float64(1.2324788576924703), np.float64(1.2324105963066259), np.float64(1.2340841138682184), np.float64(1.2319178645351951), np.float64(1.2309834349297475), np.float64(1.2300595066473565), np.float64(1.2335407204075735), np.float64(1.2339914004348655), np.float64(1.2294835241476196), np.float64(1.2326709353933034), np.float64(1.2320284682559912), np.float64(1.230641771236248), np.float64(1.2311575293361852), np.float64(1.2312018248182284), np.float64(1.2335653531068185), np.float64(1.2328334197265647), np.float64(1.2335504684241256), np.float64(1.2321371224318778), np.float64(1.2319511167704502), np.float64(1.2342303748565815), np.float64(1.233890647588552), np.float64(1.2330195005284086), np.float64(1.2318390665227912), np.float64(1.2339587914229513), np.float64(1.2329572128311352), np.float64(1.233707830474606), np.float64(1.2342084283525387), np.float64(1.2357394877393215), np.float64(1.2329654859375194), np.float64(1.2325156835914362), np.float64(1.2335111172310842), np.float64(1.2368257165728727), np.float64(1.2359341314657557), np.float64(1.23471799908837), np.float64(1.235723084865424), np.float64(1.2362781752069807), np.float64(1.234929914900563), np.float64(1.2343414204542045), np.float64(1.2373161818964726), np.float64(1.2369403533461532), np.float64(1.2350465654633398), np.float64(1.2358605763899817), np.float64(1.2355941325598718), np.float64(1.2355796007110103), np.float64(1.2354841064020108), np.float64(1.2358183518195907), np.float64(1.2362830480253761), np.float64(1.238658078314163), np.float64(1.2350957509791085), np.float64(1.2350357686725522), np.float64(1.2364695890284991), np.float64(1.237924991327007), np.float64(1.2342837771490844), np.float64(1.237273536322197), np.float64(1.2368518678493416), np.float64(1.2363475598330271), np.float64(1.2390174703601422), np.float64(1.2403514455915694), np.float64(1.2385767403219028), np.float64(1.2395415976544772), np.float64(1.240726723503914), np.float64(1.2369424278042775), np.float64(1.237173657137054), np.float64(1.2417643946361387), np.float64(1.2413530947318692), np.float64(1.2374962798579117), np.float64(1.2399244467199475), np.float64(1.2387737628453572), np.float64(1.2385903221179455), np.float64(1.2401050599594474), np.float64(1.2418459177878551), np.float64(1.2419972138616022), np.float64(1.2396392672432357), np.float64(1.240477363901112), np.float64(1.2381892427343948), np.float64(1.237738520217243), np.float64(1.23838916032134), np.float64(1.2425793579020215), np.float64(1.2381638809031408), np.float64(1.2421286010944237), np.float64(1.240832717177301), np.float64(1.2407279539106537), np.float64(1.2438191821875186), np.float64(1.240864185748116), np.float64(1.2433216552952109), np.float64(1.2429277931165628), np.float64(1.239587578633799), np.float64(1.2423225654695627), np.float64(1.2392965910418368), np.float64(1.2397789275286937), np.float64(1.2419369803993008), np.float64(1.2410995554313458), np.float64(1.2424853249877021), np.float64(1.2415939959242372), np.float64(1.242721736664817), np.float64(1.2426185835409844), np.float64(1.2419216500921793), np.float64(1.2401583960731293), np.float64(1.2429778782293315), np.float64(1.237031202588887), np.float64(1.2404173229214173), np.float64(1.23807015452205), np.float64(1.240711590419934), np.float64(1.24289087320577), np.float64(1.2413383722246438), np.float64(1.2432611720893407), np.float64(1.239424980476955), np.float64(1.2429293847643952), np.float64(1.2421794474578896), np.float64(1.2392479046746412), np.float64(1.2386808134366474), np.float64(1.2407413148563804), np.float64(1.2422910776604748), np.float64(1.2375056152166928), np.float64(1.2391626677331764), np.float64(1.2422071928476819), np.float64(1.2404978895156527), np.float64(1.2417348801929995), np.float64(1.2429172403032056), np.float64(1.2401826654413004), np.float64(1.2411317630215541), np.float64(1.2426458930063158), np.float64(1.2394003047363313), np.float64(1.241329204652887), np.float64(1.2408720913751994), np.float64(1.2408603564261365), np.float64(1.2381975344405183), np.float64(1.2409422728671664), np.float64(1.2422871254318002), np.float64(1.242632332547096), np.float64(1.2406325329556247), np.float64(1.2390607058949485), np.float64(1.2394661328405359), np.float64(1.2415780599932487), np.float64(1.2389478246281815), np.float64(1.2394194366113709), np.float64(1.2417139858074042), np.float64(1.2402119734353125), np.float64(1.2392046689203833), np.float64(1.2409128415060997), np.float64(1.242060045077224), np.float64(1.2416222650996906), np.float64(1.2392632446454055), np.float64(1.2385387541948385), np.float64(1.2384596857786996), np.float64(1.2402907783856354), np.float64(1.242868041497711), np.float64(1.2399593220195513), np.float64(1.2427713466498433), np.float64(1.2398944553135098), np.float64(1.242789597835229), np.float64(1.2443136537596033), np.float64(1.2413711073912699), np.float64(1.2409366154873842), np.float64(1.238628088062654), np.float64(1.2408626540884014), np.float64(1.2404317731707073), np.float64(1.2415788315059268), np.float64(1.2419003292071173), np.float64(1.2420943828770212), np.float64(1.242434448036377), np.float64(1.2408350188119013), np.float64(1.2421812264857834), np.float64(1.2405247075762986), np.float64(1.2418798413099317), np.float64(1.2405249440587178), np.float64(1.2431667253173588), np.float64(1.238908270058661), np.float64(1.241540060677482), np.float64(1.2425170416924658), np.float64(1.241294118613778), np.float64(1.2418698229694611), np.float64(1.241812384199625), np.float64(1.2428944272999989), np.float64(1.2410602318744215), np.float64(1.2431850829415665), np.float64(1.2417659252628266), np.float64(1.2394029918738492), np.float64(1.2409664458624654), np.float64(1.2437712917424317), np.float64(1.2418634814147942), np.float64(1.240557877925295), np.float64(1.2425056474508633), np.float64(1.242645732765505), np.float64(1.2413269587207145), np.float64(1.241823553648547), np.float64(1.2417141838515937), np.float64(1.2420905473518111), np.float64(1.2414287161523867), np.float64(1.2415690078087218), np.float64(1.2441108421245914), np.float64(1.2410349008589843), np.float64(1.2422597148873757), np.float64(1.2421141890433893), np.float64(1.2403460177802095), np.float64(1.2462954310111538), np.float64(1.2410162142875156), np.float64(1.2405158390474578), np.float64(1.2420690914825976), np.float64(1.2439776415761472), np.float64(1.2415301846771158), np.float64(1.2439868791308026), np.float64(1.2426377905300003), np.float64(1.2407339360705163), np.float64(1.2422149999903702), np.float64(1.2398329637176098), np.float64(1.2405449401273878), np.float64(1.2411143160987625), np.float64(1.242377356923743), np.float64(1.2392027933302168), np.float64(1.239179278225816), np.float64(1.2408205415163787), np.float64(1.2414462635049652), np.float64(1.2414303917833631), np.float64(1.241736028479276), np.float64(1.2413800621527837), np.float64(1.2389049344616676), np.float64(1.2403818005965996), np.float64(1.2424973501326675), np.float64(1.2400143958184997), np.float64(1.2420337660161955), np.float64(1.2420990408929884), np.float64(1.2408614519431944), np.float64(1.2406707453106083), np.float64(1.2406189095807276), np.float64(1.241425713783658), np.float64(1.2405435323915028), np.float64(1.2404086789861932), np.float64(1.2385587887173124), np.float64(1.23808734587093), np.float64(1.2392648722762745), np.float64(1.2378920441240309), np.float64(1.2386263042022427), np.float64(1.2386773053030784), np.float64(1.2398156366612674), np.float64(1.2406871370713397), np.float64(1.238899799398311), np.float64(1.240932753779921), np.float64(1.2370102397908225), np.float64(1.2389216193056733), np.float64(1.2392690887676587), np.float64(1.2401705332144397), np.float64(1.2387336493098466), np.float64(1.2353347748251728), np.float64(1.2349771283659108), np.float64(1.2360202708930115), np.float64(1.239738818961551), np.float64(1.236419691576341), np.float64(1.2395510452226832), np.float64(1.2405060626286586), np.float64(1.2397611621567082), np.float64(1.2381657482054262), np.float64(1.241774191885821), np.float64(1.242083518091227), np.float64(1.2403700858756803), np.float64(1.2432576805235902), np.float64(1.2387787384215685), np.float64(1.2410850830758176), np.float64(1.2426008063550684), np.float64(1.2425818506056907), np.float64(1.2404694377607217), np.float64(1.2414854799687047), np.float64(1.2437827723113182), np.float64(1.2402336906540161), np.float64(1.2384711311725096), np.float64(1.237155560439204), np.float64(1.2408651000065347), np.float64(1.238129444571117), np.float64(1.2410413485747727), np.float64(1.2417492595621953), np.float64(1.2412492523272503), np.float64(1.238370908895238), np.float64(1.2405743781420253), np.float64(1.2411057209263507), np.float64(1.2429488772573745), np.float64(1.240567741733545), np.float64(1.2392817399582596), np.float64(1.2404715910995754), np.float64(1.2410366664779346), np.float64(1.2412766971433604), np.float64(1.241688029690722), np.float64(1.2404702928334916), np.float64(1.242946847261607), np.float64(1.2426533963133721), np.float64(1.239467704523718), np.float64(1.2411612152009373), np.float64(1.2427212825523608), np.float64(1.2437217548522748), np.float64(1.2425273326091322), np.float64(1.242023770451513), np.float64(1.2425148125566734), np.float64(1.238635996613448), np.float64(1.2398595894818754), np.float64(1.2398154139422304), np.float64(1.2395035335312166), np.float64(1.240697046502276), np.float64(1.2447537308552667), np.float64(1.2462378637023914), np.float64(1.242519327488697), np.float64(1.246687145954978), np.float64(1.2417121200331769), np.float64(1.2466028623859031), np.float64(1.2412711558841982), np.float64(1.2455884317852608), np.float64(1.2446754025812112), np.float64(1.2440516770552943), np.float64(1.245868613818607), np.float64(1.2422434671520421), np.float64(1.2435875824553548), np.float64(1.2448135606150543), np.float64(1.243824526366663), np.float64(1.2439359762309483), np.float64(1.241693807412156), np.float64(1.241606296390511), np.float64(1.243646637157946), np.float64(1.2419495083366194), np.float64(1.2444856219441196), np.float64(1.2453698945234426), np.float64(1.2428756141032462), np.float64(1.242019901427041), np.float64(1.2414416927513727), np.float64(1.2421407075659037), np.float64(1.2427305697499575), np.float64(1.2449739628777126), np.float64(1.2443124423782452), np.float64(1.242985436331954), np.float64(1.2416827132457116), np.float64(1.2438246536758188), np.float64(1.243656158388229), np.float64(1.243691373481799), np.float64(1.2447706420096114), np.float64(1.2420514319826996), np.float64(1.2444846908061398), np.float64(1.2458419425647485), np.float64(1.2429113434200267), np.float64(1.245246233904032), np.float64(1.2454797654170795), np.float64(1.2471923340466657), np.float64(1.2450980021886022), np.float64(1.2447165403704568), np.float64(1.244728826153973), np.float64(1.2410595930727344), np.float64(1.2415912180511097), np.float64(1.2450860834059554), np.float64(1.2409341999633132), np.float64(1.2437147157741844), np.float64(1.242164863796723), np.float64(1.2397926511735202), np.float64(1.2443278946608496), np.float64(1.2432145998746673), np.float64(1.2419118845585897), np.float64(1.2425204256467852), np.float64(1.2470596154772657), np.float64(1.245289338861864), np.float64(1.242777145440631), np.float64(1.2470298208404227), np.float64(1.2441248473837256), np.float64(1.245383323342414), np.float64(1.2468144753946548), np.float64(1.2465114812007605), np.float64(1.242932770657088), np.float64(1.2446786210383451), np.float64(1.244597027634354), np.float64(1.243140699228579), np.float64(1.241085033987263), np.float64(1.2403702983218032), np.float64(1.2447649665496299), np.float64(1.2468473777702542), np.float64(1.2423290080780682), np.float64(1.2427134714786405), np.float64(1.2424888844015365), np.float64(1.2433498238697882), np.float64(1.2446655341205137), np.float64(1.239472334634905), np.float64(1.2415754692395706), np.float64(1.244984729975207), np.float64(1.2436093341296335), np.float64(1.2442664925597284), np.float64(1.2451644398184984), np.float64(1.245433598736607), np.float64(1.246548502628951), np.float64(1.2442440230923406), np.float64(1.2429932362705056), np.float64(1.246104845485128), np.float64(1.2472815508959916), np.float64(1.2462700932993012), np.float64(1.2430793098275366), np.float64(1.24297340525965), np.float64(1.2431253768961616), np.float64(1.2456640807017156), np.float64(1.246646537741154), np.float64(1.2456335644264558), np.float64(1.2438996881328641), np.float64(1.2480220936912434), np.float64(1.2475603620603188), np.float64(1.2486616066992606), np.float64(1.2449081755131217), np.float64(1.2449568256585803), np.float64(1.246926148970378), np.float64(1.2462540208125064), np.float64(1.2440099900750021), np.float64(1.2458859433186185), np.float64(1.2451531125195445), np.float64(1.2443461264372433), np.float64(1.2454996414517499), np.float64(1.2475408754987887), np.float64(1.2440874560922315), np.float64(1.2456164299374572), np.float64(1.246874527112354), np.float64(1.2456124299776983), np.float64(1.2474892994603417), np.float64(1.2493292722841618), np.float64(1.2447142172607617), np.float64(1.2457705588980375), np.float64(1.2442529578246513), np.float64(1.2421715026627964), np.float64(1.245163864586074), np.float64(1.2469109591783363), np.float64(1.2430624509747867), np.float64(1.2475569181053814), np.float64(1.2412338709843131), np.float64(1.2455164313959797), np.float64(1.2433388085229777), np.float64(1.2454986420566259), np.float64(1.2447805861512276), np.float64(1.2436508138865687), np.float64(1.244598912746929), np.float64(1.2448214104667004), np.float64(1.2432095411217232), np.float64(1.244872716096722), np.float64(1.2451592052453568), np.float64(1.244735258677318), np.float64(1.244650639102431), np.float64(1.2423494154905383), np.float64(1.2439969745535737), np.float64(1.2444417032259263), np.float64(1.2413381896964824), np.float64(1.2434678332941582), np.float64(1.2435518126449558), np.float64(1.2455297052247145), np.float64(1.246912093911863), np.float64(1.242997059528869), np.float64(1.243980472537224), np.float64(1.2453354909255125), np.float64(1.242113852695748), np.float64(1.2418848185308162), np.float64(1.243597262851778), np.float64(1.2408607493086827), np.float64(1.2411512952592405), np.float64(1.2417632464021564), np.float64(1.2435459134718925), np.float64(1.2408520224989452), np.float64(1.2443415524319141), np.float64(1.2458111571047423), np.float64(1.245236268598962), np.float64(1.2467115168337308), np.float64(1.243273892655633), np.float64(1.244689050923267), np.float64(1.2459035522183945), np.float64(1.2426655784264253), np.float64(1.2471569414865002), np.float64(1.2452802821456161), np.float64(1.2435487803734946), np.float64(1.2440708284925281), np.float64(1.2443004482263256), np.float64(1.2448624690691215), np.float64(1.2444529384552223), np.float64(1.2448715288029994), np.float64(1.2436986904604141), np.float64(1.2434454152424959), np.float64(1.2448380229851281), np.float64(1.2436532196787144), np.float64(1.2446185142611046), np.float64(1.2432177117950474), np.float64(1.2431437935905723), np.float64(1.2421163341422454), np.float64(1.2437661863498701), np.float64(1.2415187631474742), np.float64(1.243179439626034), np.float64(1.2430844085502213), np.float64(1.2405402727077632), np.float64(1.2424961642987196), np.float64(1.2446957340806297), np.float64(1.2454034052267384), np.float64(1.2471503143273335), np.float64(1.2412155412234804), np.float64(1.2444628819768215), np.float64(1.242987816843192), np.float64(1.2427117477152074), np.float64(1.2409929011990124), np.float64(1.2434324224428652), np.float64(1.2416782839638094), np.float64(1.242846439668469), np.float64(1.2417053781894547), np.float64(1.241969350087135), np.float64(1.2411848484305925), np.float64(1.2433194784682822), np.float64(1.242328018513245), np.float64(1.242560895370572), np.float64(1.2426787581020702), np.float64(1.243323735486765), np.float64(1.2462523303267743), np.float64(1.2446761555402435), np.float64(1.2433026951559152), np.float64(1.2454744690515493), np.float64(1.2447890295501236), np.float64(1.2489514911396153), np.float64(1.243210480489631), np.float64(1.244382031262262), np.float64(1.243105379014033), np.float64(1.246914556019883), np.float64(1.2458301218091405), np.float64(1.2469088322201771), np.float64(1.2410854778154308), np.float64(1.241873026682644), np.float64(1.2442873297394568), np.float64(1.2434687695339584), np.float64(1.2451350040858475), np.float64(1.246224432080897), np.float64(1.2423722727245672), np.float64(1.2424889588752477), np.float64(1.2467978809401556), np.float64(1.2496005095226612), np.float64(1.2450988524251103), np.float64(1.2444283766672963), np.float64(1.245032711315848), np.float64(1.2460684871537817), np.float64(1.2444483229752477), np.float64(1.247551880897807), np.float64(1.2481484173577324), np.float64(1.2498295052349653), np.float64(1.2452753508826282), np.float64(1.2473476039314637), np.float64(1.2456369725655367), np.float64(1.2468411132613988), np.float64(1.2481610315079983), np.float64(1.24935299880156), np.float64(1.2492313040122351), np.float64(1.2456382995162172), np.float64(1.2424015158539317), np.float64(1.2426155193811828), np.float64(1.2434560406559738), np.float64(1.2408726121497675), np.float64(1.241745717455101), np.float64(1.2414124540644986), np.float64(1.2407940758528209), np.float64(1.2397360363148806), np.float64(1.2387261379408665), np.float64(1.2420477749687284), np.float64(1.2431098729911412), np.float64(1.2447234821876987), np.float64(1.2443639873796624), np.float64(1.2445108519850199), np.float64(1.2460361658604877), np.float64(1.2445585040744822), np.float64(1.243204524748663), np.float64(1.2439447141534377), np.float64(1.2399765738693789), np.float64(1.2454827944884748), np.float64(1.2464167993075879), np.float64(1.2445869381302013), np.float64(1.2424536229319505), np.float64(1.2450967120240384), np.float64(1.2433683612993036), np.float64(1.242697672279202), np.float64(1.243214482480256), np.float64(1.2406197708636049), np.float64(1.2440501261672885), np.float64(1.2439662013479496), np.float64(1.2405842120509336), np.float64(1.2415897037839396), np.float64(1.2444297785288192), np.float64(1.2414512274516785), np.float64(1.2447501555925005), np.float64(1.2424506670592323), np.float64(1.240701177108398), np.float64(1.2430409989805014), np.float64(1.2400472611472717), np.float64(1.2408006153435336), np.float64(1.2457552902611613), np.float64(1.242194493418179), np.float64(1.241839518227213), np.float64(1.242108423187704), np.float64(1.245428354608088), np.float64(1.2424876719966202), np.float64(1.2427369236275423), np.float64(1.2406916910317294), np.float64(1.2452951063157303), np.float64(1.2432549742563672), np.float64(1.2409197708394935), np.float64(1.2418054662174431), np.float64(1.2412014176993509), np.float64(1.2449250409690784), np.float64(1.2346039571787915), np.float64(1.2406926941335055), np.float64(1.2387522401125612), np.float64(1.239174480919401), np.float64(1.2409805098238527), np.float64(1.239345363432821), np.float64(1.2384526668508977), np.float64(1.2400025595040889), np.float64(1.2418248323108891), np.float64(1.2404944564421958), np.float64(1.241055502845811), np.float64(1.2422737299077795), np.float64(1.2400348563506034), np.float64(1.2445652258495243), np.float64(1.2446708965518583), np.float64(1.2435201000590461), np.float64(1.244666985487022), np.float64(1.244692680497847), np.float64(1.2432010909525695), np.float64(1.2464863713017331), np.float64(1.2439045634149068)]
        list_of_offsets = [np.float64(0.02037347941893275), np.float64(1.2925216019844363), np.float64(1.6306895001576713), np.float64(1.832688889574754), np.float64(1.980078155785319), np.float64(2.0674427746932413), np.float64(2.1028721055798947), np.float64(2.180455669379112), np.float64(2.2061259123142634), np.float64(2.27588763310755), np.float64(2.2701878314830815), np.float64(2.303962196552193), np.float64(2.3539515987990725), np.float64(2.354426213065637), np.float64(2.3728532594208174), np.float64(2.3985150719399373), np.float64(2.4450580536774793), np.float64(2.403203249744025), np.float64(2.459591828116969), np.float64(2.445034367673302), np.float64(2.436798952030364), np.float64(2.4722570858577733), np.float64(2.4657584638029406), np.float64(2.502640496002104), np.float64(2.52637730665794), np.float64(2.5251136453153005), np.float64(2.545335878611686), np.float64(2.563385357650387), np.float64(2.5406225128129813), np.float64(2.5548010138971358), np.float64(2.5636303650016856), np.float64(2.597089062476855), np.float64(2.594265782744719), np.float64(2.618439586963173), np.float64(2.5884513344924884), np.float64(2.596380304340225), np.float64(2.594062712813102), np.float64(2.648950470289434), np.float64(2.6286665943065373), np.float64(2.625249268676835), np.float64(2.6358714013158457), np.float64(2.6100528993204986), np.float64(2.6233561786114774), np.float64(2.6331978062850028), np.float64(2.644208757273796), np.float64(2.66193694011279), np.float64(2.662102855498178), np.float64(2.6384587682050133), np.float64(2.640019646568816), np.float64(2.6822186619188737), np.float64(2.6788083508420915), np.float64(2.6861443940419196), np.float64(2.6892056585008435), np.float64(2.7015240961727196), np.float64(2.6965622921408094), np.float64(2.7007566486439183), np.float64(2.670668057858278), np.float64(2.7188826835874953), np.float64(2.717678491760551), np.float64(2.692809263248919), np.float64(2.698817165894522), np.float64(2.66966250148411), np.float64(2.677961839554135), np.float64(2.694489406759738), np.float64(2.6694736275602255), np.float64(2.697880279704362), np.float64(2.6953032523916427), np.float64(2.678520315794402), np.float64(2.7100359288788467), np.float64(2.715317384474935), np.float64(2.718780168600624), np.float64(2.6863833816063645), np.float64(2.7253954480666898), np.float64(2.7038581183975423), np.float64(2.7284069561395814), np.float64(2.7154373084171404), np.float64(2.7598043205096747), np.float64(2.716655126906634), np.float64(2.7315593634744455), np.float64(2.755389452480219), np.float64(2.6958377817815555), np.float64(2.7297850936431636), np.float64(2.717024998686088), np.float64(2.748700763537592), np.float64(2.746422537706169), np.float64(2.712753433948297), np.float64(2.727177912064372), np.float64(2.74343251200554), np.float64(2.7323038314201193), np.float64(2.7144218322185205), np.float64(2.7458299360050167), np.float64(2.7631409903418573), np.float64(2.7578435700632826), np.float64(2.752357228558094), np.float64(2.7593301227318747), np.float64(2.7278679445463436), np.float64(2.7128215222171734), np.float64(2.7422049805435598), np.float64(2.7430117215500607), np.float64(2.7216691704975777), np.float64(2.7648999553159306), np.float64(2.723906219529312), np.float64(2.7777797019956956), np.float64(2.7503191313445083), np.float64(2.7457684123153254), np.float64(2.759226044699829), np.float64(2.71041153190048), np.float64(2.7565878218997133), np.float64(2.7655271790775604), np.float64(2.7377725879841512), np.float64(2.7468676837420416), np.float64(2.757746037163129), np.float64(2.7607314503371523), np.float64(2.7789618399400107), np.float64(2.7688857738486736), np.float64(2.7809681492026495), np.float64(2.7956725462688086), np.float64(2.734531816396761), np.float64(2.7738821471394095), np.float64(2.786868335393252), np.float64(2.7920164511720977), np.float64(2.7544732599907453), np.float64(2.798010644589047), np.float64(2.809058530723213), np.float64(2.798050809807747), np.float64(2.8082674035817963), np.float64(2.805497673932007), np.float64(2.8191178664829515), np.float64(2.8051695197854136), np.float64(2.795668037422277), np.float64(2.8040193436437373), np.float64(2.771703370197395), np.float64(2.773697098610527), np.float64(2.793082584058232), np.float64(2.7798591756014592), np.float64(2.8051665891072957), np.float64(2.8072012089079643), np.float64(2.8111068479164967), np.float64(2.777352934004337), np.float64(2.809596684187939), np.float64(2.813941656922791), np.float64(2.829712076388747), np.float64(2.8302692615436906), np.float64(2.8119065186664467), np.float64(2.8522828799341915), np.float64(2.8450709902165396), np.float64(2.814683448938179), np.float64(2.8296392984175927), np.float64(2.8062024901413967), np.float64(2.8720629106323408), np.float64(2.8851096234215228), np.float64(2.832765528010307), np.float64(2.8641388921270727), np.float64(2.8512351728793135), np.float64(2.84863276049952), np.float64(2.827956285303698), np.float64(2.861561780974539), np.float64(2.8668279639540466), np.float64(2.817413413454701), np.float64(2.812216111324786), np.float64(2.838223633374727), np.float64(2.8204824483837494), np.float64(2.8311114663969077), np.float64(2.8616081350824136), np.float64(2.872014534211269), np.float64(2.8629469801520697), np.float64(2.8497722409358675), np.float64(2.8641826811658007), np.float64(2.8724348941572786), np.float64(2.86436916464329), np.float64(2.8844252561522734), np.float64(2.8577137316610397), np.float64(2.8541471158015677), np.float64(2.8603152800050724), np.float64(2.86107639970159), np.float64(2.813291284479293), np.float64(2.8722326540333487), np.float64(2.857788070637322), np.float64(2.8569298092953668), np.float64(2.8344617673930887), np.float64(2.8360512457136493), np.float64(2.8400210954514935), np.float64(2.853101605753347), np.float64(2.8438853419244796), np.float64(2.842966890346151), np.float64(2.799400937287796), np.float64(2.842301541663608), np.float64(2.838698013297005), np.float64(2.835576295348762), np.float64(2.820407651878307), np.float64(2.8314951993795807), np.float64(2.8224007681852425), np.float64(2.8372541696688236), np.float64(2.846871202441123), np.float64(2.8802516488150958), np.float64(2.872404695402646), np.float64(2.852055593164564), np.float64(2.8391815791725463), np.float64(2.8510754891196837), np.float64(2.8439546410906735), np.float64(2.8548985261588404), np.float64(2.8628411151871007), np.float64(2.8218339745850964), np.float64(2.8611906937855878), np.float64(2.8290118089899594), np.float64(2.835595797527794), np.float64(2.862374700362118), np.float64(2.82507743297106), np.float64(2.8350464069270687), np.float64(2.893016787182354), np.float64(2.9320389349115876), np.float64(2.894709342469728), np.float64(2.9229002583128127), np.float64(2.9273047588257044), np.float64(2.907689233782033), np.float64(2.935954867976768), np.float64(2.929400008431945), np.float64(2.9421990369012616), np.float64(2.90247339251007), np.float64(2.9558855856559942), np.float64(2.936523537253993), np.float64(2.916772485141296), np.float64(2.978010185964838), np.float64(2.9399120245793857), np.float64(2.9235591329467705), np.float64(2.930016596950757), np.float64(2.927656665246807), np.float64(2.92260765217539), np.float64(2.9490145632071), np.float64(2.927606283623679), np.float64(2.9566936912512896), np.float64(2.954331358973234), np.float64(2.9624537915936866), np.float64(2.9450722918289496), np.float64(2.941886664914189), np.float64(2.9169545815423015), np.float64(2.9823542348801273), np.float64(2.95522680728773), np.float64(2.9682848247744835), np.float64(2.94369212374485), np.float64(2.9498437543943825), np.float64(2.946227327821872), np.float64(2.9479668201036), np.float64(2.9531918227419616), np.float64(2.9811959820743286), np.float64(2.9709451931619673), np.float64(2.952407112879059), np.float64(2.973514162245867), np.float64(2.9837322498783077), np.float64(2.9697367291074577), np.float64(2.9513585925922032), np.float64(2.967006505874498), np.float64(2.9741620036696212), np.float64(2.9906456882603343), np.float64(2.9652561020057147), np.float64(2.946602874369599), np.float64(2.97621281066964), np.float64(2.9477607639040864), np.float64(2.9280228081249526), np.float64(2.96891564204439), np.float64(2.959409179787123), np.float64(2.975004466025337), np.float64(2.9950206087070184), np.float64(2.953695719738878), np.float64(2.9807912234110274), np.float64(2.9633742854895315), np.float64(2.977428853005821), np.float64(2.9503915114612207), np.float64(2.9797606966683117), np.float64(2.9609608369219624), np.float64(2.9829687621664434), np.float64(2.967069483030095), np.float64(2.9838839765796084), np.float64(2.9868642960106504), np.float64(2.955973457325703), np.float64(2.972224888274726), np.float64(3.0042685479017015), np.float64(2.9980543209457857), np.float64(2.979638316711646), np.float64(2.981374821792621), np.float64(2.949428278277921), np.float64(2.966579589483349), np.float64(2.984370460713318), np.float64(2.999495120852522), np.float64(2.9669749778700836), np.float64(2.9905673719916464), np.float64(2.969085330117603), np.float64(2.9825108122965163), np.float64(2.997035236480219), np.float64(2.9699958846885717), np.float64(3.016397569805184), np.float64(2.990584911010702), np.float64(3.005836849673469), np.float64(2.9430345581215906), np.float64(2.969497915604279), np.float64(2.9936684031017213), np.float64(3.0003153465956016), np.float64(2.991585684252222), np.float64(2.9941890259170107), np.float64(3.0047640871000922), np.float64(2.9732232105926286), np.float64(3.0408953143648376), np.float64(2.9863406799608003), np.float64(3.0072529055222104), np.float64(2.982918187088652), np.float64(3.027123218956714), np.float64(3.0220493503520074), np.float64(3.0207371785232797), np.float64(3.0476102007429664), np.float64(2.9937686086480952), np.float64(3.0153661497013617), np.float64(2.958035348557258), np.float64(2.989537549077886), np.float64(3.004054796926012), np.float64(2.996703816000373), np.float64(2.9849062373530693), np.float64(3.0243513155453448), np.float64(3.0219012535909884), np.float64(3.0189855246531097), np.float64(3.0308566159202166), np.float64(3.0256511100819803), np.float64(3.007860276173569), np.float64(3.0623222215975154), np.float64(3.04181228288736), np.float64(3.0385545738666555), np.float64(3.0143799432093243), np.float64(3.0461747311095264), np.float64(3.035755400426704), np.float64(3.0022917159907023), np.float64(2.996580446532225), np.float64(3.0107699527492016), np.float64(2.983258194060695), np.float64(3.016987752388719), np.float64(3.025917363213177), np.float64(3.0179437847643587), np.float64(3.0380985121850457), np.float64(3.0250364360994473), np.float64(3.0270096084517313), np.float64(3.0435461612940573), np.float64(3.0248318012707176), np.float64(3.027050653659777), np.float64(3.0247018402511894), np.float64(3.0380165494201052), np.float64(3.0389969725942403), np.float64(3.0189168485080313), np.float64(3.024694552812737), np.float64(2.982721034202431), np.float64(3.0169949275006687), np.float64(3.02874879585195), np.float64(3.0618747361184115), np.float64(3.027700543691), np.float64(3.0582184035270217), np.float64(3.0210034244144457), np.float64(3.0475940654906193), np.float64(3.0540523245380644), np.float64(3.026982518032858), np.float64(3.011431578079196), np.float64(3.0543070059345214), np.float64(3.011602096141168), np.float64(3.0680674495278972), np.float64(3.084666869685931), np.float64(3.0223186827589044), np.float64(3.0603117475225896), np.float64(3.0502089144800064), np.float64(3.07517725669987), np.float64(3.0636896615372873), np.float64(3.0409213684387324), np.float64(3.073325436122541), np.float64(3.001351783054042), np.float64(3.050060259496998), np.float64(3.038440412951615), np.float64(3.0578951329772064), np.float64(3.0710971042847914), np.float64(3.0560172057937707), np.float64(3.065255834862412), np.float64(3.0512868986781436), np.float64(3.0647633021496747), np.float64(3.0549178682692024), np.float64(3.0597956915462396), np.float64(3.0312018015476605), np.float64(3.0542301861106314), np.float64(3.0518600044419184), np.float64(3.0640927341956243), np.float64(3.1003435854889068), np.float64(3.075153076815714), np.float64(3.0914144697340142), np.float64(3.1145274413140314), np.float64(3.0928068063210574), np.float64(3.0597305271716846), np.float64(3.105668315233557), np.float64(3.0648980208720777), np.float64(3.0991309732174024), np.float64(3.105962830829308), np.float64(3.0777473909832915), np.float64(3.0586780342242674), np.float64(3.120468562869906), np.float64(3.081721072600515), np.float64(3.099283590210751), np.float64(3.0956891248276706), np.float64(3.096359750000147), np.float64(3.1120195416282117), np.float64(3.1110414156117994), np.float64(3.1375502555022017), np.float64(3.0866797108253), np.float64(3.0850091915013302), np.float64(3.091803004196743), np.float64(3.055640796672953), np.float64(3.0826527348784976), np.float64(3.0627585008855993), np.float64(3.0996508148615107), np.float64(3.0642951699070577), np.float64(3.0806617207896037), np.float64(3.079561977535409), np.float64(3.1182448962637257), np.float64(3.1314540374114905), np.float64(3.077467836151101), np.float64(3.100701735149461), np.float64(3.1031922961021654), np.float64(3.116711760072965), np.float64(3.1173935979670384), np.float64(3.106518800080999), np.float64(3.1014664735024504), np.float64(3.1187476036209274), np.float64(3.0855817284012446), np.float64(3.11326676545023), np.float64(3.1167378632553073), np.float64(3.1070331203845125), np.float64(3.111853588190545), np.float64(3.128552429860866), np.float64(3.097010485111212), np.float64(3.061785816876134), np.float64(3.0722220294209883), np.float64(3.087338655409886), np.float64(3.0930536504884376), np.float64(3.0726015622273146), np.float64(3.126822430912726), np.float64(3.1157038694425494), np.float64(3.1205392511690992), np.float64(3.1286969074934974), np.float64(3.1148212931630046), np.float64(3.080002065745363), np.float64(3.1210027194262855), np.float64(3.1179839542442), np.float64(3.0859384354306147), np.float64(3.0829203161405534), np.float64(3.083774283691593), np.float64(3.0575560280628364), np.float64(3.0943157465090563), np.float64(3.071967177870662), np.float64(3.095162585230339), np.float64(3.081934171394749), np.float64(3.090985063112434), np.float64(3.0630771192975543), np.float64(3.0994003969777593), np.float64(3.0624044594655073), np.float64(3.0526559804299853), np.float64(3.125690846354906), np.float64(3.0858021477920734), np.float64(3.1125872369014713), np.float64(3.0897892447180473), np.float64(3.074040436077344), np.float64(3.09116706132612), np.float64(3.0711890735288545), np.float64(3.1034230989970655), np.float64(3.0841999936615077), np.float64(3.0841307080185736), np.float64(3.0777877626635526), np.float64(3.0607261156516414), np.float64(3.0608756037166485), np.float64(3.084511943545214), np.float64(3.0865907385431184), np.float64(3.075758736943413), np.float64(3.071650635391002), np.float64(3.082063259965216), np.float64(3.072328121176109), np.float64(3.092575883340341), np.float64(3.090858758468952), np.float64(3.067099845003134), np.float64(3.0993444636576077), np.float64(3.1169001547998247), np.float64(3.0994884799384996), np.float64(3.0970165122741076), np.float64(3.0963858482318183), np.float64(3.125857551288547), np.float64(3.1444444238482263), np.float64(3.086280141066927), np.float64(3.086088831944223), np.float64(3.070548339272233), np.float64(3.1201120982913055), np.float64(3.077446140274772), np.float64(3.067031038864584), np.float64(3.117119756831208), np.float64(3.102466064483508), np.float64(3.1063302524674796), np.float64(3.0749686701823924), np.float64(3.0905923096883567), np.float64(3.0785142797654137), np.float64(3.118135199451735), np.float64(3.0850314351139327), np.float64(3.0626300559153345), np.float64(3.089785350460476), np.float64(3.0964073796697344), np.float64(3.123817649511312), np.float64(3.1050317269045324), np.float64(3.1166817043829846), np.float64(3.1339706744430913), np.float64(3.0831231042891587), np.float64(3.1061038792279474), np.float64(3.08011001656452), np.float64(3.146578998444191), np.float64(3.118476826021094), np.float64(3.0893793451128335), np.float64(3.095561576720842), np.float64(3.117128918302004), np.float64(3.1071856883687716), np.float64(3.094495950061927), np.float64(3.146090306702724), np.float64(3.119543086490916), np.float64(3.123398125300567), np.float64(3.125852836094584), np.float64(3.150318289886227), np.float64(3.133711190507595), np.float64(3.1342513446720304), np.float64(3.1614349131285655), np.float64(3.122503977921565), np.float64(3.13493573432504), np.float64(3.163286442788282), np.float64(3.1708836356014816), np.float64(3.177755459670153), np.float64(3.149694005163035), np.float64(3.1445339212303574), np.float64(3.122242833009465), np.float64(3.1135652845772674), np.float64(3.115880148675977), np.float64(3.1211545416806805), np.float64(3.124022401515316), np.float64(3.121061146716222), np.float64(3.1507110086378263), np.float64(3.1342314343316318), np.float64(3.1399800926249295), np.float64(3.116172428589364), np.float64(3.1095131908247406), np.float64(3.1542688604320253), np.float64(3.1468874556019295), np.float64(3.1334906674747938), np.float64(3.1372239141369276), np.float64(3.1396650917868922), np.float64(3.14700939239525), np.float64(3.1290423959503144), np.float64(3.176497951268906), np.float64(3.169495055957753), np.float64(3.1705642929276423), np.float64(3.1170820352603097), np.float64(3.1528565268819135), np.float64(3.117109831561144), np.float64(3.1317788169666647), np.float64(3.1225718900718404), np.float64(3.139662303877578), np.float64(3.130012788695786), np.float64(3.139290463812511), np.float64(3.1022064321458234), np.float64(3.146014774189839), np.float64(3.1488453402314827), np.float64(3.136687398698028), np.float64(3.1385744687808144), np.float64(3.1647874860926626), np.float64(3.1678905258077723), np.float64(3.1317907606045354), np.float64(3.1635587889778325), np.float64(3.1421098676477666), np.float64(3.119213221766588), np.float64(3.1332830296145593), np.float64(3.1599168064097856), np.float64(3.1335269363820757), np.float64(3.1443700271657313), np.float64(3.1397374837340877), np.float64(3.1375019950656955), np.float64(3.136364456266252), np.float64(3.125204604984794), np.float64(3.1312835962089767), np.float64(3.1330702626510476), np.float64(3.1168900887416684), np.float64(3.1407601780907344), np.float64(3.1321525761036786), np.float64(3.155985681297655), np.float64(3.104120936220279), np.float64(3.122447959880317), np.float64(3.1510766887395434), np.float64(3.1308170589547046), np.float64(3.118432240898911), np.float64(3.1119899356215477), np.float64(3.1248622868089435), np.float64(3.153057069055772), np.float64(3.141634594458933), np.float64(3.108901744629443), np.float64(3.1455989517359964), np.float64(3.144172049155682), np.float64(3.110135999954331), np.float64(3.1029142034671193), np.float64(3.1631400455264993), np.float64(3.1402315872717557), np.float64(3.1184981695297185), np.float64(3.1564029281598747), np.float64(3.135832367193225), np.float64(3.1326281418837447), np.float64(3.1124883072849685), np.float64(3.116038726097379), np.float64(3.1668341873202692), np.float64(3.131069468624899), np.float64(3.1211312369609274), np.float64(3.118573095998178), np.float64(3.156253549056509), np.float64(3.1613424513271573), np.float64(3.100887070027899), np.float64(3.1479180746183233), np.float64(3.1510418522563177), np.float64(3.1031978568969323), np.float64(3.108820015960224), np.float64(3.120624308011272), np.float64(3.123886982051317), np.float64(3.1145032434841964), np.float64(3.0922601260572393), np.float64(3.155443566325686), np.float64(3.1247072398245876), np.float64(3.123017696967737), np.float64(3.117600527834415), np.float64(3.1139908636197724), np.float64(3.1358425031440125), np.float64(3.118673758654803), np.float64(3.1412009645679815), np.float64(3.140415943525269), np.float64(3.1530053559159303), np.float64(3.1344783807844845), np.float64(3.1418773651836256), np.float64(3.1214043695840537), np.float64(3.134947280078242), np.float64(3.123425659413871), np.float64(3.155307209577317), np.float64(3.1441875393460568), np.float64(3.1410015868136267), np.float64(3.1258984484257253), np.float64(3.1565720648181483), np.float64(3.097659827086755), np.float64(3.173168970060578), np.float64(3.140951194247238), np.float64(3.1514674333201778), np.float64(3.161783869346089), np.float64(3.132815069225144), np.float64(3.1341324320151664), np.float64(3.099091186814303), np.float64(3.124219051075129), np.float64(3.0974381635861543), np.float64(3.1249761064289765), np.float64(3.128874335904306), np.float64(3.1023175241902865), np.float64(3.1153792570929557), np.float64(3.1712390429843142), np.float64(3.169595734023023), np.float64(3.1388145657910753), np.float64(3.180211157326203), np.float64(3.152025942584883), np.float64(3.1700840770811083), np.float64(3.1604279724481104), np.float64(3.1365793269553546), np.float64(3.176962498710407), np.float64(3.1592384343745312), np.float64(3.164365994067228), np.float64(3.1451551707413703), np.float64(3.1895927929164816), np.float64(3.1199620522165854), np.float64(3.157611405260637), np.float64(3.1755411382367336), np.float64(3.1555177207133744), np.float64(3.171572600048527), np.float64(3.1553238300567745), np.float64(3.161811437262987), np.float64(3.150618201437938), np.float64(3.1630827662945666), np.float64(3.1580177425982074), np.float64(3.173521562771994), np.float64(3.184404333379322), np.float64(3.1818851732785483), np.float64(3.1608269846776533), np.float64(3.157599059368893), np.float64(3.1663743109766416), np.float64(3.150981984032167), np.float64(3.1491552534366587), np.float64(3.159795461961948), np.float64(3.1964954243874217), np.float64(3.1912063234079584), np.float64(3.1881857312554374), np.float64(3.1791219570213483), np.float64(3.137944511455164), np.float64(3.1728095065962476), np.float64(3.176085868796394), np.float64(3.169508921102139), np.float64(3.1394763147588702), np.float64(3.1438865859724885), np.float64(3.1420933772391693), np.float64(3.130286100278385), np.float64(3.177243467481488), np.float64(3.1464832287407893), np.float64(3.1800806414340874), np.float64(3.1765211027022313), np.float64(3.1675799316899664), np.float64(3.162354974956915), np.float64(3.126726934409397), np.float64(3.1498197433194437), np.float64(3.12573845501916), np.float64(3.1415358942219513), np.float64(3.137366898440721), np.float64(3.151768213372804), np.float64(3.1610413553177628), np.float64(3.1358509211763557), np.float64(3.1441924878397858), np.float64(3.172312860178429), np.float64(3.192499494092966), np.float64(3.1811977364796773), np.float64(3.177933485932161), np.float64(3.160535013235685), np.float64(3.1511991405173303), np.float64(3.129909545700993), np.float64(3.1260423486579505), np.float64(3.147421600054176), np.float64(3.164761416914599), np.float64(3.1658388893651606), np.float64(3.1953943299240737), np.float64(3.163067997611519), np.float64(3.20708654974478), np.float64(3.171665332954422), np.float64(3.2045246669004075), np.float64(3.1686861601817307), np.float64(3.1585681254280846), np.float64(3.1438918025426963), np.float64(3.1712969310978307), np.float64(3.152847714073279), np.float64(3.1626950302417343), np.float64(3.1602786065713366), np.float64(3.220731537731725), np.float64(3.2033141692929115), np.float64(3.1810178881126636), np.float64(3.230183108403251), np.float64(3.1778799172929615), np.float64(3.2010704051107823), np.float64(3.1888038789400075), np.float64(3.1831918210589936), np.float64(3.173046197621969), np.float64(3.1803462588865115), np.float64(3.1824609132293187), np.float64(3.1707981772422755), np.float64(3.1868687086750636), np.float64(3.2053241412735805), np.float64(3.1331656648651243), np.float64(3.1682645663374163), np.float64(3.1755313732006463), np.float64(3.2012482463442926), np.float64(3.1510605307061885), np.float64(3.189277430057908), np.float64(3.1728347721783345), np.float64(3.1472716372850864), np.float64(3.2173784557611875), np.float64(3.2120564067097015), np.float64(3.163964100196391), np.float64(3.1639945523266935), np.float64(3.1527964368278263), np.float64(3.1539868845658057), np.float64(3.1656717790393487), np.float64(3.1203901517997634), np.float64(3.1564705596057787), np.float64(3.171736883907523), np.float64(3.1612109020967845), np.float64(3.1930483899367093), np.float64(3.212421684984992), np.float64(3.2007599478579873), np.float64(3.1346268537536153), np.float64(3.1326261826577504), np.float64(3.1324154927711123), np.float64(3.1535483042033214), np.float64(3.1512025211002803), np.float64(3.164197609732757), np.float64(3.182715489287008), np.float64(3.134453355160445), np.float64(3.1504292048186318), np.float64(3.1578679303755695), np.float64(3.157155906343476), np.float64(3.140741725552327), np.float64(3.1565338285666584), np.float64(3.15874585473132), np.float64(3.1741718578152915), np.float64(3.159958900376185), np.float64(3.15461589610707), np.float64(3.155314989685879), np.float64(3.176114809743875), np.float64(3.1929527330196503), np.float64(3.160357772305782), np.float64(3.154420014450445), np.float64(3.169383775140342), np.float64(3.137151872741909), np.float64(3.172118519125852), np.float64(3.157558908497588), np.float64(3.1587971227403946), np.float64(3.181530597405656), np.float64(3.225425471205295), np.float64(3.200201424329805), np.float64(3.2007568174981387), np.float64(3.218788863504454), np.float64(3.1632868217068784), np.float64(3.2065491474135746), np.float64(3.228845954161622), np.float64(3.1992927316443276), np.float64(3.2140394237958287), np.float64(3.2302981010735476), np.float64(3.21918849863924), np.float64(3.1896644113271098), np.float64(3.2138868111299455), np.float64(3.1753916601193604), np.float64(3.1977115619651095), np.float64(3.2192959093543183), np.float64(3.191863315032899), np.float64(3.170700838562261), np.float64(3.2009804829331676), np.float64(3.2062305646379383), np.float64(3.1663967127603514), np.float64(3.1834645685729384), np.float64(3.162225014661969), np.float64(3.159231724242031), np.float64(3.1815662437253556), np.float64(3.1608998383375586), np.float64(3.1809608858011873), np.float64(3.19326819747895), np.float64(3.1851042569101873), np.float64(3.2119633812644324), np.float64(3.1981860152587336), np.float64(3.1771827481128647), np.float64(3.203579340876691), np.float64(3.2192184650991766), np.float64(3.2079395187126667), np.float64(3.202166513792337), np.float64(3.1826377758514495), np.float64(3.255928902184431), np.float64(3.1936142028275314), np.float64(3.2215508977725698), np.float64(3.220245210429399), np.float64(3.198857558273312), np.float64(3.2053395900693396), np.float64(3.1883613769557146), np.float64(3.211127568516905), np.float64(3.1927559276875783), np.float64(3.211687621603558), np.float64(3.238785681638873), np.float64(3.1992771595250056), np.float64(3.2263155003520794), np.float64(3.2141507670136016), np.float64(3.223340838502276), np.float64(3.2010380360346615), np.float64(3.191453185913387), np.float64(3.2111957916861007), np.float64(3.1799552829550386), np.float64(3.1930915416516474), np.float64(3.214261890463299), np.float64(3.2300110238369113), np.float64(3.2004793370725992), np.float64(3.168316338240771), np.float64(3.1832132899375893), np.float64(3.183311703418625), np.float64(3.1869130354688777), np.float64(3.169399447470097), np.float64(3.189795454474149), np.float64(3.2079923891135804), np.float64(3.1648652392889596), np.float64(3.1703594010487985), np.float64(3.185182443089174), np.float64(3.19670210638642), np.float64(3.1956110234837194), np.float64(3.1825173349537246), np.float64(3.185137418212819), np.float64(3.1942669057753212), np.float64(3.1777899555531652), np.float64(3.166152406604994), np.float64(3.2117409642497874), np.float64(3.206381578963817), np.float64(3.1914867199644403), np.float64(3.16575618409088), np.float64(3.1995164910182257), np.float64(3.1753908356684257), np.float64(3.1943314250160335), np.float64(3.1867459450767677), np.float64(3.169091892246948), np.float64(3.1308994364682867), np.float64(3.1653163323149256), np.float64(3.1314342566188387), np.float64(3.1417237790011026), np.float64(3.1836144027483932), np.float64(3.166115558100624), np.float64(3.122774634719015), np.float64(3.1389027195999515), np.float64(3.177971647508801), np.float64(3.1566405082358617), np.float64(3.1439400805703577), np.float64(3.167627876153729), np.float64(3.1474275367406763), np.float64(3.1269761617888108), np.float64(3.1130315943269897), np.float64(3.1425671195033487), np.float64(3.1377180006404597), np.float64(3.1349613472581686), np.float64(3.1708295526685704), np.float64(3.154969170587467), np.float64(3.122159695107629), np.float64(3.1670078631849834), np.float64(3.1161728371655415), np.float64(3.1341772617784733), np.float64(3.1424691583191864), np.float64(3.096625088557309), np.float64(3.1401168063328715), np.float64(3.110095301707642), np.float64(3.1043156827900815), np.float64(3.163045605554937), np.float64(3.1316121334220104), np.float64(3.1459903319439544), np.float64(3.152266790242874), np.float64(3.130778355798131), np.float64(3.1394078826782166), np.float64(3.1192408939113494), np.float64(3.1472851201312606), np.float64(3.140192612407694), np.float64(3.1415674321219846), np.float64(3.125146524185444), np.float64(3.1614501980500243), np.float64(3.116175324580907), np.float64(3.184629524262923), np.float64(3.128719464935174), np.float64(3.173752122876155), np.float64(3.1672398603115433), np.float64(3.1310353609918837), np.float64(3.160514424071155), np.float64(3.1212233803562026), np.float64(3.1748047949395843), np.float64(3.1331704285024937), np.float64(3.1248803143324224), np.float64(3.1687106503795386), np.float64(3.1835698005072866), np.float64(3.1646281497296664), np.float64(3.141053252537226), np.float64(3.18713628026846), np.float64(3.1688527843794296), np.float64(3.149649763036417), np.float64(3.154542947857719), np.float64(3.1416034827378283), np.float64(3.146275709033714), np.float64(3.1546968978669403), np.float64(3.138609128948028), np.float64(3.1524077306388194), np.float64(3.178312740901251), np.float64(3.161736925859166), np.float64(3.1735322720871393), np.float64(3.173433362847886), np.float64(3.2239854978052103), np.float64(3.175688386883788), np.float64(3.1512073617623146), np.float64(3.1511952255113265), np.float64(3.1627095276463573), np.float64(3.2021701679678203), np.float64(3.2030998597226903), np.float64(3.1553259588687603), np.float64(3.2243998270437975), np.float64(3.1682994228832744), np.float64(3.1371602351145103), np.float64(3.1746761601939424), np.float64(3.191360324937676), np.float64(3.162916870445683), np.float64(3.183564466969095), np.float64(3.157741220882383), np.float64(3.19047401748836), np.float64(3.1972620246321126), np.float64(3.1712697642065506), np.float64(3.168165590042605), np.float64(3.129565986930201), np.float64(3.183260420049243), np.float64(3.1511263036498756), np.float64(3.1860906628539385), np.float64(3.1412692102291295), np.float64(3.152136888597049), np.float64(3.1597701192385017), np.float64(3.2138918285662736), np.float64(3.201803382593627), np.float64(3.1602219056492125), np.float64(3.195104648918807), np.float64(3.1695519505726932), np.float64(3.1613636758729378), np.float64(3.155638705142884), np.float64(3.1584554534823743), np.float64(3.176282271755094), np.float64(3.1659324687762385), np.float64(3.1895217241265748), np.float64(3.1741966914392616), np.float64(3.188130477150181), np.float64(3.155686312207445), np.float64(3.222546944690801), np.float64(3.184906062288999), np.float64(3.163168888503319), np.float64(3.184243634876624), np.float64(3.1592394434870914), np.float64(3.140748769000321), np.float64(3.1434255532556636), np.float64(3.16425390850506), np.float64(3.1461791551522644), np.float64(3.173104311089427), np.float64(3.200863451021464), np.float64(3.1593033096778127), np.float64(3.159691320525993), np.float64(3.185304565448789), np.float64(3.1904043509099935), np.float64(3.1757836928791416), np.float64(3.1554161837648445), np.float64(3.18210533200776), np.float64(3.159834231844651), np.float64(3.1842212901409135), np.float64(3.179063660919539), np.float64(3.1654583835332013), np.float64(3.167153096565123), np.float64(3.160112702485059), np.float64(3.176255127695888), np.float64(3.189486099458572), np.float64(3.1702651087846396), np.float64(3.197381974759611), np.float64(3.1352651550421973), np.float64(3.191386221505166), np.float64(3.220668064122843), np.float64(3.1823752029674846), np.float64(3.1568644992804584), np.float64(3.1899811551177852), np.float64(3.1672639819524497), np.float64(3.1678291543754957), np.float64(3.204327139596095), np.float64(3.178372580319517), np.float64(3.198805652401863), np.float64(3.218217081856635), np.float64(3.1918395905246935), np.float64(3.1876423211922207), np.float64(3.2183182100804357), np.float64(3.2239219281753533), np.float64(3.1989663399520936), np.float64(3.208547660556831), np.float64(3.199117981153889), np.float64(3.1695233308355717), np.float64(3.1920295490338684), np.float64(3.2210340277311755), np.float64(3.1913025333404423), np.float64(3.181722333110686), np.float64(3.2003524399958345), np.float64(3.1939472083224905), np.float64(3.1968481272390226), np.float64(3.2017589179234016), np.float64(3.193555458144138), np.float64(3.211088800673597), np.float64(3.196954859270859), np.float64(3.2170784328431243), np.float64(3.214952266102067), np.float64(3.2514700975646016), np.float64(3.2439104618831944), np.float64(3.2488668099985163), np.float64(3.2733485764416836), np.float64(3.2539487390477024), np.float64(3.245683551729382), np.float64(3.2227558737383397), np.float64(3.2245349912518244), np.float64(3.2316540559734266), np.float64(3.1868724153524637), np.float64(3.256225216648783), np.float64(3.225956418201505), np.float64(3.246341679416017), np.float64(3.227305235995749), np.float64(3.221353578823812), np.float64(3.272970145924087), np.float64(3.28934412780853), np.float64(3.2888705103312854), np.float64(3.223450640846536), np.float64(3.259884994673219), np.float64(3.2410603187872145), np.float64(3.2412080495748428), np.float64(3.230048550637961), np.float64(3.261538703101765), np.float64(3.2315951350902936), np.float64(3.2217536188821176), np.float64(3.231978554832608), np.float64(3.2126574042442937), np.float64(3.25274065704444), np.float64(3.2310861621433875), np.float64(3.211704349016935), np.float64(3.2180076067836927), np.float64(3.249964787721561), np.float64(3.230294371497137), np.float64(3.1982909729424294), np.float64(3.25594139447345), np.float64(3.2464229465722934), np.float64(3.291889765806913), np.float64(3.231218811968751), np.float64(3.26364365758765), np.float64(3.237809451019515), np.float64(3.2308535814743893), np.float64(3.2342822958120174), np.float64(3.2606327706482947), np.float64(3.2246626769905533), np.float64(3.2339215560549266), np.float64(3.2116190596728753), np.float64(3.2465224248721847), np.float64(3.274498606891359), np.float64(3.249012290402471), np.float64(3.232978337936777), np.float64(3.214502200989814), np.float64(3.202280607472969), np.float64(3.217544566104318), np.float64(3.2032925281819824), np.float64(3.206337215213214), np.float64(3.230280728581853), np.float64(3.218931234162882), np.float64(3.2071343167593915), np.float64(3.202314319830831), np.float64(3.214104389459294), np.float64(3.214405327489468), np.float64(3.216919130226067), np.float64(3.261811052860564), np.float64(3.228888959965585), np.float64(3.2315051207845364), np.float64(3.2370034993007), np.float64(3.214553429002815), np.float64(3.173503054840355), np.float64(3.1610241033564397), np.float64(3.218354749437346), np.float64(3.1881628338907095), np.float64(3.213812736213987), np.float64(3.1569334387581733), np.float64(3.2418246380397098), np.float64(3.1816583931454407), np.float64(3.1927151290930267), np.float64(3.195137006940513), np.float64(3.1947464721425036), np.float64(3.2182454530024494), np.float64(3.2110432986931907), np.float64(3.1968204745290736), np.float64(3.1936687876201506), np.float64(3.1822288916643324), np.float64(3.2203708003790927), np.float64(3.2168609445213803), np.float64(3.201085475699364), np.float64(3.2269705023784887), np.float64(3.2049123666935935), np.float64(3.1883409331108234), np.float64(3.2076283811088446), np.float64(3.2170489713108386), np.float64(3.226193671649455), np.float64(3.220192884492777), np.float64(3.2208815328779274), np.float64(3.207485217013679), np.float64(3.205039643729959), np.float64(3.2340895556351246), np.float64(3.2384003200307427), np.float64(3.2065303999666055), np.float64(3.2172884379638904), np.float64(3.2182441742416077), np.float64(3.208592625578948), np.float64(3.239312817944205), np.float64(3.2203301245869724), np.float64(3.1982477426766107), np.float64(3.2279129699650566), np.float64(3.1945325014253103), np.float64(3.2065666823356755), np.float64(3.1775903777924226), np.float64(3.2152623202466217), np.float64(3.220441603511669), np.float64(3.2140859960785724), np.float64(3.2346450603177233), np.float64(3.24960589364544), np.float64(3.20448056724598), np.float64(3.273618595447928), np.float64(3.2215843230199463), np.float64(3.2273283294272805), np.float64(3.261597765190496), np.float64(3.215100533244444), np.float64(3.2167879563591617), np.float64(3.2377378759317668), np.float64(3.218785169932668), np.float64(3.1853303833258675), np.float64(3.2183853159105897), np.float64(3.2362232045626724), np.float64(3.182938541299396), np.float64(3.2227469573644063), np.float64(3.2122539238257355), np.float64(3.1871164974579123), np.float64(3.183373829449095), np.float64(3.229640063256023), np.float64(3.211931017628795), np.float64(3.2082566310663134), np.float64(3.251027770493618), np.float64(3.2597430045838287), np.float64(3.251721453401289), np.float64(3.2137477847341738), np.float64(3.1904406937166834), np.float64(3.230082313322285), np.float64(3.2359232715256088), np.float64(3.2518547434522316), np.float64(3.221425623485162), np.float64(3.2114840793746846), np.float64(3.2618535355609826), np.float64(3.2389431526927304), np.float64(3.2155817992459648), np.float64(3.2196161111974058), np.float64(3.220169910362315), np.float64(3.198518860169401), np.float64(3.193140020509605), np.float64(3.190646761070535), np.float64(3.2241128702354334), np.float64(3.225406148597709), np.float64(3.20462020868632), np.float64(3.178525775385739), np.float64(3.206026909395773), np.float64(3.2235920130564564), np.float64(3.237750204504613), np.float64(3.2344848527854912), np.float64(3.221908990656634), np.float64(3.178995903505695), np.float64(3.2088154459150378), np.float64(3.2364650520425413), np.float64(3.1654173954086477), np.float64(3.1932280218862203), np.float64(3.1725956245298184), np.float64(3.190562379964995), np.float64(3.2176270432065044), np.float64(3.2009546824929123), np.float64(3.207638697312613), np.float64(3.2305212389802036), np.float64(3.2101366995587144), np.float64(3.2338786863642555), np.float64(3.21258554956413), np.float64(3.2051131263403705), np.float64(3.208182127072695), np.float64(3.2323381620185887), np.float64(3.222045295195376), np.float64(3.1862020287726094), np.float64(3.1888876515827573), np.float64(3.188657769941056), np.float64(3.191910819360598), np.float64(3.252480802586073), np.float64(3.2292583600919413), np.float64(3.244973357178085), np.float64(3.2704372961635495), np.float64(3.2318276301704847), np.float64(3.2075496065020985), np.float64(3.2444654732968785), np.float64(3.205001858136852), np.float64(3.271877727848977), np.float64(3.2410763993556184), np.float64(3.2576601584715683), np.float64(3.2378507768086715), np.float64(3.2387860771086707), np.float64(3.239908798622249), np.float64(3.23620299587446), np.float64(3.23429168167176), np.float64(3.258522799375398), np.float64(3.2347667568096883), np.float64(3.2279450007008665), np.float64(3.231760785101473), np.float64(3.2350784815733977), np.float64(3.2700194131378493), np.float64(3.2517522961402583), np.float64(3.2377918917667228), np.float64(3.291403073466721), np.float64(3.2683709574775865), np.float64(3.258285171999126), np.float64(3.2306367882639817), np.float64(3.1910923518866525), np.float64(3.2353087886223757), np.float64(3.224126557317786), np.float64(3.2121570942300712), np.float64(3.252680331610336), np.float64(3.254295974466706), np.float64(3.2449136999258643), np.float64(3.292548389677851), np.float64(3.261762519391044), np.float64(3.2706331176654717), np.float64(3.249470525127596), np.float64(3.2954307094701276), np.float64(3.2549685237983255), np.float64(3.230288431242782), np.float64(3.2244615594421706), np.float64(3.198791558989484), np.float64(3.2515283779841644), np.float64(3.2359461374877143), np.float64(3.226776540720084), np.float64(3.260513745678436), np.float64(3.203256600964612), np.float64(3.2326769364431556), np.float64(3.261541374640935), np.float64(3.254431716997433), np.float64(3.27325469740358), np.float64(3.2659849224249298), np.float64(3.273304637935408), np.float64(3.25093355978617), np.float64(3.262460215929654), np.float64(3.2493998920257963), np.float64(3.2441051874073437), np.float64(3.2583426829561275), np.float64(3.2619596741617216), np.float64(3.2783104539283254), np.float64(3.2682818022375173), np.float64(3.2733621258437355), np.float64(3.2714741822927764), np.float64(3.2947213418302117), np.float64(3.2691052687563813), np.float64(3.252914402456718), np.float64(3.2954360834090832), np.float64(3.2931372015367195), np.float64(3.2523727985094237), np.float64(3.24016533639735), np.float64(3.2448838306202017), np.float64(3.292895115493176), np.float64(3.2596290292263044), np.float64(3.2635553907894796), np.float64(3.2583184742477935), np.float64(3.318162356117114), np.float64(3.2775011377594), np.float64(3.2797432194190224), np.float64(3.2764481015228415), np.float64(3.3209965852659837), np.float64(3.2842036923932465), np.float64(3.3115410947919615), np.float64(3.2591212453520892), np.float64(3.266610403125611), np.float64(3.2743060686797665), np.float64(3.2884075804369117), np.float64(3.2798116309506398), np.float64(3.245264140728403), np.float64(3.2593484773996737), np.float64(3.27937741351797), np.float64(3.25729288192771), np.float64(3.2850712398028996), np.float64(3.2290434351164907), np.float64(3.2813092679508418), np.float64(3.273470169304794), np.float64(3.280777062572382), np.float64(3.260478412233392), np.float64(3.2698904411652507), np.float64(3.2701478319550517), np.float64(3.3323307941114306), np.float64(3.324180083681111), np.float64(3.2822984151866037), np.float64(3.2932790218041554), np.float64(3.2908581039197844), np.float64(3.271646787045729), np.float64(3.305893772478565), np.float64(3.3191890512616657), np.float64(3.266476743775257), np.float64(3.243043856020924), np.float64(3.2738622869637), np.float64(3.272703436988067), np.float64(3.2888909040943375), np.float64(3.2858913031010917), np.float64(3.295808382196931), np.float64(3.2593988897394603), np.float64(3.232158465862446), np.float64(3.2265040515919083), np.float64(3.268182064240273), np.float64(3.2427173249210766), np.float64(3.2809771014911413), np.float64(3.2634213305491957), np.float64(3.258994927218773), np.float64(3.2407740319363056), np.float64(3.2114538648935445), np.float64(3.2674658679864295), np.float64(3.27645656473454), np.float64(3.282017020835045), np.float64(3.2823122873102517), np.float64(3.2998524597748555), np.float64(3.284235188286715), np.float64(3.2723785977392037), np.float64(3.2770987749666274), np.float64(3.2659112259940764), np.float64(3.3085351160500864), np.float64(3.2651350416556526), np.float64(3.263729304253724), np.float64(3.2338908528695636), np.float64(3.2200418152870847), np.float64(3.2320316090237173), np.float64(3.2129819463587914), np.float64(3.2459127836703456), np.float64(3.2546318897507), np.float64(3.2530549119320025), np.float64(3.305419364031439), np.float64(3.2367054600676894), np.float64(3.2132949809904527), np.float64(3.2558432209802652), np.float64(3.258920798063735), np.float64(3.255606416822408), np.float64(3.27672855498368), np.float64(3.2759660611446715), np.float64(3.2631745275442356), np.float64(3.294729031774836), np.float64(3.2794834995506026), np.float64(3.2667217173446685), np.float64(3.312066213655501), np.float64(3.29970149632648), np.float64(3.2541110586919957), np.float64(3.2703420638067877), np.float64(3.2206559656114733), np.float64(3.2670425062189103), np.float64(3.2814189215226364), np.float64(3.277221115157622), np.float64(3.302404263577199), np.float64(3.3031687411893915), np.float64(3.232908869330931), np.float64(3.2932935274511568), np.float64(3.299828322190864), np.float64(3.271067368578491), np.float64(3.2460702452007406), np.float64(3.2677334525957447), np.float64(3.278348369472943), np.float64(3.2766892168039488), np.float64(3.225063303545964), np.float64(3.2617398470185246), np.float64(3.285068623566455), np.float64(3.2669772816877924), np.float64(3.291155107155741), np.float64(3.279084260566996), np.float64(3.3755941967849665), np.float64(3.3115890437873485), np.float64(3.345784882738669), np.float64(3.3329103783779317), np.float64(3.3190172999022174), np.float64(3.344603189877541), np.float64(3.3483556276125936), np.float64(3.3132925324020888), np.float64(3.2961861072390417), np.float64(3.3111360078019865), np.float64(3.300872021032114), np.float64(3.271744293495014), np.float64(3.300543876744251), np.float64(3.260070102305029), np.float64(3.2637497618998976), np.float64(3.2779688187605593), np.float64(3.287278172630751), np.float64(3.2531056492062347), np.float64(3.2715356695103694), np.float64(3.2409754534827893), np.float64(3.2638705119928386)]
        list_of_slopes_err = [np.float64(0.0046242932647590045), np.float64(0.0027788665034447243), np.float64(0.0034329331307116134), np.float64(0.0038611525774452943), np.float64(0.004374143636088077), np.float64(0.004503382175865463), np.float64(0.004463189151573804), np.float64(0.004821392536003733), np.float64(0.005148072225599982), np.float64(0.005251556848439496), np.float64(0.005254962431949179), np.float64(0.00531057191955681), np.float64(0.005322945688606771), np.float64(0.005429009246693155), np.float64(0.0055684172070030935), np.float64(0.005604293927067486), np.float64(0.00552849500162602), np.float64(0.0057753565332538635), np.float64(0.005604980719261648), np.float64(0.00602664616637369), np.float64(0.005812064540302187), np.float64(0.006149681175985074), np.float64(0.005971321512200363), np.float64(0.006093241745121729), np.float64(0.006313846486389233), np.float64(0.006306271788930255), np.float64(0.006056180306006168), np.float64(0.006257000404808564), np.float64(0.006217034945622043), np.float64(0.006454556748433781), np.float64(0.00672607790084024), np.float64(0.006491642961164085), np.float64(0.006602831511663777), np.float64(0.0064734714012999045), np.float64(0.006529379089346851), np.float64(0.006868784837599393), np.float64(0.006758896457517278), np.float64(0.006828181290956642), np.float64(0.006749309890387115), np.float64(0.006773395915676926), np.float64(0.006857795606044285), np.float64(0.007016229451549505), np.float64(0.006729637496220262), np.float64(0.006953024594875001), np.float64(0.007027755566241172), np.float64(0.007034747903732089), np.float64(0.007343210028451107), np.float64(0.006950352542784232), np.float64(0.007286677207853759), np.float64(0.007189864590431774), np.float64(0.007336722612549014), np.float64(0.007587202289155068), np.float64(0.0075317357358196885), np.float64(0.007631837343014686), np.float64(0.007664081550886332), np.float64(0.00777655027728379), np.float64(0.007842505360751399), np.float64(0.007779779245840693), np.float64(0.007982591960889188), np.float64(0.008058529429368107), np.float64(0.008014082643650323), np.float64(0.00777447271019602), np.float64(0.008093177558200394), np.float64(0.008045901735587177), np.float64(0.007873401596845662), np.float64(0.008128112722981763), np.float64(0.008155694197818017), np.float64(0.00818686696405311), np.float64(0.008416507330564742), np.float64(0.008341356498227771), np.float64(0.008413489238681193), np.float64(0.008332706626603001), np.float64(0.008178923560995551), np.float64(0.007977008626105321), np.float64(0.008074219400672612), np.float64(0.008085245695327047), np.float64(0.008438108465214863), np.float64(0.008162481187963705), np.float64(0.008166309664914124), np.float64(0.00843891026121501), np.float64(0.008203600170667603), np.float64(0.008260211021455941), np.float64(0.008514257261868264), np.float64(0.008578869067335343), np.float64(0.008348825766065824), np.float64(0.008615161403694256), np.float64(0.008490428763026591), np.float64(0.00856193979701082), np.float64(0.008930951745009661), np.float64(0.008583388611844385), np.float64(0.009007363056047839), np.float64(0.008697211678753097), np.float64(0.008617448947776489), np.float64(0.008787375491347489), np.float64(0.008598187430550498), np.float64(0.008529034520069577), np.float64(0.008533857607025205), np.float64(0.008489275929745181), np.float64(0.008612230061951857), np.float64(0.008463279354823617), np.float64(0.008444015914005204), np.float64(0.008437910855174445), np.float64(0.00859133275037517), np.float64(0.008391410477675415), np.float64(0.008476630584399274), np.float64(0.0086074952524126), np.float64(0.008457885689652727), np.float64(0.008489974145281345), np.float64(0.00879039762875219), np.float64(0.008535440568158342), np.float64(0.008426244135680003), np.float64(0.008386063321835918), np.float64(0.008531077089694227), np.float64(0.008386683196728143), np.float64(0.00849000588518976), np.float64(0.008690011327413752), np.float64(0.008870900698471447), np.float64(0.008774065346433425), np.float64(0.008734241429961653), np.float64(0.008920213503661486), np.float64(0.008700047924902587), np.float64(0.008693298128597747), np.float64(0.00866473187979429), np.float64(0.008833710606852419), np.float64(0.008788234202011145), np.float64(0.008898827114533436), np.float64(0.00872310993770183), np.float64(0.008932060552542725), np.float64(0.00870839448120196), np.float64(0.009000815258308346), np.float64(0.008790668981927477), np.float64(0.0088799399328381), np.float64(0.008985747484032315), np.float64(0.008764119732507855), np.float64(0.008915179436827634), np.float64(0.009081066113847689), np.float64(0.009224456736325357), np.float64(0.008903809236750378), np.float64(0.008819590448342431), np.float64(0.008832064227217135), np.float64(0.008927337081094004), np.float64(0.009086727712309086), np.float64(0.009086843694705379), np.float64(0.008767571631563241), np.float64(0.009199708669041334), np.float64(0.009025916975185289), np.float64(0.009080324357015154), np.float64(0.008973677135788564), np.float64(0.009104983516875442), np.float64(0.008929605669944142), np.float64(0.008763757299119731), np.float64(0.00911947371064771), np.float64(0.008854200153228842), np.float64(0.00905532717120853), np.float64(0.009096214414330405), np.float64(0.009105579136104176), np.float64(0.008913211855775973), np.float64(0.009049178762442947), np.float64(0.009027459308949881), np.float64(0.00899959296946686), np.float64(0.008901240844075495), np.float64(0.0088905994766004), np.float64(0.00901220563693342), np.float64(0.009177522888467772), np.float64(0.009236050698287337), np.float64(0.009142467330696992), np.float64(0.009247870132106527), np.float64(0.008883622195202525), np.float64(0.00900371058689317), np.float64(0.009106454693361238), np.float64(0.00889874930252086), np.float64(0.008983925379153412), np.float64(0.008748659671409833), np.float64(0.009248936255324912), np.float64(0.008918545910129338), np.float64(0.00909071367192713), np.float64(0.00918327427818435), np.float64(0.009157860557574435), np.float64(0.009112924698453788), np.float64(0.00925903413828153), np.float64(0.00912541832413962), np.float64(0.009026348912461062), np.float64(0.009087966103984222), np.float64(0.009445789892866081), np.float64(0.00909744532257385), np.float64(0.009045624500372601), np.float64(0.009133363236103916), np.float64(0.008939237770462067), np.float64(0.009037444128195874), np.float64(0.009028997856921), np.float64(0.008921110581258109), np.float64(0.009040715419043078), np.float64(0.009022688641601372), np.float64(0.008758115923914997), np.float64(0.008787286114942634), np.float64(0.009158495062888556), np.float64(0.009067746196345736), np.float64(0.009260829945954906), np.float64(0.009093176844869827), np.float64(0.009408932155468456), np.float64(0.009195695740074753), np.float64(0.008990032175586589), np.float64(0.008935583029053477), np.float64(0.00887054497495982), np.float64(0.00880448593411105), np.float64(0.009016114286536278), np.float64(0.00926266821549114), np.float64(0.009082855356583906), np.float64(0.009122546669565825), np.float64(0.009129758995816674), np.float64(0.009339610527117997), np.float64(0.008960165726741999), np.float64(0.009176067047510068), np.float64(0.009157737112247134), np.float64(0.009299515552824685), np.float64(0.009116403353344717), np.float64(0.009262462516333997), np.float64(0.009396896651632682), np.float64(0.009166299318078209), np.float64(0.009331612224232445), np.float64(0.009409844573619723), np.float64(0.009412657165631345), np.float64(0.009197380810067574), np.float64(0.009093648539225277), np.float64(0.009143596949153182), np.float64(0.009235155967783584), np.float64(0.009299801384216856), np.float64(0.009250331740852654), np.float64(0.009314132486503838), np.float64(0.008980542681742478), np.float64(0.009220253204071901), np.float64(0.009397090995018651), np.float64(0.009073083186224275), np.float64(0.009001350330623908), np.float64(0.008835385266485554), np.float64(0.009084524562565347), np.float64(0.00935439546474901), np.float64(0.009089563723815473), np.float64(0.009136181389733069), np.float64(0.008951392592509873), np.float64(0.008994772600550583), np.float64(0.00877833092904942), np.float64(0.009135306612532985), np.float64(0.009192768275444308), np.float64(0.009361607710737347), np.float64(0.009304875737447078), np.float64(0.009334380171393413), np.float64(0.00921955141094957), np.float64(0.009246762070668432), np.float64(0.009357341174960976), np.float64(0.009236559864846235), np.float64(0.00923721208679072), np.float64(0.009352887210842584), np.float64(0.00945575029026481), np.float64(0.009272572724693541), np.float64(0.009105867740619758), np.float64(0.009254795095204434), np.float64(0.009309927819556254), np.float64(0.009186472675438396), np.float64(0.009200613941171599), np.float64(0.009205735686171257), np.float64(0.009680280071734352), np.float64(0.009265210200768), np.float64(0.009339440371461994), np.float64(0.009425797449130987), np.float64(0.009380107595122564), np.float64(0.009431347680442336), np.float64(0.00910298799975079), np.float64(0.009344238948761616), np.float64(0.009203113331231104), np.float64(0.009583568188002102), np.float64(0.00952120165882041), np.float64(0.009528247446283599), np.float64(0.009282779172528472), np.float64(0.009629619743382606), np.float64(0.009704513705400722), np.float64(0.009386515726901244), np.float64(0.009608784005227949), np.float64(0.009483144744866565), np.float64(0.009447802979541878), np.float64(0.009322631432078997), np.float64(0.009609079018603684), np.float64(0.009690972505500092), np.float64(0.009511495378647511), np.float64(0.009451372565185015), np.float64(0.009368546103395867), np.float64(0.009568640825808428), np.float64(0.009393185621484496), np.float64(0.009719058355864387), np.float64(0.009453470999740225), np.float64(0.009753036589238397), np.float64(0.0095764724925872), np.float64(0.009143607871379543), np.float64(0.009402549325781667), np.float64(0.009569034522746173), np.float64(0.009483077428556972), np.float64(0.009590230715076952), np.float64(0.009589157617216215), np.float64(0.009544535045095417), np.float64(0.009364977113463797), np.float64(0.009425687425862493), np.float64(0.009692233892072907), np.float64(0.009725782922383604), np.float64(0.009596673340705726), np.float64(0.009692290507884721), np.float64(0.009681381863833594), np.float64(0.009740325103076822), np.float64(0.009823059728642692), np.float64(0.009857896397973243), np.float64(0.009890928696459892), np.float64(0.00964243940368673), np.float64(0.009735383143616912), np.float64(0.009935837169267051), np.float64(0.0098359605502587), np.float64(0.009738820647277255), np.float64(0.009847426110334186), np.float64(0.009924337432032597), np.float64(0.009638898735838003), np.float64(0.009870330276764455), np.float64(0.00987829825899136), np.float64(0.010073207124101617), np.float64(0.009879530628145887), np.float64(0.010009744239541462), np.float64(0.010104988030449234), np.float64(0.010067382872041179), np.float64(0.009783625099809387), np.float64(0.010060737130413446), np.float64(0.009621806588177278), np.float64(0.009781486304506079), np.float64(0.009819311996731838), np.float64(0.010062509726651861), np.float64(0.009692166502779827), np.float64(0.009703933326604602), np.float64(0.009719718304997621), np.float64(0.009528820514635376), np.float64(0.0096309146837196), np.float64(0.009778071838463189), np.float64(0.009761191478087537), np.float64(0.009781777900946625), np.float64(0.009940612592161242), np.float64(0.00981270095970205), np.float64(0.00987160204223737), np.float64(0.009807115157559393), np.float64(0.009759937276667904), np.float64(0.009839989837406991), np.float64(0.00979279271050034), np.float64(0.009778257068405483), np.float64(0.009843628614107409), np.float64(0.00999749204996884), np.float64(0.00961265049133824), np.float64(0.009921261754627075), np.float64(0.009804515596348078), np.float64(0.009713657993726319), np.float64(0.009787283402673682), np.float64(0.010011263341520667), np.float64(0.009607641269216139), np.float64(0.009796874346270402), np.float64(0.009654373670212974), np.float64(0.009549647275241043), np.float64(0.010011544316091463), np.float64(0.009827132425310069), np.float64(0.009928064395409619), np.float64(0.009911444247698723), np.float64(0.010122080414765078), np.float64(0.009745455214494405), np.float64(0.009953921887729048), np.float64(0.009738705595298571), np.float64(0.009656993009205004), np.float64(0.00973596727961718), np.float64(0.009710857560400785), np.float64(0.009849886812689258), np.float64(0.009383196837374068), np.float64(0.009949631584280989), np.float64(0.009906281635456541), np.float64(0.009918787728047824), np.float64(0.0098664561096531), np.float64(0.00996547825827079), np.float64(0.009738096262458274), np.float64(0.00986850009884356), np.float64(0.010072008507646387), np.float64(0.009858427083626786), np.float64(0.01001901036136266), np.float64(0.010162833813493585), np.float64(0.009889561085398183), np.float64(0.009830546624581176), np.float64(0.009991293157340124), np.float64(0.010108375614703041), np.float64(0.009972007511380486), np.float64(0.009900187427671573), np.float64(0.010043783279171067), np.float64(0.010207880993898322), np.float64(0.01018370371898485), np.float64(0.010069562692244472), np.float64(0.010044543389308954), np.float64(0.009862003864660176), np.float64(0.009815938951597354), np.float64(0.009738179708539149), np.float64(0.009862806890788644), np.float64(0.010016578260225681), np.float64(0.010233557514868109), np.float64(0.009953093005839691), np.float64(0.010122877683338133), np.float64(0.009980925736906331), np.float64(0.010016805777040335), np.float64(0.010119337519885837), np.float64(0.00998174385467839), np.float64(0.009955220918855586), np.float64(0.01006439200243956), np.float64(0.009938986479137457), np.float64(0.009909096178421763), np.float64(0.00992440541671866), np.float64(0.010149192388109268), np.float64(0.010026346512562607), np.float64(0.010249978552808068), np.float64(0.01037178913601012), np.float64(0.010023973344519809), np.float64(0.009832823429100588), np.float64(0.009897400677387548), np.float64(0.010210398407148278), np.float64(0.010214298048234566), np.float64(0.010158008889450837), np.float64(0.010402866152674729), np.float64(0.010254155842095877), np.float64(0.010497252175543178), np.float64(0.010417003317836421), np.float64(0.010295305227580709), np.float64(0.010082113737961008), np.float64(0.010318181772463198), np.float64(0.010317831670178843), np.float64(0.010457910079479115), np.float64(0.010490122066993804), np.float64(0.010179181213355945), np.float64(0.010284490709832967), np.float64(0.010333264559161691), np.float64(0.010464209910197484), np.float64(0.010022285157859072), np.float64(0.010368259483236136), np.float64(0.009842260285641571), np.float64(0.010487984451721364), np.float64(0.010456671664179268), np.float64(0.010297245867258968), np.float64(0.010311736067510525), np.float64(0.010352589440873403), np.float64(0.010259286056272604), np.float64(0.010317446974148239), np.float64(0.01010526369696211), np.float64(0.010361657002761144), np.float64(0.010143791314667763), np.float64(0.00996549977992922), np.float64(0.010095874065086224), np.float64(0.010099554999187159), np.float64(0.010116098211514895), np.float64(0.010098494238591927), np.float64(0.01042977092171122), np.float64(0.01002554546000259), np.float64(0.009704960409980317), np.float64(0.009943309710207276), np.float64(0.010044055596976487), np.float64(0.010131448435642803), np.float64(0.01001111367324261), np.float64(0.009894519641832543), np.float64(0.010087595274535984), np.float64(0.010066128775927078), np.float64(0.009880425073283995), np.float64(0.009946278433683238), np.float64(0.01002848737166793), np.float64(0.010361654176228061), np.float64(0.010117402418841084), np.float64(0.010058484567970151), np.float64(0.010228860491951972), np.float64(0.010159981056222891), np.float64(0.009906697780953684), np.float64(0.010186945618233969), np.float64(0.009971508801421808), np.float64(0.010262341664863752), np.float64(0.010340801539663375), np.float64(0.010175294040521116), np.float64(0.0103768532140519), np.float64(0.010334278748249662), np.float64(0.010377992237141956), np.float64(0.010286046036787382), np.float64(0.010131681370631314), np.float64(0.010318458705336162), np.float64(0.010271125175941774), np.float64(0.010647733782212896), np.float64(0.010414994587180551), np.float64(0.010459769131924151), np.float64(0.010015456276797672), np.float64(0.010199690425516788), np.float64(0.010063797159507547), np.float64(0.01007524693771625), np.float64(0.010287718470265331), np.float64(0.010272426397409109), np.float64(0.010230985927496971), np.float64(0.009976744172073178), np.float64(0.010444270460666547), np.float64(0.010034250842177454), np.float64(0.010207335835095801), np.float64(0.010279859578814532), np.float64(0.010380461798339709), np.float64(0.010348437404701753), np.float64(0.010356430959775854), np.float64(0.010138618265060703), np.float64(0.010319543719728774), np.float64(0.009994619639602716), np.float64(0.010017878120717), np.float64(0.01019317279614887), np.float64(0.010365957072742134), np.float64(0.010225740948266978), np.float64(0.010369057987331554), np.float64(0.010656280505871054), np.float64(0.010356445179152277), np.float64(0.010276661825702658), np.float64(0.010188557003078698), np.float64(0.010363213354685088), np.float64(0.010005806982203597), np.float64(0.010149098123585252), np.float64(0.010264684671986032), np.float64(0.010463798603297079), np.float64(0.010214334624254365), np.float64(0.010088047611628967), np.float64(0.01005443239281723), np.float64(0.010238524757836405), np.float64(0.01036388790203659), np.float64(0.010608843152932762), np.float64(0.010319216151800552), np.float64(0.010464596577934386), np.float64(0.010246600561155327), np.float64(0.010494163426393227), np.float64(0.01049820059410882), np.float64(0.010668810248092766), np.float64(0.010336885267072753), np.float64(0.01015268862552602), np.float64(0.010370868487485358), np.float64(0.0103315750308714), np.float64(0.010322680390187048), np.float64(0.010377838605172688), np.float64(0.010582077055584083), np.float64(0.010298474023959983), np.float64(0.0099649104211169), np.float64(0.010187381582508235), np.float64(0.010193053381654948), np.float64(0.010194824319117928), np.float64(0.010057801019286881), np.float64(0.01048748435672018), np.float64(0.010461978209075589), np.float64(0.010381176219719606), np.float64(0.010363168206578602), np.float64(0.010327221800349042), np.float64(0.010205281172792145), np.float64(0.010402021533736802), np.float64(0.010671384889991793), np.float64(0.010451258198934782), np.float64(0.010477994954994546), np.float64(0.01061996993381083), np.float64(0.010313552446081807), np.float64(0.010372270554540998), np.float64(0.010424972893561292), np.float64(0.010034855631588159), np.float64(0.010606016016864579), np.float64(0.010308470290229028), np.float64(0.01018259985110432), np.float64(0.01033128430675802), np.float64(0.010451019641175298), np.float64(0.010670616109632358), np.float64(0.010521280224150846), np.float64(0.01082118434793754), np.float64(0.010275245320581113), np.float64(0.010280711516056175), np.float64(0.010193317098982814), np.float64(0.01009785681804572), np.float64(0.010222681238465325), np.float64(0.010569367494741966), np.float64(0.010412276015594501), np.float64(0.010467742441620454), np.float64(0.010190929444186918), np.float64(0.010355272706597543), np.float64(0.010349520275877253), np.float64(0.010050162508706039), np.float64(0.010761008639340671), np.float64(0.010632390547202412), np.float64(0.010594697264128154), np.float64(0.010710975585832096), np.float64(0.01041598730139916), np.float64(0.010484093363485688), np.float64(0.010327487804074234), np.float64(0.010856369450228517), np.float64(0.011038425674192257), np.float64(0.010814301485355511), np.float64(0.01070248569728776), np.float64(0.010694686169899601), np.float64(0.010559870807233645), np.float64(0.010605981913049699), np.float64(0.010586473683401777), np.float64(0.010188191257093865), np.float64(0.010566388224858524), np.float64(0.010409122861107203), np.float64(0.010466003358728843), np.float64(0.01038689963991441), np.float64(0.010469632625482441), np.float64(0.010229959306548086), np.float64(0.010181808869752342), np.float64(0.010409074457139044), np.float64(0.010185467741966651), np.float64(0.010330675998685736), np.float64(0.010483112127647113), np.float64(0.010509904150996738), np.float64(0.010401268736416827), np.float64(0.010176332666600969), np.float64(0.010397279847776561), np.float64(0.010317139650578679), np.float64(0.010504529951434765), np.float64(0.010190480510053375), np.float64(0.010322421820962381), np.float64(0.010093092461417081), np.float64(0.01022614780838237), np.float64(0.010188962466043783), np.float64(0.010080560499799874), np.float64(0.010263662316540058), np.float64(0.010360889356165389), np.float64(0.010371364461818921), np.float64(0.01013103941126394), np.float64(0.010613658334728864), np.float64(0.010043300180520584), np.float64(0.010078386031409965), np.float64(0.010264599257323674), np.float64(0.010231500827116595), np.float64(0.01074168379876015), np.float64(0.010577310238695922), np.float64(0.01074689171372724), np.float64(0.010536870873079375), np.float64(0.01048091768262318), np.float64(0.01038305839439073), np.float64(0.010320740787915735), np.float64(0.01052284381414686), np.float64(0.010401835187877188), np.float64(0.010732993333048014), np.float64(0.010508900771775406), np.float64(0.01077847784611619), np.float64(0.01020515020285893), np.float64(0.010312064154585163), np.float64(0.010248527760860228), np.float64(0.01046305567542215), np.float64(0.010337598506937743), np.float64(0.010499087709921499), np.float64(0.010292587918113555), np.float64(0.010310475655380891), np.float64(0.01054262932406826), np.float64(0.010403528874177968), np.float64(0.01033927901608271), np.float64(0.010280435573748706), np.float64(0.010402180201116449), np.float64(0.010504420679237243), np.float64(0.010538658528944655), np.float64(0.010543458200710195), np.float64(0.010635757120660626), np.float64(0.010549496503421), np.float64(0.010510984559571064), np.float64(0.010667078899085696), np.float64(0.0105385702166934), np.float64(0.010552579437893199), np.float64(0.0106487483703727), np.float64(0.010488139928107921), np.float64(0.010492228747509899), np.float64(0.010278389413806663), np.float64(0.010373214162525726), np.float64(0.010758437590119125), np.float64(0.010322499629552), np.float64(0.010654240196974613), np.float64(0.010831434978731852), np.float64(0.01058665829219572), np.float64(0.010342700993985229), np.float64(0.010607067505718307), np.float64(0.010524740988194529), np.float64(0.010536064043731959), np.float64(0.010538663276574272), np.float64(0.010305785317289509), np.float64(0.010206699022921345), np.float64(0.010594478914412096), np.float64(0.010474821638656257), np.float64(0.01023738895233145), np.float64(0.010698543401883085), np.float64(0.010585688040158794), np.float64(0.010580276572330618), np.float64(0.010289399957433564), np.float64(0.010750500498116678), np.float64(0.010587201674865631), np.float64(0.010698077525285647), np.float64(0.010703361837706489), np.float64(0.01078984238074515), np.float64(0.01058880639746361), np.float64(0.010572684201989558), np.float64(0.010567884134015739), np.float64(0.010478919519509548), np.float64(0.010590855094502423), np.float64(0.010530353373847437), np.float64(0.010608449068329862), np.float64(0.010286250195914316), np.float64(0.01072835903395094), np.float64(0.01043254121556567), np.float64(0.010480864106315418), np.float64(0.010601233773675977), np.float64(0.010295769284754223), np.float64(0.01019508451480406), np.float64(0.01046569374618965), np.float64(0.010440102113111596), np.float64(0.010482611424481451), np.float64(0.010600986501019942), np.float64(0.010504087217549725), np.float64(0.010916232688278241), np.float64(0.010555224553891313), np.float64(0.01038360059407806), np.float64(0.010599247389886907), np.float64(0.010517874733844594), np.float64(0.010260678044076727), np.float64(0.010694049090078936), np.float64(0.010468400901067647), np.float64(0.010301437223078384), np.float64(0.010618628965081303), np.float64(0.010765855883329244), np.float64(0.010518751256342128), np.float64(0.010205666745992043), np.float64(0.010565561319639908), np.float64(0.010367394938402366), np.float64(0.01044961513087489), np.float64(0.010621720139395163), np.float64(0.010637042583708388), np.float64(0.010541988976377127), np.float64(0.0105486931864874), np.float64(0.01086942648118215), np.float64(0.01033304489777865), np.float64(0.010570296047952302), np.float64(0.010556783932419768), np.float64(0.01016970628530852), np.float64(0.010724429918423381), np.float64(0.010787862293398464), np.float64(0.01065446964833378), np.float64(0.01094931369978585), np.float64(0.010681391349434396), np.float64(0.010574213585459027), np.float64(0.010768653177946546), np.float64(0.010656609279612754), np.float64(0.010725332592538982), np.float64(0.010800698607011912), np.float64(0.010701046030203339), np.float64(0.010786602578939092), np.float64(0.01080189638716673), np.float64(0.010567192108556413), np.float64(0.01066173192314959), np.float64(0.010684544116604735), np.float64(0.010565718502589654), np.float64(0.010504356143853028), np.float64(0.010475949687768275), np.float64(0.010637740602465785), np.float64(0.010468287701281456), np.float64(0.01078746617999013), np.float64(0.010701230479463406), np.float64(0.01113920254570921), np.float64(0.010702984458535245), np.float64(0.01079119602314185), np.float64(0.010733631415967355), np.float64(0.01113296704353406), np.float64(0.010905404604291572), np.float64(0.010633035125114777), np.float64(0.010793000121286249), np.float64(0.010396960433697556), np.float64(0.010677765228775535), np.float64(0.010639987572381396), np.float64(0.010727414600993286), np.float64(0.011000374578142494), np.float64(0.01076852270990194), np.float64(0.01076080258213309), np.float64(0.010698183062470439), np.float64(0.01082774070676925), np.float64(0.010453542148232206), np.float64(0.010191315458759212), np.float64(0.010587472282693852), np.float64(0.010548304214632882), np.float64(0.010686632154777043), np.float64(0.010894755912406802), np.float64(0.010498032497630524), np.float64(0.010351091210010889), np.float64(0.010101906572911984), np.float64(0.010196200885010684), np.float64(0.010400145378207255), np.float64(0.01058940618126804), np.float64(0.010586730865760176), np.float64(0.01056571615347407), np.float64(0.010470213676205438), np.float64(0.010408935906115331), np.float64(0.010422092787062928), np.float64(0.01029964924250879), np.float64(0.010137938355041946), np.float64(0.010499333913594373), np.float64(0.010841190436142192), np.float64(0.010543110784871074), np.float64(0.010569252756199328), np.float64(0.010480177818213943), np.float64(0.010608009773433534), np.float64(0.01050396275936903), np.float64(0.010495617713162488), np.float64(0.010407300872168596), np.float64(0.010537625186979312), np.float64(0.010369154615492855), np.float64(0.010366601209588938), np.float64(0.01041247816426584), np.float64(0.010159740711279935), np.float64(0.010223480017766722), np.float64(0.010343224168101597), np.float64(0.010156266268224306), np.float64(0.010414033637874685), np.float64(0.010539565996643822), np.float64(0.010203599354704341), np.float64(0.010461503240349874), np.float64(0.010342628811085006), np.float64(0.010349057920135948), np.float64(0.010261227022366324), np.float64(0.010404348418713993), np.float64(0.01036262348848911), np.float64(0.010157480972131308), np.float64(0.010606781773111536), np.float64(0.010671526526182687), np.float64(0.010058573306832243), np.float64(0.010262922315933783), np.float64(0.010545727853822363), np.float64(0.010471072562899645), np.float64(0.010439291620022936), np.float64(0.010323022552707662), np.float64(0.010398864966620411), np.float64(0.010536993847891005), np.float64(0.010647856961509269), np.float64(0.010361067482541812), np.float64(0.01058604620038745), np.float64(0.010245469356090862), np.float64(0.01038019885462614), np.float64(0.010775669874361234), np.float64(0.010534122792566896), np.float64(0.010420519459113238), np.float64(0.0106324310642623), np.float64(0.010800973976888924), np.float64(0.010646009445113036), np.float64(0.010537777312912971), np.float64(0.010413441371245722), np.float64(0.010871930992455783), np.float64(0.010840057736432497), np.float64(0.010620678072661283), np.float64(0.01050392188024372), np.float64(0.010534426800933533), np.float64(0.010831258545985137), np.float64(0.010705269801986213), np.float64(0.01091066010984716), np.float64(0.010805535119687766), np.float64(0.010786208720853086), np.float64(0.010324155899695353), np.float64(0.010528808264786754), np.float64(0.01043534836334493), np.float64(0.01074425224036687), np.float64(0.010606386141132969), np.float64(0.010400654290008738), np.float64(0.010364689067965465), np.float64(0.010366762074009219), np.float64(0.010234805746825275), np.float64(0.010631185627259695), np.float64(0.010436662796986844), np.float64(0.010723581866114959), np.float64(0.01077459199019174), np.float64(0.010660401962723314), np.float64(0.01079028848957419), np.float64(0.010768354122338005), np.float64(0.010819173444805258), np.float64(0.010845186259830105), np.float64(0.010726772280858617), np.float64(0.01071789925370165), np.float64(0.011341458412345115), np.float64(0.01106005575618331), np.float64(0.011510002567599353), np.float64(0.011170495190429933), np.float64(0.011311532175247177), np.float64(0.011148483240305875), np.float64(0.011151698767066799), np.float64(0.010901688924332041), np.float64(0.011504806169154842), np.float64(0.011165835951597553), np.float64(0.01104260138558591), np.float64(0.010866625398588842), np.float64(0.011064269757616138), np.float64(0.011434812643544465), np.float64(0.011530476789514696), np.float64(0.011228818082703327), np.float64(0.011390989121308379), np.float64(0.011723864576624707), np.float64(0.01125614235364821), np.float64(0.011539828992634145), np.float64(0.01147845497295856), np.float64(0.011234241666558674), np.float64(0.011244223073069649), np.float64(0.011149303763409934), np.float64(0.011290423208119581), np.float64(0.011171384257691312), np.float64(0.011367162511847976), np.float64(0.011321127754471545), np.float64(0.011374924301524094), np.float64(0.01134787464183738), np.float64(0.011497550557602015), np.float64(0.011020939013980614), np.float64(0.011436177801195544), np.float64(0.011191642883756304), np.float64(0.010834677146581952), np.float64(0.011396549377298332), np.float64(0.01111689863333533), np.float64(0.011020613562678578), np.float64(0.011045654712967623), np.float64(0.011207978211357014), np.float64(0.01113415546543535), np.float64(0.011414928233753889), np.float64(0.011516756736555585), np.float64(0.011348049710426739), np.float64(0.011320898996029475), np.float64(0.010995219743017483), np.float64(0.011033239482641524), np.float64(0.011304457183494142), np.float64(0.011154176068927834), np.float64(0.011211478071498223), np.float64(0.011389937703007518), np.float64(0.011058813255092673), np.float64(0.011310539195866387), np.float64(0.01104114944606058), np.float64(0.011147944204095263), np.float64(0.011495905932952967), np.float64(0.011403292056292301), np.float64(0.01137652436703612), np.float64(0.011289596213132904), np.float64(0.011307836955518109), np.float64(0.011070881828368523), np.float64(0.011235987979282665), np.float64(0.011239048949316769), np.float64(0.011445766616356146), np.float64(0.011366877776021297), np.float64(0.011332840082756146), np.float64(0.011332541524714034), np.float64(0.011280962656356919), np.float64(0.011122281832547628), np.float64(0.011332873425064505), np.float64(0.011151015391835667), np.float64(0.01140824516078367), np.float64(0.011269571088710456), np.float64(0.011455360220413739), np.float64(0.011566009193443005), np.float64(0.011462382396738642), np.float64(0.011277313710815135), np.float64(0.011601812852254341), np.float64(0.011437438359711935), np.float64(0.011115983781118286), np.float64(0.011222972327031108), np.float64(0.01113694496978989), np.float64(0.01145085149800809), np.float64(0.011487954752287217), np.float64(0.011175509187433777), np.float64(0.010972181760692304), np.float64(0.01135928286662551), np.float64(0.011049226125526553), np.float64(0.010933789112460651), np.float64(0.011247242752807411), np.float64(0.011040968348399549), np.float64(0.011208137562761891), np.float64(0.010950936908520732), np.float64(0.011028611873123309), np.float64(0.011257228749973664), np.float64(0.011439607327918837), np.float64(0.01157757249624176), np.float64(0.011291456967223397), np.float64(0.011222432845644977), np.float64(0.011520554963699605), np.float64(0.011304356667796948), np.float64(0.011073171137286206), np.float64(0.011147210782778532), np.float64(0.01106002930102775), np.float64(0.01115212586413255), np.float64(0.011074209322304239), np.float64(0.010907804303022179), np.float64(0.011234435253185151), np.float64(0.01118501241368228), np.float64(0.010929653337013458), np.float64(0.010861311890384239), np.float64(0.011100670702201344), np.float64(0.011264534393854986), np.float64(0.011024922974689265), np.float64(0.011394874737975746), np.float64(0.011214836341635401), np.float64(0.01115038967035723), np.float64(0.010965000810561717), np.float64(0.01123543109805391), np.float64(0.011429445914118545), np.float64(0.010781975690725136), np.float64(0.010574691653564746), np.float64(0.011262959141641752), np.float64(0.011361240943456603), np.float64(0.011244799532929707), np.float64(0.011374999629707299), np.float64(0.011225026325189168), np.float64(0.011507240540698576), np.float64(0.01121205428038917), np.float64(0.010859567004199606), np.float64(0.011378254005655345), np.float64(0.011035421781763415), np.float64(0.01149604951486248), np.float64(0.011489475238529594), np.float64(0.011531800009079685), np.float64(0.011262741600480039), np.float64(0.011623352316069879), np.float64(0.011150628196046711), np.float64(0.011410343754730106), np.float64(0.011434515014658904), np.float64(0.011348047162468379), np.float64(0.01149147646989394), np.float64(0.01153963398501623), np.float64(0.011430839759809602), np.float64(0.011176568908834672), np.float64(0.011526984037933438), np.float64(0.011547287735687434), np.float64(0.01155461556562589), np.float64(0.01132921741112348), np.float64(0.011732940061998605), np.float64(0.011634553630813742), np.float64(0.0116221418353859), np.float64(0.011428597641004862), np.float64(0.01155783614189493), np.float64(0.011372844248239988), np.float64(0.011439811179147278), np.float64(0.011434767760460601), np.float64(0.011412061355880982), np.float64(0.011350127757845297), np.float64(0.011210576545543643), np.float64(0.011325677301801035), np.float64(0.011430913814517664), np.float64(0.011382232448062055), np.float64(0.011607756888282783), np.float64(0.011642324288914608), np.float64(0.011849874301192978), np.float64(0.011870918013736144), np.float64(0.011629454423718648), np.float64(0.011824846555437666), np.float64(0.011713554913902274), np.float64(0.011993779962368783), np.float64(0.011877493890009767), np.float64(0.011822984254229946), np.float64(0.011857346824957085), np.float64(0.012044981130217684), np.float64(0.011928714598454371), np.float64(0.011768779274302378), np.float64(0.01165752367248598), np.float64(0.011731655787069998), np.float64(0.01177762427657675), np.float64(0.011759951892042451), np.float64(0.01172157402029255), np.float64(0.011824759160773876), np.float64(0.011886787094249281), np.float64(0.011819511317504779), np.float64(0.011994313585557326), np.float64(0.012165691423350222), np.float64(0.011997382247245779), np.float64(0.01220062603322364), np.float64(0.01194179623856634), np.float64(0.011932183448144488), np.float64(0.01197692641498067), np.float64(0.011968816403887906), np.float64(0.011882712026008038), np.float64(0.012158866457626287), np.float64(0.01199317079303987), np.float64(0.012095314838171266), np.float64(0.011759890936563764), np.float64(0.011853662763701688), np.float64(0.011927211273358189), np.float64(0.011766725217788526), np.float64(0.011825078753081086), np.float64(0.012013681094257636), np.float64(0.01213728659551765), np.float64(0.012861272649940159), np.float64(0.012753328182273257), np.float64(0.012455950125806688), np.float64(0.012431436827962595), np.float64(0.012658002788634874), np.float64(0.012731121202854898), np.float64(0.012555146886175253), np.float64(0.012316270873862912), np.float64(0.012477333940489426), np.float64(0.01194147377922148), np.float64(0.012540167344948849), np.float64(0.012409037585603716), np.float64(0.012610484757225178), np.float64(0.012653887167749026), np.float64(0.012666978280300195), np.float64(0.012509697815530922), np.float64(0.01195115027831852), np.float64(0.012404885722805504), np.float64(0.012048753954323001), np.float64(0.012387109748233238), np.float64(0.012374119339067135), np.float64(0.012277436549075831), np.float64(0.011840142825770445), np.float64(0.012693147334905331), np.float64(0.012506864531947888), np.float64(0.012012595056592447), np.float64(0.012212073984467183), np.float64(0.012044194463705346), np.float64(0.012074325743948412), np.float64(0.012125574573432838), np.float64(0.012220096417346874), np.float64(0.012144093630753706), np.float64(0.011999680030316059), np.float64(0.01237403546058343), np.float64(0.011924508388554245), np.float64(0.01207094591995651), np.float64(0.01181292297904798), np.float64(0.012012089115869769), np.float64(0.012388206175534835), np.float64(0.012509063791803395), np.float64(0.01251027357708488), np.float64(0.012182434591407812), np.float64(0.011983447703106978), np.float64(0.01233950785574809), np.float64(0.01189586867019603), np.float64(0.011864623326748779), np.float64(0.01218480328784824), np.float64(0.01242277545541836), np.float64(0.01243080977597436), np.float64(0.012566706961502017), np.float64(0.012598884265679532), np.float64(0.012502737195413399), np.float64(0.0120109329921473), np.float64(0.01235101610467373), np.float64(0.012321928946069724), np.float64(0.012167716679873276), np.float64(0.012300923855720462), np.float64(0.012474160509388585), np.float64(0.012797755148569161), np.float64(0.012393760732056967), np.float64(0.012202789268536181), np.float64(0.012844760497340837), np.float64(0.012380876315983013), np.float64(0.012560782927420735), np.float64(0.012360322461811622), np.float64(0.012107975422515238), np.float64(0.012573985807116094), np.float64(0.012510800958204369), np.float64(0.01251405852674419), np.float64(0.012465302960804744), np.float64(0.012313681618036997), np.float64(0.012052784498405592), np.float64(0.012238448896894507), np.float64(0.012365037441253996), np.float64(0.012231352142386024), np.float64(0.01200790173991014), np.float64(0.012630068187496632), np.float64(0.012583583569476723), np.float64(0.012525958707591423), np.float64(0.012003290764142277), np.float64(0.012617733919078677), np.float64(0.012257123591879584), np.float64(0.012182337035177246), np.float64(0.0122479201211453), np.float64(0.012436099083045134), np.float64(0.012906347095686865), np.float64(0.012349970188594274), np.float64(0.012493822959735933), np.float64(0.012215207750891199), np.float64(0.012111944194537315), np.float64(0.012293544824009735), np.float64(0.012630685646419266), np.float64(0.012165265592493115), np.float64(0.012408090891300162), np.float64(0.012482437015056934), np.float64(0.01283328115806126), np.float64(0.012709992675779292), np.float64(0.012474393371923274), np.float64(0.012295394097708152), np.float64(0.01232536428331635), np.float64(0.01235055862872188), np.float64(0.012713854775827878), np.float64(0.01239052875153524), np.float64(0.012695438032256237), np.float64(0.012450074048653755), np.float64(0.012475511750224873), np.float64(0.012568958980681652), np.float64(0.012695123055272726), np.float64(0.012501237967392498), np.float64(0.012316322043940757), np.float64(0.012328963637708183), np.float64(0.012486834412054913), np.float64(0.012722050598892346), np.float64(0.012322452575641355), np.float64(0.012790574272085064), np.float64(0.012690093728717865), np.float64(0.012418315153639482), np.float64(0.012899273049558742), np.float64(0.012717114436734693), np.float64(0.012671843338904285), np.float64(0.012716090134126196), np.float64(0.01186692571111563), np.float64(0.012884933704083813), np.float64(0.01291029601983551), np.float64(0.01271246731628864), np.float64(0.012564738695685689), np.float64(0.01228678259057946), np.float64(0.012204815984402272), np.float64(0.012303093698750775), np.float64(0.012147848601485364), np.float64(0.012561203530125116), np.float64(0.012594973777479245), np.float64(0.012746968844106078), np.float64(0.01231358972511579), np.float64(0.012454559497699958), np.float64(0.012825359790707544), np.float64(0.013053060058633194), np.float64(0.013000226352574544), np.float64(0.01304157109677687), np.float64(0.01279545095747501), np.float64(0.012616107898611935), np.float64(0.012528955784436798), np.float64(0.012659583596610322), np.float64(0.012783744481542488), np.float64(0.01247097629489954), np.float64(0.012948787036969096), np.float64(0.012802719543812584), np.float64(0.013016824043337686), np.float64(0.012956005357630616), np.float64(0.012616426760610632), np.float64(0.012904481346454711), np.float64(0.012708817261786777), np.float64(0.012562467622823011), np.float64(0.012823324402529332), np.float64(0.012841742657453869), np.float64(0.013117234286205665), np.float64(0.013070812914601033), np.float64(0.012652544673287412), np.float64(0.01263758649794418), np.float64(0.012872403679894208), np.float64(0.012681372154818148), np.float64(0.012712406872710783), np.float64(0.01280704571024888), np.float64(0.012685643358713354), np.float64(0.012683682584967297), np.float64(0.012555177832017391), np.float64(0.012610638216999382), np.float64(0.013057667419640555), np.float64(0.012843280694195557), np.float64(0.012961067955551082), np.float64(0.012790780514670535), np.float64(0.01274604458140693), np.float64(0.01251849073769162), np.float64(0.012756949728446244), np.float64(0.012903084076897314), np.float64(0.012893395134566518), np.float64(0.01241053615440648), np.float64(0.01265485614444711), np.float64(0.01253948800590533), np.float64(0.012865413145314787), np.float64(0.012465777515843497), np.float64(0.012115611958071497), np.float64(0.012228299072210727), np.float64(0.012605246841072847), np.float64(0.012453876725447968), np.float64(0.012143986032883407), np.float64(0.012073210316795096), np.float64(0.01241805434496001), np.float64(0.01267307056187045), np.float64(0.012364042163182353), np.float64(0.012558702350128698), np.float64(0.012573167053286196), np.float64(0.012569725040470997), np.float64(0.011890562523691122), np.float64(0.01261983538258693), np.float64(0.012275686234976633), np.float64(0.01239929238347891), np.float64(0.012644546950052817), np.float64(0.012799420051577443), np.float64(0.012965652129568277), np.float64(0.012643465727903673), np.float64(0.012895458017550472), np.float64(0.012399265335195121), np.float64(0.012264735198178875), np.float64(0.012603629680350847), np.float64(0.01293663371615347), np.float64(0.012485961147593228), np.float64(0.012999423712096026), np.float64(0.012971019801341067), np.float64(0.013012335571151794), np.float64(0.012757336759236106), np.float64(0.01277033235289828), np.float64(0.012639817874356266), np.float64(0.012983016363727281), np.float64(0.012667670990803686), np.float64(0.012566369932900567), np.float64(0.012719204701934644), np.float64(0.013139697827372907), np.float64(0.012976920006503495), np.float64(0.013076571865757345), np.float64(0.013095820719274372), np.float64(0.012992390243567012), np.float64(0.013699125852850646), np.float64(0.01370676778271547), np.float64(0.013785421019576607), np.float64(0.01408806457868848), np.float64(0.013571987976352538), np.float64(0.013773567242593994), np.float64(0.01350806244799491), np.float64(0.013697291689528145), np.float64(0.013469487987590768), np.float64(0.013664847388829018), np.float64(0.01341893240195748), np.float64(0.014032984150851259), np.float64(0.013468839370231151), np.float64(0.01381488533985553), np.float64(0.013560895479907216), np.float64(0.013847789477973353), np.float64(0.01372289259232396), np.float64(0.01350105610814847), np.float64(0.013731840253652062), np.float64(0.013751679759542088), np.float64(0.01364013478126491), np.float64(0.013846397393557282), np.float64(0.014122489982326472), np.float64(0.013596841553765126), np.float64(0.013616537943269223), np.float64(0.014075369378369888), np.float64(0.014036392245548954), np.float64(0.014145787991760878), np.float64(0.014215094827464459), np.float64(0.013778661631877081), np.float64(0.014216710911051067), np.float64(0.014356807717967808), np.float64(0.013841590315562386), np.float64(0.013417705819036757), np.float64(0.01384614039786257), np.float64(0.013985876943132206), np.float64(0.014144810511749577), np.float64(0.014306749513209146), np.float64(0.013783243785547088), np.float64(0.01398467602777638), np.float64(0.01436711275177878), np.float64(0.014132425760923342), np.float64(0.014385123404013253), np.float64(0.013896175278193902), np.float64(0.013756477090752759), np.float64(0.014129217448451716), np.float64(0.014267140584243741), np.float64(0.013922481506349237), np.float64(0.014249004026566548), np.float64(0.013917894276759287), np.float64(0.013925977176910988), np.float64(0.014065810197597516), np.float64(0.013774940140416962), np.float64(0.013666644343331775), np.float64(0.01417489859886181), np.float64(0.014111023243604307), np.float64(0.014117907408021664), np.float64(0.013763130403769482), np.float64(0.01403346803811055), np.float64(0.01392716868360999), np.float64(0.014226964172098615), np.float64(0.013939374594386671), np.float64(0.013985688560599847), np.float64(0.014028725541225343), np.float64(0.014273529760960427), np.float64(0.014430918609724404), np.float64(0.014286188296600008), np.float64(0.014101984544387342), np.float64(0.014363649625634698), np.float64(0.013706184981292923), np.float64(0.014360981439115014), np.float64(0.013356829020647389), np.float64(0.012938208534112029), np.float64(0.012660963506182316), np.float64(0.012861724645099772), np.float64(0.013057396690975774), np.float64(0.012806029426775407), np.float64(0.013165648970637525), np.float64(0.01291789719823688), np.float64(0.013350915558373517), np.float64(0.012983426170491984), np.float64(0.01230990203847219), np.float64(0.012874671362316284), np.float64(0.01240179533730194), np.float64(0.01306999796162955), np.float64(0.0133277008514865), np.float64(0.012855741207909268), np.float64(0.01319244761481573), np.float64(0.013459491886017912), np.float64(0.013421723502768096), np.float64(0.013686937594266223), np.float64(0.013654893675493539), np.float64(0.013304968614766923), np.float64(0.013144065405003253), np.float64(0.013202471560063966), np.float64(0.0132117391366277), np.float64(0.013145062392178958), np.float64(0.013640016285567455), np.float64(0.013291793104095433), np.float64(0.013735687764824565), np.float64(0.01318927731948232), np.float64(0.013249253078265886), np.float64(0.013904103779820732), np.float64(0.013460080234144054), np.float64(0.013481553897052774), np.float64(0.01360632214281493), np.float64(0.013614547942558463), np.float64(0.013153921852393053), np.float64(0.013584009949373354), np.float64(0.013141268707680691), np.float64(0.013394727152827517), np.float64(0.013091970038156719), np.float64(0.013758503187084698), np.float64(0.013849337908999173), np.float64(0.012965394379215926), np.float64(0.013567093069592303), np.float64(0.013481250606658767), np.float64(0.01313367074831828), np.float64(0.013479768919201437), np.float64(0.013356642729048832), np.float64(0.013503849359407499), np.float64(0.013598847456623162), np.float64(0.013582115653080951), np.float64(0.013068742231628006), np.float64(0.013554035134880313), np.float64(0.013091677218688263), np.float64(0.013234084414535515), np.float64(0.013079171041520518), np.float64(0.01311288731097789), np.float64(0.012529679175542525), np.float64(0.012977321591828642), np.float64(0.013493824169856472), np.float64(0.013135594638070883), np.float64(0.013125574855277216), np.float64(0.013012805054509472), np.float64(0.013100904522386776), np.float64(0.013226546183507973), np.float64(0.013101233667961154), np.float64(0.012959846282433567), np.float64(0.012888256834781503), np.float64(0.01326502599952518), np.float64(0.013614711444421349), np.float64(0.013329276647404692), np.float64(0.013043342691281884), np.float64(0.013714673968570559), np.float64(0.013288374530585132)]
        list_of_offsets_err = [np.float64(0.07636147086939273), np.float64(0.04611006559262165), np.float64(0.056963075359872904), np.float64(0.06406857318402714), np.float64(0.07258069235033504), np.float64(0.07472516132915306), np.float64(0.07405823562501716), np.float64(0.08000194788253676), np.float64(0.08542258210024205), np.float64(0.08713971993706227), np.float64(0.08719622592804518), np.float64(0.08811896046093612), np.float64(0.08832428303647641), np.float64(0.09008420752744666), np.float64(0.09239741803126841), np.float64(0.09299272245066774), np.float64(0.09173498417632607), np.float64(0.09583118825020627), np.float64(0.09300412323193877), np.float64(0.10000086532521191), np.float64(0.09644028698691474), np.float64(0.10204240161206544), np.float64(0.09908286206482142), np.float64(0.10110589613149765), np.float64(0.10476641951403068), np.float64(0.10464073026564019), np.float64(0.10049093033898925), np.float64(0.10382316025349612), np.float64(0.10316001205264433), np.float64(0.10710124006599801), np.float64(0.11160662347541064), np.float64(0.1077166155063132), np.float64(0.10956158192577126), np.float64(0.10741509004923583), np.float64(0.10834277731976695), np.float64(0.11321907676378272), np.float64(0.110825760721335), np.float64(0.11196182647095478), np.float64(0.10998929746561933), np.float64(0.10968636751248347), np.float64(0.11032428598608261), np.float64(0.11200483821318255), np.float64(0.10742977627016116), np.float64(0.11099585809221517), np.float64(0.11218883589912337), np.float64(0.11230046328011994), np.float64(0.1163681358903443), np.float64(0.11014250502960607), np.float64(0.11547225650410374), np.float64(0.11393806588713061), np.float64(0.11531905891856785), np.float64(0.11819005656618116), np.float64(0.11732602518173924), np.float64(0.11800063562488285), np.float64(0.11762175170158935), np.float64(0.11934782526613766), np.float64(0.12036004545391159), np.float64(0.1193973824372279), np.float64(0.12125539569902904), np.float64(0.12159364828429417), np.float64(0.12092299935636791), np.float64(0.11641414825809461), np.float64(0.1202610765842225), np.float64(0.11955857783777947), np.float64(0.11699530160960406), np.float64(0.12078019487007882), np.float64(0.12119004726141801), np.float64(0.1216532612216254), np.float64(0.12506561822060502), np.float64(0.1239489086902323), np.float64(0.12502076762998937), np.float64(0.12382037626558587), np.float64(0.12153522518420368), np.float64(0.11853485832820013), np.float64(0.11997936679549037), np.float64(0.12014320899277975), np.float64(0.12538659972588173), np.float64(0.12129089572200123), np.float64(0.12134778607484005), np.float64(0.12539851048431386), np.float64(0.12048030468045799), np.float64(0.1213117063491746), np.float64(0.12504270088020855), np.float64(0.12599160801650702), np.float64(0.12261312967350961), np.float64(0.12652460499711432), np.float64(0.12469274579330981), np.float64(0.12574297975811385), np.float64(0.13116238888562196), np.float64(0.1255506994567522), np.float64(0.13175224635719354), np.float64(0.12721561088604091), np.float64(0.12604891051315328), np.float64(0.12853445345315156), np.float64(0.12576716629559898), np.float64(0.12475565822626197), np.float64(0.12482620265855982), np.float64(0.12417409643165943), np.float64(0.12597257227710207), np.float64(0.12379384270230455), np.float64(0.12351207386319193), np.float64(0.12342277384806469), np.float64(0.12566690019283971), np.float64(0.12274260250278035), np.float64(0.123989136695407), np.float64(0.12590331308305427), np.float64(0.12371494946887894), np.float64(0.12418431421888877), np.float64(0.12857865940710905), np.float64(0.12484935601811419), np.float64(0.12325212438154541), np.float64(0.12266439228813791), np.float64(0.12478553177265012), np.float64(0.1223129102525742), np.float64(0.12381978728324734), np.float64(0.12673670540413617), np.float64(0.12937482749742968), np.float64(0.1279625624770251), np.float64(0.12738176575219828), np.float64(0.1300940160018574), np.float64(0.12688307900666934), np.float64(0.12678464142458368), np.float64(0.12636802601701355), np.float64(0.12883244422381515), np.float64(0.1281692060432212), np.float64(0.12978211177878143), np.float64(0.12721942177380818), np.float64(0.13026679305365077), np.float64(0.12700480682490922), np.float64(0.1312695256961906), np.float64(0.12820471228931835), np.float64(0.12950665960109523), np.float64(0.13104977411024804), np.float64(0.12781751780461104), np.float64(0.13002059516501122), np.float64(0.13243992173637678), np.float64(0.13453115590824075), np.float64(0.1298547729958966), np.float64(0.12862651361858554), np.float64(0.1288084309905149), np.float64(0.13019790856039304), np.float64(0.1325224934336657), np.float64(0.1325241823350619), np.float64(0.12786785874097042), np.float64(0.1341702218660828), np.float64(0.13163561326097173), np.float64(0.13242910311032363), np.float64(0.13087374099819277), np.float64(0.13278873450682754), np.float64(0.13023099259218215), np.float64(0.1278122311143893), np.float64(0.1330000641196664), np.float64(0.12913126649475812), np.float64(0.13206453779853755), np.float64(0.13266084549264615), np.float64(0.13279742366028993), np.float64(0.129991902508875), np.float64(0.13197486809507086), np.float64(0.13165810982261433), np.float64(0.13125170160391983), np.float64(0.12981731481633416), np.float64(0.1296621179596422), np.float64(0.1314356467240314), np.float64(0.13384665930315776), np.float64(0.13470024275200818), np.float64(0.13333540626678048), np.float64(0.13487262039628692), np.float64(0.12956036103819552), np.float64(0.13131175402558176), np.float64(0.1328101883568693), np.float64(0.12978097644593514), np.float64(0.13102320302931242), np.float64(0.12686660696536672), np.float64(0.1341212543621753), np.float64(0.12933017236625893), np.float64(0.13182682437826596), np.float64(0.1331690684055078), np.float64(0.13280053797240246), np.float64(0.13214891226933956), np.float64(0.134267680944288), np.float64(0.13233008507062707), np.float64(0.13089345256235105), np.float64(0.13178697862046232), np.float64(0.13697587741195932), np.float64(0.13192444288522165), np.float64(0.1311729745315835), np.float64(0.13244529431116067), np.float64(0.12963023300886758), np.float64(0.13105434827583728), np.float64(0.1309318641937502), np.float64(0.12936736275260552), np.float64(0.13110178427609231), np.float64(0.13084037616084165), np.float64(0.12700373314738542), np.float64(0.12742674106825075), np.float64(0.1328097360988369), np.float64(0.13149376698499166), np.float64(0.13429372204646814), np.float64(0.13186254124665767), np.float64(0.13644138968890518), np.float64(0.13334919746732715), np.float64(0.13036681408056458), np.float64(0.1295772361109606), np.float64(0.1286341006328913), np.float64(0.1276761633780081), np.float64(0.13074503817748645), np.float64(0.13432038031231014), np.float64(0.13171286870681256), np.float64(0.1322884448601708), np.float64(0.13239302982963358), np.float64(0.13625548396162482), np.float64(0.13071976652433612), np.float64(0.1338695501122363), np.float64(0.13360213531917584), np.float64(0.13567053943921742), np.float64(0.1329991182349752), np.float64(0.13512997405865557), np.float64(0.13709123099071135), np.float64(0.13254249281397348), np.float64(0.13493287319354502), np.float64(0.13606409741909378), np.float64(0.13610476528525006), np.float64(0.13299192112111916), np.float64(0.1314919764346963), np.float64(0.13221421894047614), np.float64(0.13353814121897395), np.float64(0.1344728983293527), np.float64(0.13375757882951836), np.float64(0.13468012286622422), np.float64(0.12985649745301664), np.float64(0.13332264658392776), np.float64(0.13587968159422373), np.float64(0.13119460643696873), np.float64(0.13015736870142539), np.float64(0.12775755282504453), np.float64(0.1313600477776925), np.float64(0.1352623159338591), np.float64(0.13143291002164395), np.float64(0.13210699178796487), np.float64(0.12943498939088516), np.float64(0.13006225650824457), np.float64(0.12693255979506043), np.float64(0.13209434339266976), np.float64(0.13292522158332437), np.float64(0.1353665991352194), np.float64(0.13454627344327083), np.float64(0.13497289832654452), np.float64(0.13331249929749225), np.float64(0.13370596160226805), np.float64(0.1353049081440571), np.float64(0.13355844149358406), np.float64(0.13356787258134245), np.float64(0.13524050741530821), np.float64(0.136727881019866), np.float64(0.13407917558699656), np.float64(0.13166865990711984), np.float64(0.13456132732570325), np.float64(0.13536293429492505), np.float64(0.13356794528732213), np.float64(0.1337735494371774), np.float64(0.13464309675297567), np.float64(0.14158378635877458), np.float64(0.1355129730710461), np.float64(0.13659866357668385), np.float64(0.13786172131540494), np.float64(0.13719346554559084), np.float64(0.13794289979867608), np.float64(0.13314031079548247), np.float64(0.13666884907563837), np.float64(0.13460474357889238), np.float64(0.1401692778395531), np.float64(0.139257103088705), np.float64(0.1393601581459365), np.float64(0.1357699357581486), np.float64(0.14084282836774348), np.float64(0.14193822908669387), np.float64(0.13728719011654902), np.float64(0.14053808532192225), np.float64(0.13870048398245682), np.float64(0.1381835744739765), np.float64(0.1363528157278993), np.float64(0.14054239752357914), np.float64(0.1417401724463527), np.float64(0.13911514360408075), np.float64(0.13823578528228664), np.float64(0.13702436321362726), np.float64(0.13995095249430192), np.float64(0.13738474364767786), np.float64(0.14215095666034194), np.float64(0.13826647621580307), np.float64(0.14264792351162386), np.float64(0.14006549712380328), np.float64(0.13373441873136196), np.float64(0.13752169413737533), np.float64(0.13995670800485147), np.float64(0.1386995015895668), np.float64(0.1402667270483804), np.float64(0.14025102717779567), np.float64(0.13959837861923402), np.float64(0.13697216384573802), np.float64(0.13786011235046983), np.float64(0.14175862504787573), np.float64(0.14224931038460498), np.float64(0.14036095284480157), np.float64(0.14175945155734476), np.float64(0.1415999000932593), np.float64(0.14246200344563595), np.float64(0.14367208044011046), np.float64(0.1441815998188061), np.float64(0.1446647363651767), np.float64(0.14103032461218226), np.float64(0.14238971974099943), np.float64(0.1453215658897084), np.float64(0.14386076875222886), np.float64(0.14244000002886004), np.float64(0.1440284628722608), np.float64(0.14607627069053133), np.float64(0.14187490397881752), np.float64(0.1452813417814612), np.float64(0.14539862187152636), np.float64(0.1482674882094019), np.float64(0.145416762431321), np.float64(0.14733337616405462), np.float64(0.14873526990844035), np.float64(0.14818175481034593), np.float64(0.14400512703266402), np.float64(0.1480839384574172), np.float64(0.1416233225638185), np.float64(0.14397364937743778), np.float64(0.14453040070914264), np.float64(0.14811002766010778), np.float64(0.14265895106127896), np.float64(0.14283214510595327), np.float64(0.14306448673071384), np.float64(0.1402546600091114), np.float64(0.14175738636457658), np.float64(0.14392339092344564), np.float64(0.1436749282374167), np.float64(0.1439779372697249), np.float64(0.14631582792630068), np.float64(0.14415919325374565), np.float64(0.1450245166074105), np.float64(0.14407713179713197), np.float64(0.1433840413691672), np.float64(0.144560098452064), np.float64(0.14386671951547333), np.float64(0.1436531800501394), np.float64(0.14461355520861463), np.float64(0.14687397618106132), np.float64(0.14122023684466545), np.float64(0.14575407350933087), np.float64(0.14403894289951413), np.float64(0.14270414600940196), np.float64(0.14378578397232442), np.float64(0.1470762937855034), np.float64(0.14114664629949744), np.float64(0.14392668711451403), np.float64(0.14183319597446456), np.float64(0.14029465165973698), np.float64(0.14708041624448648), np.float64(0.14437121013966783), np.float64(0.14585400756684783), np.float64(0.14560984129293872), np.float64(0.14870431081918156), np.float64(0.14317128468879087), np.float64(0.14623388424445957), np.float64(0.14307212166612018), np.float64(0.14187167595540434), np.float64(0.14303189448469183), np.float64(0.1426630078518179), np.float64(0.1447054952320216), np.float64(0.1378493159549452), np.float64(0.14617085342085187), np.float64(0.14553399855745952), np.float64(0.1457177234241645), np.float64(0.1449489154651797), np.float64(0.14640365914457373), np.float64(0.14306317119990264), np.float64(0.1449789462564303), np.float64(0.14796870475501417), np.float64(0.14483096018764874), np.float64(0.14594579322121426), np.float64(0.14804085574853948), np.float64(0.14406012315618555), np.float64(0.14320046511758983), np.float64(0.1455420402245956), np.float64(0.14724756784466897), np.float64(0.14526110832425987), np.float64(0.14421491211709572), np.float64(0.14630665865742265), np.float64(0.14869705175575437), np.float64(0.14834486521359902), np.float64(0.14668218490833188), np.float64(0.14631772905732998), np.float64(0.14365869909630985), np.float64(0.14298767466209836), np.float64(0.1418549655842963), np.float64(0.14367039265627277), np.float64(0.14591036692122425), np.float64(0.14907107727623345), np.float64(0.14498558339238968), np.float64(0.14745881899506755), np.float64(0.14539101755591816), np.float64(0.14591368021915926), np.float64(0.147407249498135), np.float64(0.14540293677144642), np.float64(0.14501657979451743), np.float64(0.14660686155901614), np.float64(0.14478009544565487), np.float64(0.1443446877006831), np.float64(0.14456769709802814), np.float64(0.14784214035959461), np.float64(0.14605265857599586), np.float64(0.14931028226159385), np.float64(0.15108468341199952), np.float64(0.14601808786832154), np.float64(0.1432336289791528), np.float64(0.14417431821002957), np.float64(0.1487337224784284), np.float64(0.14879052516702976), np.float64(0.14797057219539597), np.float64(0.1515373756435753), np.float64(0.14937113379857764), np.float64(0.15291228714646174), np.float64(0.15174331439175298), np.float64(0.14997055054652422), np.float64(0.14686501114233455), np.float64(0.1503037902876676), np.float64(0.15029868930861515), np.float64(0.1523391983590306), np.float64(0.15280842487871882), np.float64(0.14827898249432428), np.float64(0.1498130175886974), np.float64(0.15052349977993867), np.float64(0.15243096552459265), np.float64(0.14599349503726955), np.float64(0.1510332661092077), np.float64(0.143371096152557), np.float64(0.15277728666309195), np.float64(0.1523211543633774), np.float64(0.14999881749615426), np.float64(0.15020989707732824), np.float64(0.15080499986342902), np.float64(0.14944586630201845), np.float64(0.15029308775201433), np.float64(0.1472022345979361), np.float64(0.15093709159223725), np.float64(0.14776346544300234), np.float64(0.1458805999073575), np.float64(0.14778909545840443), np.float64(0.14784297671360416), np.float64(0.14808514575019002), np.float64(0.14782745055323948), np.float64(0.15267686682564657), np.float64(0.14675958059347471), np.float64(0.14206668039854617), np.float64(0.14555577438879974), np.float64(0.14703054791062534), np.float64(0.1483098522529725), np.float64(0.14654832631291498), np.float64(0.14484155503631693), np.float64(0.14766790454747442), np.float64(0.14735366357655924), np.float64(0.14463523063130188), np.float64(0.1455992261652877), np.float64(0.14680264866787834), np.float64(0.15167973115971387), np.float64(0.14810423767727374), np.float64(0.1472417653105992), np.float64(0.14973582364349813), np.float64(0.14872752858285512), np.float64(0.14501982627674426), np.float64(0.14912225206202703), np.float64(0.14596856612785097), np.float64(0.15022594297932718), np.float64(0.15137448292392708), np.float64(0.1489516857163599), np.float64(0.1519022260276659), np.float64(0.1512789953194525), np.float64(0.15191889831330882), np.float64(0.15057294014610778), np.float64(0.14831326007499157), np.float64(0.15104741133968183), np.float64(0.1503545160984576), np.float64(0.15586752972640638), np.float64(0.15246055757699054), np.float64(0.1531159955905907), np.float64(0.1466118947028231), np.float64(0.14930881470353058), np.float64(0.14731953771193274), np.float64(0.14748713986023598), np.float64(0.15059742261697556), np.float64(0.15037356595648121), np.float64(0.14976693745106417), np.float64(0.14604520500302984), np.float64(0.15288911831113156), np.float64(0.1468870207131895), np.float64(0.14942073385604865), np.float64(0.1504823761678901), np.float64(0.15195505179102461), np.float64(0.15148625937055576), np.float64(0.15160327174554267), np.float64(0.14841480729937642), np.float64(0.1510632954024252), np.float64(0.14630687594321673), np.float64(0.14664734602975277), np.float64(0.14921340469246575), np.float64(0.1517427208538461), np.float64(0.14969015751978024), np.float64(0.1517881149239517), np.float64(0.15599263758470083), np.float64(0.1516034790008746), np.float64(0.1504355670906659), np.float64(0.14914584153640628), np.float64(0.15170255830388255), np.float64(0.1464706395721895), np.float64(0.14856821719909546), np.float64(0.15026023997766358), np.float64(0.15317498179049327), np.float64(0.14952318669577191), np.float64(0.14767452476812123), np.float64(0.14718244503261904), np.float64(0.1498772948206125), np.float64(0.1517124320828542), np.float64(0.15529822538159488), np.float64(0.1510584995440483), np.float64(0.15318666307397802), np.float64(0.14999551273942507), np.float64(0.1536194777265335), np.float64(0.15367857831194895), np.float64(0.1561760571268375), np.float64(0.15131715448912836), np.float64(0.14862077356749545), np.float64(0.15181461736692797), np.float64(0.1512394171733943), np.float64(0.15110921322805582), np.float64(0.15191665053806516), np.float64(0.15490640493344404), np.float64(0.1507548641929649), np.float64(0.1458719741380286), np.float64(0.14912863576879404), np.float64(0.14921166094624605), np.float64(0.14923758420335304), np.float64(0.14723176258984533), np.float64(0.15352170602826934), np.float64(0.15314833254606777), np.float64(0.1519655091597972), np.float64(0.1517018959904183), np.float64(0.15117569305453896), np.float64(0.14939065724610048), np.float64(0.1522706532125909), np.float64(0.15621374826833306), np.float64(0.15299140794033803), np.float64(0.15338279394829651), np.float64(0.15546110309206426), np.float64(0.15097559315386433), np.float64(0.1518351417561723), np.float64(0.15260662987974508), np.float64(0.14689587295321166), np.float64(0.15525684038374116), np.float64(0.1509011960819709), np.float64(0.14905863753320478), np.float64(0.1512351654246847), np.float64(0.15298791766731037), np.float64(0.15620248821786964), np.float64(0.15401642849496183), np.float64(0.1584065973889276), np.float64(0.1504148286187841), np.float64(0.15049484826958712), np.float64(0.1492155198810398), np.float64(0.14781812037560546), np.float64(0.1496453689899479), np.float64(0.15472035651941485), np.float64(0.15242076182006295), np.float64(0.1532327120732105), np.float64(0.14918056933104665), np.float64(0.15158631636631664), np.float64(0.15150210986120763), np.float64(0.1471199431971726), np.float64(0.157525705301321), np.float64(0.15564292292515716), np.float64(0.1550911467661822), np.float64(0.1567932966246933), np.float64(0.15247509168542758), np.float64(0.15347206367188637), np.float64(0.15117958562999842), np.float64(0.15892165411990533), np.float64(0.16158669264215278), np.float64(0.1583058385029319), np.float64(0.15666901501820274), np.float64(0.15655484181326734), np.float64(0.15458133825822637), np.float64(0.15525633910600872), np.float64(0.15497076536358437), np.float64(0.14914048595133722), np.float64(0.15467674444142349), np.float64(0.15237460640259035), np.float64(0.15320725632534693), np.float64(0.15204928845376217), np.float64(0.15326038309418885), np.float64(0.14975190899019825), np.float64(0.149047059336378), np.float64(0.1523738985515893), np.float64(0.14910061417218315), np.float64(0.15122626068537398), np.float64(0.15345770399352018), np.float64(0.15384989802778926), np.float64(0.15225963545904414), np.float64(0.14896689007430688), np.float64(0.15220124326514828), np.float64(0.15102810262556696), np.float64(0.15377122869807813), np.float64(0.1491739981567401), np.float64(0.1511054276450896), np.float64(0.14774837772819482), np.float64(0.1496961153153945), np.float64(0.1491517762409922), np.float64(0.14756492506800994), np.float64(0.15024527229107584), np.float64(0.15166853821012022), np.float64(0.15182187959053542), np.float64(0.14830386488623662), np.float64(0.1553687137517788), np.float64(0.1470194888079874), np.float64(0.1475330918480771), np.float64(0.15025898639797175), np.float64(0.14977447643351563), np.float64(0.15724282045073468), np.float64(0.15483663004527468), np.float64(0.15731905551436642), np.float64(0.15424465485228478), np.float64(0.15342557803074305), np.float64(0.15199306042847735), np.float64(0.15108081884792035), np.float64(0.15403931830594297), np.float64(0.152267925414943), np.float64(0.15711560643405495), np.float64(0.15383521373331935), np.float64(0.15778143177723764), np.float64(0.14938873874691275), np.float64(0.15095380694070173), np.float64(0.15002372435096065), np.float64(0.15316410510377496), np.float64(0.15132759188092512), np.float64(0.1536915638574319), np.float64(0.15066870075715272), np.float64(0.1509305528199348), np.float64(0.15432895275042546), np.float64(0.15229271815655726), np.float64(0.15135219312811668), np.float64(0.1504908105214177), np.float64(0.15227297779582183), np.float64(0.15376963113101744), np.float64(0.1542708206826608), np.float64(0.1543410829633471), np.float64(0.15569220459783273), np.float64(0.15442947531302903), np.float64(0.15386571660570936), np.float64(0.15615071510312004), np.float64(0.1542695290904847), np.float64(0.15447460553516684), np.float64(0.15588238143558988), np.float64(0.1535313063532677), np.float64(0.1535911578639485), np.float64(0.15046085495033867), np.float64(0.15184895468507315), np.float64(0.15748807288936836), np.float64(0.15110656675342163), np.float64(0.15596277297867395), np.float64(0.1585566501937628), np.float64(0.15497347067064013), np.float64(0.15140228857496663), np.float64(0.15527222864396864), np.float64(0.154067091267739), np.float64(0.15423284248002792), np.float64(0.15427089358646337), np.float64(0.15086189297890423), np.float64(0.1494114123378499), np.float64(0.15508794854153735), np.float64(0.15333634396916712), np.float64(0.14986066859218397), np.float64(0.15661130612493498), np.float64(0.1549592676719948), np.float64(0.15488005031155214), np.float64(0.15062203648104067), np.float64(0.15737188715799794), np.float64(0.1549814235536178), np.float64(0.15660448707893726), np.float64(0.1566818433390743), np.float64(0.15794779022754288), np.float64(0.1550049160126464), np.float64(0.15476890854060577), np.float64(0.15469864356877733), np.float64(0.1533963308119603), np.float64(0.1550349074783144), np.float64(0.15414924665346796), np.float64(0.1552924555317462), np.float64(0.15057592812742196), np.float64(0.15704776669780318), np.float64(0.15271742048440032), np.float64(0.15342479515165802), np.float64(0.15518682976809858), np.float64(0.15071527324760753), np.float64(0.14924139397136715), np.float64(0.15320272315741781), np.float64(0.15282809810756626), np.float64(0.15427186994236336), np.float64(0.15601398537668856), np.float64(0.15458792914369465), np.float64(0.16065344064846468), np.float64(0.15534051244148123), np.float64(0.1528147343907502), np.float64(0.1559883902512211), np.float64(0.15479083660194082), np.float64(0.1510056932089016), np.float64(0.15738358447871312), np.float64(0.15406273702049386), np.float64(0.1516055431601022), np.float64(0.15627363037059633), np.float64(0.15844035691402553), np.float64(0.15480373801482675), np.float64(0.150196095395198), np.float64(0.15549263836192925), np.float64(0.15257623675419227), np.float64(0.1537862622402774), np.float64(0.15631912515822993), np.float64(0.15654462411790673), np.float64(0.15514572835626808), np.float64(0.1552443893985896), np.float64(0.15996459666889734), np.float64(0.1520707085698162), np.float64(0.15556231824402406), np.float64(0.15536346404075824), np.float64(0.14966686614962746), np.float64(0.15783069168424732), np.float64(0.15876422525563177), np.float64(0.15680109676321208), np.float64(0.1611402902332206), np.float64(0.15719730183914826), np.float64(0.1556199741592641), np.float64(0.15848152475180555), np.float64(0.1568325833325609), np.float64(0.1578439811649072), np.float64(0.15895313939213315), np.float64(0.15748655639852474), np.float64(0.158745687631082), np.float64(0.15897076136535132), np.float64(0.15551663734912238), np.float64(0.1569079717946698), np.float64(0.1572437001495346), np.float64(0.1554949519009153), np.float64(0.15459188649919453), np.float64(0.15417383188738526), np.float64(0.15655489430646358), np.float64(0.15406106858642724), np.float64(0.1587583979715488), np.float64(0.15748927208718502), np.float64(0.1639348769614761), np.float64(0.15751508161924754), np.float64(0.15881328838096154), np.float64(0.1579661153968076), np.float64(0.16384310841904048), np.float64(0.1604940898318645), np.float64(0.156485644245934), np.float64(0.15883983741692104), np.float64(0.1530113546332799), np.float64(0.15714393945092525), np.float64(0.15658796337227315), np.float64(0.15787462022505333), np.float64(0.16189175760824454), np.float64(0.15847960376718098), np.float64(0.1583659877814059), np.float64(0.15744442101471942), np.float64(0.15935110877463754), np.float64(0.15384405860785744), np.float64(0.14998488719907696), np.float64(0.1558150996554531), np.float64(0.1552386637094403), np.float64(0.15727443040596167), np.float64(0.16033736912167976), np.float64(0.1543894868443892), np.float64(0.15222849131152397), np.float64(0.1485638529966942), np.float64(0.1499505949362413), np.float64(0.15294990667201463), np.float64(0.15573327674075416), np.float64(0.15569392964231613), np.float64(0.15538487965636927), np.float64(0.1539803663357954), np.float64(0.15307918625829578), np.float64(0.15327267745459452), np.float64(0.15147196089150894), np.float64(0.14909375590373497), np.float64(0.15440862163956845), np.float64(0.15943614409910567), np.float64(0.15505242874489372), np.float64(0.15543688707760342), np.float64(0.15412690520987932), np.float64(0.1560068712270679), np.float64(0.15447669776091855), np.float64(0.1543539750337822), np.float64(0.15305513959484196), np.float64(0.15497175600388025), np.float64(0.15249414006769774), np.float64(0.1524565880114259), np.float64(0.153131278929548), np.float64(0.14941439325021505), np.float64(0.15035177499661065), np.float64(0.15211279156549712), np.float64(0.14936329374875912), np.float64(0.15315415450776457), np.float64(0.15500030023352301), np.float64(0.15005939978443222), np.float64(0.15385226496470203), np.float64(0.15210404038853131), np.float64(0.15219858907728748), np.float64(0.15090690338359744), np.float64(0.15301171623974585), np.float64(0.15239808887846534), np.float64(0.14938116141975355), np.float64(0.1559888088150789), np.float64(0.15694097808057864), np.float64(0.14792657425156802), np.float64(0.15093183264883278), np.float64(0.15509091849213485), np.float64(0.15399299796640195), np.float64(0.15352561254064934), np.float64(0.15181570028357902), np.float64(0.15293107613051604), np.float64(0.15496247375159802), np.float64(0.15659287918802334), np.float64(0.15237520841922111), np.float64(0.1556838630530642), np.float64(0.15067516344841728), np.float64(0.15265656135319206), np.float64(0.15847256578808142), np.float64(0.1549202479605498), np.float64(0.15324953856096205), np.float64(0.1563660211734971), np.float64(0.1588446984010538), np.float64(0.15656571338667463), np.float64(0.15497399250301266), np.float64(0.15314544511790837), np.float64(0.15988823016868242), np.float64(0.15941948455746524), np.float64(0.15619317507746247), np.float64(0.1544760953943777), np.float64(0.15492471931536367), np.float64(0.15929008141063047), np.float64(0.15743722641402427), np.float64(0.16045779915012082), np.float64(0.158911778357107), np.float64(0.15862755643989254), np.float64(0.1518323656288538), np.float64(0.1548420892058712), np.float64(0.15346762251289015), np.float64(0.15801051601384186), np.float64(0.1559829920391809), np.float64(0.1529573906397167), np.float64(0.1524284665773045), np.float64(0.15245895522665623), np.float64(0.15051833837904455), np.float64(0.15634770434234932), np.float64(0.15348694787107678), np.float64(0.15873872831202016), np.float64(0.15949382111871968), np.float64(0.1578034901184537), np.float64(0.15972617517443002), np.float64(0.15940148566954043), np.float64(0.160153749625816), np.float64(0.16053881161376612), np.float64(0.15878595780790827), np.float64(0.1586546103646848), np.float64(0.16549772155700057), np.float64(0.16139141696770007), np.float64(0.1679571644494987), np.float64(0.16300298280776324), np.float64(0.16506103032793756), np.float64(0.16268177455774885), np.float64(0.16272869681879618), np.float64(0.1590804832621741), np.float64(0.16788134247798298), np.float64(0.16293499057908845), np.float64(0.16113671640451144), np.float64(0.1585688238356296), np.float64(0.16145290905447998), np.float64(0.16685997518247955), np.float64(0.1682559341058069), np.float64(0.1638540422035134), np.float64(0.16727342860034267), np.float64(0.17216161138488426), np.float64(0.16529324621901867), np.float64(0.16945909824346006), np.float64(0.16855784223802064), np.float64(0.16497163879801982), np.float64(0.16511821264492602), np.float64(0.16372434984725903), np.float64(0.16579664973014982), np.float64(0.1640485919360663), np.float64(0.16692354333915352), np.float64(0.16624753811377008), np.float64(0.16703752104544267), np.float64(0.16664030512973343), np.float64(0.16883825288744914), np.float64(0.16183934815081383), np.float64(0.16793701184039275), np.float64(0.16434608735807785), np.float64(0.15910414651877355), np.float64(0.16735507913761413), np.float64(0.16324849124172844), np.float64(0.161834569589413), np.float64(0.16220228852942079), np.float64(0.16458596754700647), np.float64(0.1635018994881702), np.float64(0.16762496833825086), np.float64(0.16912029148714952), np.float64(0.1666428742627694), np.float64(0.1662441740460529), np.float64(0.16146166963972894), np.float64(0.16201997678149638), np.float64(0.16600273257748369), np.float64(0.16379589865097297), np.float64(0.16463736065004983), np.float64(0.1672579878158568), np.float64(0.1623955255075902), np.float64(0.1660920486357869), np.float64(0.16213613382634814), np.float64(0.16370438145423563), np.float64(0.16881409883195947), np.float64(0.1674540967118607), np.float64(0.1670610207170608), np.float64(0.1657844993501966), np.float64(0.16605236514120064), np.float64(0.1625727434299066), np.float64(0.16499728365788596), np.float64(0.16504222760684456), np.float64(0.16807781991826135), np.float64(0.16691935938150537), np.float64(0.1664195261055185), np.float64(0.16641514359477938), np.float64(0.165657721041209), np.float64(0.16332753693753735), np.float64(0.16642001460095548), np.float64(0.16374948397916644), np.float64(0.16752682832281265), np.float64(0.1654904397899408), np.float64(0.16821869844018314), np.float64(0.1698435498703792), np.float64(0.16832181955864756), np.float64(0.16560413731290352), np.float64(0.17036931556323612), np.float64(0.16795552270077974), np.float64(0.16323505228707708), np.float64(0.16480615084867342), np.float64(0.16354286346661773), np.float64(0.16815249132021964), np.float64(0.16869734073830536), np.float64(0.16410916843850729), np.float64(0.1611233593535381), np.float64(0.16680783115355524), np.float64(0.1622547355106483), np.float64(0.1605595760896932), np.float64(0.16516255475137206), np.float64(0.16213347641569417), np.float64(0.16458830693923088), np.float64(0.16081138868798583), np.float64(0.1619520223529375), np.float64(0.16530919492718762), np.float64(0.16798737450286333), np.float64(0.17001335386927144), np.float64(0.16581182853243975), np.float64(0.16479823080446107), np.float64(0.16917606629473894), np.float64(0.16600125678971078), np.float64(0.16260636216541957), np.float64(0.16369361164845606), np.float64(0.16241337795743996), np.float64(0.16376578797795968), np.float64(0.16262160878921864), np.float64(0.16017799641151784), np.float64(0.16497447881149588), np.float64(0.1642487204801815), np.float64(0.160498846883702), np.float64(0.15949526923567142), np.float64(0.16301018627686834), np.float64(0.16541647692250588), np.float64(0.16189785373346754), np.float64(0.16733048781031387), np.float64(0.16468667576094345), np.float64(0.16374029620969813), np.float64(0.1610179124626444), np.float64(0.16498910444305392), np.float64(0.16783815888152967), np.float64(0.158330241953929), np.float64(0.1552863330802458), np.float64(0.16539334486438564), np.float64(0.16683658909555485), np.float64(0.16512667882549645), np.float64(0.16703862863048735), np.float64(0.16483631431211176), np.float64(0.16898054825117528), np.float64(0.16464582068703026), np.float64(0.15946964513624295), np.float64(0.1670864178309989), np.float64(0.16205202543609507), np.float64(0.16881621221247176), np.float64(0.16871966952418294), np.float64(0.17012281841702803), np.float64(0.1661535359151957), np.float64(0.17147343758347922), np.float64(0.16449957936990658), np.float64(0.1683310315090473), np.float64(0.16868761893420567), np.float64(0.16741200294233596), np.float64(0.16952794556214892), np.float64(0.17023838366802058), np.float64(0.16863340185628167), np.float64(0.16488227117852233), np.float64(0.17005176693793833), np.float64(0.17035130208833818), np.float64(0.1704594033288367), np.float64(0.1671342159517035), np.float64(0.17309013087849265), np.float64(0.1716386897173879), np.float64(0.17145558288023738), np.float64(0.16860032348853568), np.float64(0.1705069161035505), np.float64(0.16777782230781596), np.float64(0.16876575184773718), np.float64(0.16869134909005715), np.float64(0.16835637210076934), np.float64(0.16744269522402058), np.float64(0.16538396813692055), np.float64(0.167081991647447), np.float64(0.168634494396298), np.float64(0.1679163216823201), np.float64(0.17124337282752347), np.float64(0.1698578204079044), np.float64(0.1728859132377206), np.float64(0.173192929458959), np.float64(0.16967005361780285), np.float64(0.1725207663289693), np.float64(0.17089704930788607), np.float64(0.17498544874520106), np.float64(0.173288871115212), np.float64(0.172493587937718), np.float64(0.1729949281572753), np.float64(0.1757324564866902), np.float64(0.1740361644455777), np.float64(0.17170275735499665), np.float64(0.1700795741871795), np.float64(0.17116113579963524), np.float64(0.17183180377618323), np.float64(0.17157397005963512), np.float64(0.17101405071683753), np.float64(0.17251948580567195), np.float64(0.173424452000098), np.float64(0.17244292209975332), np.float64(0.17625143945718041), np.float64(0.17876976185058105), np.float64(0.1762965292156413), np.float64(0.17928311623143228), np.float64(0.17547971936594503), np.float64(0.1753384621884938), np.float64(0.17599594216409217), np.float64(0.17587676831665103), np.float64(0.17461149929166506), np.float64(0.17784041948330698), np.float64(0.17541688863352936), np.float64(0.17691088853617162), np.float64(0.1720048438693716), np.float64(0.17337638885711854), np.float64(0.17445214024087186), np.float64(0.17210480640286344), np.float64(0.17295830863314476), np.float64(0.17571688215774864), np.float64(0.17752478461259455), np.float64(0.18538327130489538), np.float64(0.18382735240951575), np.float64(0.17954092781939224), np.float64(0.1791875898571904), np.float64(0.18245332680302845), np.float64(0.18350726194953274), np.float64(0.18097075577672855), np.float64(0.17752757911218903), np.float64(0.17984914954013312), np.float64(0.17212522750555603), np.float64(0.18075483950361831), np.float64(0.17886472527298092), np.float64(0.1817683980184944), np.float64(0.18239400344679893), np.float64(0.1825826993526964), np.float64(0.18031565117597925), np.float64(0.17226470839794894), np.float64(0.178804879427618), np.float64(0.17367157441626274), np.float64(0.17854865452998553), np.float64(0.1783614098506519), np.float64(0.1769678173593759), np.float64(0.17066463694712072), np.float64(0.18295990182338284), np.float64(0.18027480909594612), np.float64(0.1731503718738206), np.float64(0.17602567997630103), np.float64(0.17360584964820175), np.float64(0.17404016548045703), np.float64(0.17477886968986037), np.float64(0.1761413173030024), np.float64(0.17504580433413314), np.float64(0.17296421601940445), np.float64(0.17916863556787885), np.float64(0.17265975540684636), np.float64(0.17478008413457383), np.float64(0.1710440673241801), np.float64(0.17392787328688322), np.float64(0.17937382570124652), np.float64(0.18112376678190586), np.float64(0.18114128330949858), np.float64(0.17639437089324375), np.float64(0.17351316202894115), np.float64(0.17866870036859506), np.float64(0.17224506974632103), np.float64(0.17179265741639757), np.float64(0.17642867136013743), np.float64(0.17987436363133424), np.float64(0.1799906987729523), np.float64(0.18195840993544624), np.float64(0.1824243192870918), np.float64(0.18103216326075147), np.float64(0.1739111347910937), np.float64(0.17883533110911687), np.float64(0.1784141714644718), np.float64(0.17618126393113714), np.float64(0.17811002673535994), np.float64(0.18061838819017198), np.float64(0.18530384484024665), np.float64(0.17945424818119365), np.float64(0.17668909902329186), np.float64(0.18598445560136628), np.float64(0.17926768952963998), np.float64(0.18187263334476397), np.float64(0.17897008200852774), np.float64(0.17531625043759308), np.float64(0.18206380200895875), np.float64(0.18114891978519718), np.float64(0.18119609326696315), np.float64(0.18049014111285658), np.float64(0.178294755178528), np.float64(0.174517118245863), np.float64(0.17720542761237268), np.float64(0.17903834962424553), np.float64(0.17710267068074675), np.float64(0.17386724204979545), np.float64(0.18287584272675347), np.float64(0.18220276961837317), np.float64(0.18136839525362547), np.float64(0.17380047983864425), np.float64(0.1826972436877205), np.float64(0.17747582632855097), np.float64(0.1763929595545619), np.float64(0.17734256388827305), np.float64(0.18006728362373833), np.float64(0.1868761948534041), np.float64(0.17882018918933879), np.float64(0.18090309132824633), np.float64(0.1768689076587599), np.float64(0.17537371341215577), np.float64(0.17800318377091523), np.float64(0.1828847816326847), np.float64(0.1761457784604974), np.float64(0.17966173936618257), np.float64(0.18073823128428787), np.float64(0.18581824067385194), np.float64(0.18403309834468648), np.float64(0.1806217610693613), np.float64(0.17802995546000713), np.float64(0.17846391050441318), np.float64(0.1788287097082392), np.float64(0.18408902025722554), np.float64(0.17940745107182077), np.float64(0.1838223601731844), np.float64(0.18026963257379675), np.float64(0.18063795751727418), np.float64(0.18199101417354188), np.float64(0.183817797556645), np.float64(0.18101045417280817), np.float64(0.17833298320034036), np.float64(0.1785160287802399), np.float64(0.18080189709176506), np.float64(0.18420768700088602), np.float64(0.1784217461624768), np.float64(0.18372183309167128), np.float64(0.18227854533584584), np.float64(0.17837475785839574), np.float64(0.1852831637031948), np.float64(0.1826666673901831), np.float64(0.18201639846615605), np.float64(0.1826519548198789), np.float64(0.17045468760807103), np.float64(0.18507719668605208), np.float64(0.18544150019018865), np.float64(0.1825999145173337), np.float64(0.18047796402250302), np.float64(0.17648544636589458), np.float64(0.1753080915728271), np.float64(0.17671973676310054), np.float64(0.17448981791388646), np.float64(0.18042718569773167), np.float64(0.18091225610788536), np.float64(0.18309548667523587), np.float64(0.17687049935076812), np.float64(0.1788953650589065), np.float64(0.18422148643486635), np.float64(0.1874921369414597), np.float64(0.1867332421664609), np.float64(0.1873271128324152), np.float64(0.18379188253528778), np.float64(0.18121582285818794), np.float64(0.17996398138778397), np.float64(0.18184030382624972), np.float64(0.18362372838719462), np.float64(0.17913117209550153), np.float64(0.18599437817494832), np.float64(0.18389628059762908), np.float64(0.1869716514471546), np.float64(0.1860980597981979), np.float64(0.18122040687577595), np.float64(0.18535797243715943), np.float64(0.18254748662806541), np.float64(0.1804453422636674), np.float64(0.1841922495738871), np.float64(0.18445680482973598), np.float64(0.1884139230344429), np.float64(0.18774713824957281), np.float64(0.18173919556733728), np.float64(0.18152434451193633), np.float64(0.18489721767737674), np.float64(0.1821532691702572), np.float64(0.18222614629404316), np.float64(0.18358274417410664), np.float64(0.18184249849008474), np.float64(0.1818143956615841), np.float64(0.17997233679278896), np.float64(0.18076733975562), np.float64(0.18717527968233996), np.float64(0.184102157213504), np.float64(0.18579057737475402), np.float64(0.18334959118436545), np.float64(0.18270832298869882), np.float64(0.1794464489071978), np.float64(0.18286464443684738), np.float64(0.18495940702874736), np.float64(0.18482051944822647), np.float64(0.1778989743297697), np.float64(0.18140117851446527), np.float64(0.17974743813242544), np.float64(0.18441940785857863), np.float64(0.17869083233781843), np.float64(0.17367137717494807), np.float64(0.1752866948302499), np.float64(0.18069005713968075), np.float64(0.17852024091658483), np.float64(0.17407810840156357), np.float64(0.1730635717095255), np.float64(0.17800674057992458), np.float64(0.18166227532172574), np.float64(0.1772325022324104), np.float64(0.1800228642998274), np.float64(0.18023020836746104), np.float64(0.18018087021106424), np.float64(0.17044540482658344), np.float64(0.18089917779502254), np.float64(0.17596596634160563), np.float64(0.1777377990589133), np.float64(0.1812533999771456), np.float64(0.18347343425449922), np.float64(0.1858562931365905), np.float64(0.18123790356682462), np.float64(0.184850090458398), np.float64(0.17773741366924314), np.float64(0.1758089884503589), np.float64(0.1806668768676324), np.float64(0.1854403252656784), np.float64(0.17898015293135344), np.float64(0.1863403914165848), np.float64(0.18593323208809812), np.float64(0.18652547652856594), np.float64(0.18287018953734968), np.float64(0.1830564771215488), np.float64(0.1811856164785342), np.float64(0.18610519545559545), np.float64(0.18158487643782145), np.float64(0.1801327736425083), np.float64(0.1823235870118892), np.float64(0.18835114927944263), np.float64(0.18601781026496786), np.float64(0.18744626995672087), np.float64(0.1877221916667723), np.float64(0.1862395654787401), np.float64(0.19637027654585906), np.float64(0.19647982004566153), np.float64(0.197607273516742), np.float64(0.2009272712704518), np.float64(0.19356686502019846), np.float64(0.19644183839217436), np.float64(0.19265514999503014), np.float64(0.19535397844557248), np.float64(0.1921049915736198), np.float64(0.1948912492878142), np.float64(0.19138395232403643), np.float64(0.20146932830650252), np.float64(0.19336998840221772), np.float64(0.19833811795022388), np.float64(0.19469162747288457), np.float64(0.19881052069840696), np.float64(0.19701739198463192), np.float64(0.19383251992859793), np.float64(0.19714585265194393), np.float64(0.19743068698731475), np.float64(0.1958292538673387), np.float64(0.19879052900555175), np.float64(0.20275434955645819), np.float64(0.19520769834263754), np.float64(0.19549047863860647), np.float64(0.2020778428429097), np.float64(0.20151826092494773), np.float64(0.20141702332458186), np.float64(0.20240385931408222), np.float64(0.19618963634208803), np.float64(0.20242686954860886), np.float64(0.20442165747404395), np.float64(0.1970856579118908), np.float64(0.19105010703360406), np.float64(0.19715044418663233), np.float64(0.19914010273750068), np.float64(0.2014031081899638), np.float64(0.20370889805255443), np.float64(0.1962548799972835), np.float64(0.19912300544337969), np.float64(0.2045683918956789), np.float64(0.20122676505324077), np.float64(0.20482484140000934), np.float64(0.19786286946761333), np.float64(0.19587375888320907), np.float64(0.20118107856043724), np.float64(0.20314491966536544), np.float64(0.19823743441158867), np.float64(0.20288668037027427), np.float64(0.19817212598251155), np.float64(0.1982872132108318), np.float64(0.20027824928069152), np.float64(0.19613664816138707), np.float64(0.19459465679961738), np.float64(0.20183151899163637), np.float64(0.20092201806058418), np.float64(0.20102004168541382), np.float64(0.19596849459587087), np.float64(0.1998177391010181), np.float64(0.19830417857173221), np.float64(0.2025728656570715), np.float64(0.19847797346084325), np.float64(0.19913742398337234), np.float64(0.19975020808013122), np.float64(0.20323589675327136), np.float64(0.2054769013455088), np.float64(0.2034161359511636), np.float64(0.20079332363627928), np.float64(0.20451907484905013), np.float64(0.19595004074749892), np.float64(0.2053113184112256), np.float64(0.19036200927715433), np.float64(0.18439581332496663), np.float64(0.18044450550061947), np.float64(0.1833057586120229), np.float64(0.1860944866263585), np.float64(0.18251199433023468), np.float64(0.18763730640703574), np.float64(0.18410633451494543), np.float64(0.19027772989177563), np.float64(0.18504025696583037), np.float64(0.1754411660597713), np.float64(0.18349027873183385), np.float64(0.17675083780403522), np.float64(0.18627408330820971), np.float64(0.18994687397402873), np.float64(0.18322048609804642), np.float64(0.18954292282988863), np.float64(0.1933796881143851), np.float64(0.19283704503574825), np.float64(0.196647518833854), np.float64(0.19618712740055164), np.float64(0.19115956645527377), np.float64(0.18884778434913024), np.float64(0.18968693940728526), np.float64(0.18982008898667438), np.float64(0.18886211051545856), np.float64(0.19597337778918517), np.float64(0.1909702716258728), np.float64(0.1973479425722888), np.float64(0.18949737126383073), np.float64(0.19035907627391674), np.float64(0.19976766553849926), np.float64(0.1933881411260389), np.float64(0.19369666451641065), np.float64(0.19548927357541113), np.float64(0.19560745886766528), np.float64(0.18898940087414792), np.float64(0.19516870015262866), np.float64(0.18880760515804998), np.float64(0.19244917747011137), np.float64(0.18809930209080947), np.float64(0.1976757397513844), np.float64(0.19898080987626912), np.float64(0.18628072206382534), np.float64(0.19492565320809996), np.float64(0.1936923085440459), np.float64(0.18869843922736346), np.float64(0.1936710157308606), np.float64(0.191902001753329), np.float64(0.19401699674018816), np.float64(0.19538188592689046), np.float64(0.19514149096238076), np.float64(0.18776557859208307), np.float64(0.19473804178614681), np.float64(0.18809509897678045), np.float64(0.19014113616354975), np.float64(0.18791541740888015), np.float64(0.1883998365839686), np.float64(0.1800205703205025), np.float64(0.18645209222554326), np.float64(0.19387295447056258), np.float64(0.1887260853458263), np.float64(0.18858212157535087), np.float64(0.1869619001640386), np.float64(0.18822767050257597), np.float64(0.19003283393039097), np.float64(0.18823239741707734), np.float64(0.18620101289237728), np.float64(0.18517245148257994), np.float64(0.19058568999223677), np.float64(0.19682294370321876), np.float64(0.19269651885109748), np.float64(0.18856287583407394), np.float64(0.19826806754500165), np.float64(0.1921052108898228)]
        list_of_n_samples = [104, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 103, 102, 101, 101, 100, 99, 98, 97, 97, 97, 97, 97, 96, 96, 96, 96, 95, 94, 94, 93, 92, 92, 92, 92, 91, 90, 90, 88, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 86, 86, 86, 86, 86, 86, 86, 86, 86, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 82, 82, 82, 82, 82, 82, 82, 82, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 80, 80, 80, 80, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 72, 72, 72, 72, 72, 72, 72, 72, 72, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 67, 67, 67, 67, 67, 67, 67, 67, 67, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 61, 61, 61, 61, 61, 61, 61, 61, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 58, 58, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 55, 55, 55, 55, 55]
        list_of_rms_deviations = [np.float64(0.3794729460181765), np.float64(0.22452337436665937), np.float64(0.2773698589594881), np.float64(0.3119685979501878), np.float64(0.35341659049452584), np.float64(0.3638586339216531), np.float64(0.36061118388686014), np.float64(0.38955285160768743), np.float64(0.4159475047195915), np.float64(0.42430876143195945), np.float64(0.4245838939825604), np.float64(0.4290769587949764), np.float64(0.43007674047996625), np.float64(0.4386463226179285), np.float64(0.44991002353169285), np.float64(0.45280872428846736), np.float64(0.44668443050019696), np.float64(0.46663005113634004), np.float64(0.45286424956722654), np.float64(0.4869334134228193), np.float64(0.4695959252955719), np.float64(0.4968742565903053), np.float64(0.4824634110464886), np.float64(0.49231414274420177), np.float64(0.5101383098357531), np.float64(0.5095262843420443), np.float64(0.48931970001309), np.float64(0.5055452920796337), np.float64(0.5023162449696429), np.float64(0.5215072318304611), np.float64(0.5434452551840345), np.float64(0.5245036665238131), np.float64(0.5334873513087781), np.float64(0.5230354514564398), np.float64(0.5275526330554895), np.float64(0.5498901106490641), np.float64(0.5374855051948191), np.float64(0.5429952305912258), np.float64(0.5323367706076476), np.float64(0.5297468974108644), np.float64(0.5316227654005585), np.float64(0.5380355548624559), np.float64(0.516058415703175), np.float64(0.5331887390604878), np.float64(0.5389194104754754), np.float64(0.539455640635094), np.float64(0.557464890276127), np.float64(0.5276408234237556), np.float64(0.5531731527106353), np.float64(0.5458235727181654), np.float64(0.5505744509959306), np.float64(0.5620081822976484), np.float64(0.5578996129074788), np.float64(0.5596039563163365), np.float64(0.5563649738482783), np.float64(0.5645295096154678), np.float64(0.569317436703068), np.float64(0.564763915393081), np.float64(0.5706843507226121), np.float64(0.571160205221099), np.float64(0.568009973566387), np.float64(0.5445528266294489), np.float64(0.56118121538408), np.float64(0.5579031089465906), np.float64(0.545941938334398), np.float64(0.5636036025813118), np.float64(0.565516132263033), np.float64(0.5676776542049007), np.float64(0.583600934503148), np.float64(0.5783899745139051), np.float64(0.5833916408187696), np.float64(0.5777901929553371), np.float64(0.5671268497385838), np.float64(0.5531260684169511), np.float64(0.5598666600059491), np.float64(0.5606312014776446), np.float64(0.5850987389565024), np.float64(0.5659867233598989), np.float64(0.5662521949870379), np.float64(0.5851543201109128), np.float64(0.5589523311356206), np.float64(0.5628095067237989), np.float64(0.5801189603784186), np.float64(0.5845212936323507), np.float64(0.5688472934709492), np.float64(0.586994061257666), np.float64(0.5784953781703785), np.float64(0.5833678195681777), np.float64(0.6085104431776536), np.float64(0.5822041297436216), np.float64(0.6109619656142965), np.float64(0.5899246751682454), np.float64(0.5845144485770504), np.float64(0.5960404121675087), np.float64(0.5832079390499735), np.float64(0.5785173719787275), np.float64(0.5788444947789005), np.float64(0.5758205445223368), np.float64(0.5841604440496909), np.float64(0.5740572377349711), np.float64(0.572750617713132), np.float64(0.5723365044544715), np.float64(0.5827429773232384), np.float64(0.5691824156530843), np.float64(0.5749628534886114), np.float64(0.5838392735281862), np.float64(0.5736913891917054), np.float64(0.575867930414599), np.float64(0.596245404641963), np.float64(0.5789518582236428), np.float64(0.5715451806246018), np.float64(0.5688197495843921), np.float64(0.5786559011286361), np.float64(0.5670733684237216), np.float64(0.5740596250124131), np.float64(0.5875831837925847), np.float64(0.5998141770573285), np.float64(0.5932665720314939), np.float64(0.5905738529489579), np.float64(0.6031485200168178), np.float64(0.5882618117563323), np.float64(0.5878054319869109), np.float64(0.5858739015783161), np.float64(0.5972995612331025), np.float64(0.5942246269261185), np.float64(0.6017024528621612), np.float64(0.5898211842890008), np.float64(0.6039495509879114), np.float64(0.5888261817922941), np.float64(0.6085984732544962), np.float64(0.5943892336656482), np.float64(0.6004253859543474), np.float64(0.6075796535353047), np.float64(0.592594113835194), np.float64(0.6028081251745911), np.float64(0.6140247302908517), np.float64(0.6237202280744969), np.float64(0.602039335398155), np.float64(0.5963448153866785), np.float64(0.597188227549681), np.float64(0.6036302015766906), np.float64(0.6144075591747088), np.float64(0.6144153794607702), np.float64(0.5928274900078877), np.float64(0.6220468313891513), np.float64(0.610295753778503), np.float64(0.613974571419273), np.float64(0.6067635208303636), np.float64(0.6156419180246676), np.float64(0.6037835785371279), np.float64(0.5925696003884229), np.float64(0.6166216962650453), np.float64(0.5986849758562274), np.float64(0.612284352636475), np.float64(0.6150489950389366), np.float64(0.615682203925674), np.float64(0.602675097524373), np.float64(0.611868625537807), np.float64(0.6104000580072605), np.float64(0.6085158477483943), np.float64(0.60186565921675), np.float64(0.6011461308499342), np.float64(0.6093686613380251), np.float64(0.6205467196785276), np.float64(0.6245041479576827), np.float64(0.6181764212239053), np.float64(0.6253033373465067), np.float64(0.6006743658220373), np.float64(0.6087942571363048), np.float64(0.6157413751725547), np.float64(0.6016971946969669), np.float64(0.6074564680023654), np.float64(0.5875694116484348), np.float64(0.621168559841489), np.float64(0.5989791594809576), np.float64(0.6105421399663435), np.float64(0.616758599754433), np.float64(0.6150517859413615), np.float64(0.6120338599860922), np.float64(0.6218467128717703), np.float64(0.6128729448179551), np.float64(0.6062193269105857), np.float64(0.6103575928871734), np.float64(0.6343894337330901), np.float64(0.6109942553350743), np.float64(0.6075138996955762), np.float64(0.6134065175085562), np.float64(0.6003688555584561), np.float64(0.6069644975225877), np.float64(0.6063972242515214), np.float64(0.5991513942118378), np.float64(0.607184191439249), np.float64(0.6059735071208575), np.float64(0.5882044891834411), np.float64(0.590163612573065), np.float64(0.615094391420382), np.float64(0.6089996119100539), np.float64(0.6219673136881512), np.float64(0.6107075617184838), np.float64(0.6319140161439156), np.float64(0.6175928457072953), np.float64(0.6037802498904116), np.float64(0.6001234111836163), np.float64(0.5957553720255989), np.float64(0.5913187888121999), np.float64(0.6055319532680166), np.float64(0.6220907796764105), np.float64(0.6100143654242819), np.float64(0.6126800924934936), np.float64(0.6131644633923075), np.float64(0.618137508166304), np.float64(0.5930241470998251), np.float64(0.6073134768429326), np.float64(0.6061003138831349), np.float64(0.6154838534359078), np.float64(0.6033646621935275), np.float64(0.6130315227844885), np.float64(0.6219289839566635), np.float64(0.5994694689061166), np.float64(0.610280786964421), np.float64(0.6153971414008185), np.float64(0.6155810759639081), np.float64(0.6015021574297247), np.float64(0.5947181406774117), np.float64(0.5979847204068838), np.float64(0.6039726274013288), np.float64(0.6082003881376572), np.float64(0.6049651143494911), np.float64(0.6091376336152676), np.float64(0.5873211227542793), np.float64(0.6029979687254099), np.float64(0.6145630540659355), np.float64(0.5933731807705754), np.float64(0.5886819109811956), np.float64(0.5778279059578935), np.float64(0.594121451568395), np.float64(0.6117708131660632), np.float64(0.5944509929777148), np.float64(0.5974997512954253), np.float64(0.5854146970072901), np.float64(0.5882517357605106), np.float64(0.5740965948644541), np.float64(0.5974425510991128), np.float64(0.6012004751998137), np.float64(0.6122424543766054), np.float64(0.6085322607170003), np.float64(0.6104618125528863), np.float64(0.60295208165183), np.float64(0.6047316603942914), np.float64(0.6119634399686926), np.float64(0.6040644350941198), np.float64(0.6041071078641937), np.float64(0.6116721676901089), np.float64(0.6183993297322864), np.float64(0.6064196403826352), np.float64(0.5955172430737816), np.float64(0.601143656217194), np.float64(0.6047247772159887), np.float64(0.5967057941759649), np.float64(0.5976243050615729), np.float64(0.5921402510946352), np.float64(0.6226643769537126), np.float64(0.5959658343292074), np.float64(0.600740540496677), np.float64(0.6062952737051599), np.float64(0.6033563973030124), np.float64(0.6066522830348179), np.float64(0.5855312151069827), np.float64(0.6010492153765865), np.float64(0.5919715767806564), np.float64(0.6164435663658123), np.float64(0.6124319603821883), np.float64(0.6128851862013108), np.float64(0.5970959167895092), np.float64(0.61940574766854), np.float64(0.6242231631187287), np.float64(0.6037685874730913), np.float64(0.6180655335997284), np.float64(0.6099840297522552), np.float64(0.6077107513374678), np.float64(0.5996593399275816), np.float64(0.6180844944662084), np.float64(0.6233521344628585), np.float64(0.6118076439984522), np.float64(0.6079403564774337), np.float64(0.6026127018254805), np.float64(0.615483414186035), np.float64(0.6041976071768997), np.float64(0.6251587028363297), np.float64(0.608075331710402), np.float64(0.627344289280692), np.float64(0.6159871658858774), np.float64(0.5881440168281905), np.float64(0.6047998888350495), np.float64(0.6155087238191731), np.float64(0.6099797214305985), np.float64(0.6168721423925463), np.float64(0.616803095476515), np.float64(0.6139328432097899), np.float64(0.6023831399151175), np.float64(0.6062882083000948), np.float64(0.6234332767092394), np.float64(0.6255912394881048), np.float64(0.6172865264014068), np.float64(0.6234369101722965), np.float64(0.6227352361625066), np.float64(0.6265266271779527), np.float64(0.6318483795698192), np.float64(0.6340891616392605), np.float64(0.6362139304072433), np.float64(0.6202303156632436), np.float64(0.6262087369906358), np.float64(0.6391025738881974), np.float64(0.6326781981941405), np.float64(0.6264298645828659), np.float64(0.6334156828050602), np.float64(0.6278831882494147), np.float64(0.6098243581657472), np.float64(0.6244663286750386), np.float64(0.6249704425556443), np.float64(0.6373017518636027), np.float64(0.6250484203740264), np.float64(0.6332866442018692), np.float64(0.639312431616512), np.float64(0.6369332471253119), np.float64(0.6189807457211517), np.float64(0.636512795680363), np.float64(0.6087429819316778), np.float64(0.6188454421564026), np.float64(0.6212385411202456), np.float64(0.6366249371445513), np.float64(0.6131944426572835), np.float64(0.6139388783533327), np.float64(0.6149375667516538), np.float64(0.6028600156163149), np.float64(0.6093192344083183), np.float64(0.6186294113775396), np.float64(0.617561441015361), np.float64(0.6188638760486423), np.float64(0.6289128828904988), np.float64(0.6195349917807345), np.float64(0.6232537854613113), np.float64(0.6191823260541095), np.float64(0.6162037307520419), np.float64(0.6212579164292327), np.float64(0.618278064442986), np.float64(0.617360363846587), np.float64(0.6214876532078034), np.float64(0.6312019883126193), np.float64(0.6069046119322151), np.float64(0.6263891089147053), np.float64(0.6190182047548712), np.float64(0.6132818254330618), np.float64(0.6179302481548516), np.float64(0.6320714578105987), np.float64(0.6065883370511649), np.float64(0.6185357888575879), np.float64(0.609538840299325), np.float64(0.6029268419330637), np.float64(0.632089169524852), np.float64(0.6204461520297132), np.float64(0.6268185884781747), np.float64(0.6257692510103814), np.float64(0.639067977180861), np.float64(0.6152893778560091), np.float64(0.6284511370208263), np.float64(0.6148632188724434), np.float64(0.6097042164008267), np.float64(0.6146903383160434), np.float64(0.6131050270034303), np.float64(0.6218827561445448), np.float64(0.5924178217117919), np.float64(0.6281802579793998), np.float64(0.625443319623148), np.float64(0.6262329012288299), np.float64(0.6229288912151887), np.float64(0.6291807523623143), np.float64(0.6148247560288529), np.float64(0.6230579499625459), np.float64(0.6359066584424077), np.float64(0.6224219661963242), np.float64(0.6254547859523372), np.float64(0.6344332458930961), np.float64(0.617373700714089), np.float64(0.6136896072920871), np.float64(0.6237245003355955), np.float64(0.6310335853790366), np.float64(0.6225205575890914), np.float64(0.6180370541754959), np.float64(0.6270012884357079), np.float64(0.6372453865652801), np.float64(0.6357360854962106), np.float64(0.6286106212663071), np.float64(0.6270487362413932), np.float64(0.615653383999528), np.float64(0.6127776923223128), np.float64(0.6079234325323685), np.float64(0.6157034967077281), np.float64(0.6253029747833485), np.float64(0.6388482848572506), np.float64(0.6213397859141344), np.float64(0.6319389081629236), np.float64(0.6230772836149474), np.float64(0.6253171660726499), np.float64(0.6317179109111046), np.float64(0.6231283684704149), np.float64(0.6214726220942297), np.float64(0.628287824000238), np.float64(0.6204591614981819), np.float64(0.6185932111657445), np.float64(0.6195489265499685), np.float64(0.6335816485527968), np.float64(0.6259127603462312), np.float64(0.6398733998488202), np.float64(0.6474776486066877), np.float64(0.6257646139403205), np.float64(0.6138317360480023), np.float64(0.6178630964500306), np.float64(0.6374025446624642), np.float64(0.6376459740682259), np.float64(0.6341320304455008), np.float64(0.6494176704913216), np.float64(0.640134184260316), np.float64(0.6553099000860052), np.float64(0.6503002330757915), np.float64(0.6427030004760111), np.float64(0.6293941273792836), np.float64(0.6441311078310442), np.float64(0.6441092504309144), np.float64(0.6528539119037766), np.float64(0.654864798207816), np.float64(0.6354537436626759), np.float64(0.6420278969262219), np.float64(0.645072683353487), np.float64(0.6532471814349158), np.float64(0.6256592243274444), np.float64(0.647257296405986), np.float64(0.6144208551433219), np.float64(0.6547313516823202), np.float64(0.6527765765214755), np.float64(0.6428241416369245), np.float64(0.6437287415395739), np.float64(0.6462790546664506), np.float64(0.640454455390067), np.float64(0.6440852377342916), np.float64(0.630839315068414), np.float64(0.6468451428035554), np.float64(0.6332444848003073), np.float64(0.6198683603562979), np.float64(0.6279778482050566), np.float64(0.6282067919680888), np.float64(0.6292358026209092), np.float64(0.6281408169508719), np.float64(0.6487467144122199), np.float64(0.6236032745892749), np.float64(0.6036624503270752), np.float64(0.618488134022637), np.float64(0.6247546700363057), np.float64(0.6301906164677509), np.float64(0.6227056274169485), np.float64(0.6154533090380442), np.float64(0.6274628827741519), np.float64(0.6261276246486969), np.float64(0.6145766018912368), np.float64(0.6186727596533784), np.float64(0.623786278651935), np.float64(0.6445097308525057), np.float64(0.6293169237922814), np.float64(0.6256521619508816), np.float64(0.6362497767614669), np.float64(0.6319653819142517), np.float64(0.6162108011923668), np.float64(0.633642621090924), np.float64(0.6202421514045765), np.float64(0.6383323688372322), np.float64(0.6432126809461662), np.float64(0.6329178586933933), np.float64(0.6454551477940403), np.float64(0.6428069490631737), np.float64(0.6455259932385775), np.float64(0.6398068155519071), np.float64(0.6302051008499802), np.float64(0.6418229103710729), np.float64(0.6388786966188565), np.float64(0.6623043241646174), np.float64(0.647827579631086), np.float64(0.6506126339899647), np.float64(0.6229757477668826), np.float64(0.6344353629551089), np.float64(0.6259826218980862), np.float64(0.6266947825253409), np.float64(0.6399108467652046), np.float64(0.6389596456800432), np.float64(0.6363819895163858), np.float64(0.6205677897615517), np.float64(0.649648601138073), np.float64(0.624144794971213), np.float64(0.6349109214257175), np.float64(0.639421988418749), np.float64(0.645679613965186), np.float64(0.6436876403855198), np.float64(0.6441848413787912), np.float64(0.6306365888143775), np.float64(0.6418904045050565), np.float64(0.621679666727686), np.float64(0.6231263824039006), np.float64(0.6340299441551512), np.float64(0.6447773857824047), np.float64(0.6360557362855888), np.float64(0.6449702655519509), np.float64(0.6628359142160664), np.float64(0.6441857188777506), np.float64(0.6392230976860476), np.float64(0.6337428513708904), np.float64(0.6446067361972571), np.float64(0.6223755268317094), np.float64(0.6312884496035562), np.float64(0.6384781060650525), np.float64(0.6508632714666432), np.float64(0.635346254359125), np.float64(0.6274910112742532), np.float64(0.6254000871118315), np.float64(0.6368509157011237), np.float64(0.6446486881936169), np.float64(0.6598852497604482), np.float64(0.641870030010085), np.float64(0.6509129148040734), np.float64(0.6373532337787251), np.float64(0.6527520082869082), np.float64(0.6530031346498768), np.float64(0.6636152979324448), np.float64(0.6429690973013011), np.float64(0.6315117643852194), np.float64(0.6450828871069114), np.float64(0.6426387809803589), np.float64(0.6420855194622591), np.float64(0.6455164487524782), np.float64(0.6582203525495307), np.float64(0.6405798366606613), np.float64(0.6198317126420834), np.float64(0.6336697507842635), np.float64(0.6340225321207453), np.float64(0.6341326881841203), np.float64(0.625609656422181), np.float64(0.6523365625020773), np.float64(0.6507500413071263), np.float64(0.6457240503622266), np.float64(0.6446039214905505), np.float64(0.6423680001500047), np.float64(0.6347831177278617), np.float64(0.6470206388280234), np.float64(0.6637754496105224), np.float64(0.6500832402359286), np.float64(0.6517463042210618), np.float64(0.6605773477164846), np.float64(0.6415177517312929), np.float64(0.6451700896073467), np.float64(0.6484482572422763), np.float64(0.6241824116255204), np.float64(0.6597093979175253), np.float64(0.6412016164771173), np.float64(0.6333723203624597), np.float64(0.6426207088277663), np.float64(0.6500684135490278), np.float64(0.6637275920967394), np.float64(0.654438708654254), np.float64(0.6730931932179217), np.float64(0.639134974236572), np.float64(0.6394749921383102), np.float64(0.6340389343570713), np.float64(0.6281011786798778), np.float64(0.6358654312131632), np.float64(0.6574298087617796), np.float64(0.6476584813246719), np.float64(0.6511085758434364), np.float64(0.6338904205922012), np.float64(0.6441127964494959), np.float64(0.6437549880513596), np.float64(0.6251345191082572), np.float64(0.6693501394240805), np.float64(0.6613499156494975), np.float64(0.6590053431441624), np.float64(0.6662380288158882), np.float64(0.6478893480120578), np.float64(0.6521256170692702), np.float64(0.642384540034372), np.float64(0.6752817419629343), np.float64(0.6866058834368741), np.float64(0.6726650453541797), np.float64(0.665709937131707), np.float64(0.6652248085029203), np.float64(0.6568390869955693), np.float64(0.6597072802528345), np.float64(0.6584938228931828), np.float64(0.6337201000813927), np.float64(0.6572444919877929), np.float64(0.6474623582960874), np.float64(0.6510004206904026), np.float64(0.64608003820711), np.float64(0.6512261609488443), np.float64(0.6363181252875305), np.float64(0.6333231143099151), np.float64(0.6474593583235214), np.float64(0.633550670978812), np.float64(0.6425828777547752), np.float64(0.6520646000981738), np.float64(0.6537310955088066), np.float64(0.6469738300489264), np.float64(0.6329824684389003), np.float64(0.6467257232730373), np.float64(0.641740863980464), np.float64(0.6533968109559657), np.float64(0.6338625016694552), np.float64(0.6420694302788997), np.float64(0.6278048320695122), np.float64(0.636081059032542), np.float64(0.6337680756176063), np.float64(0.6270253102527567), np.float64(0.6384145086515265), np.float64(0.6444621684200569), np.float64(0.6451137390515701), np.float64(0.6301651742227737), np.float64(0.6601847703873417), np.float64(0.6247076693591532), np.float64(0.6268900445098969), np.float64(0.6384727697101137), np.float64(0.636414026376022), np.float64(0.6681481239213516), np.float64(0.6579238741179666), np.float64(0.668472059254871), np.float64(0.6554084687853222), np.float64(0.6519280944399636), np.float64(0.6458411239004929), np.float64(0.6419648639785746), np.float64(0.6545359726369437), np.float64(0.6470090523070542), np.float64(0.6676075712510331), np.float64(0.6536687098843253), np.float64(0.6704367708020583), np.float64(0.6347749664031029), np.float64(0.6414251673225146), np.float64(0.6374731067870287), np.float64(0.6508170624487679), np.float64(0.6430134391612538), np.float64(0.6530583094489146), np.float64(0.640213716304014), np.float64(0.6413263639824559), np.float64(0.6557666779768712), np.float64(0.6471144072648368), np.float64(0.6431179709589204), np.float64(0.6394578391434125), np.float64(0.6470305210329614), np.float64(0.6533900343371066), np.float64(0.6555196533041965), np.float64(0.6558182103395739), np.float64(0.6615593288245597), np.float64(0.6561938040724367), np.float64(0.6537983064571397), np.float64(0.6635076076193417), np.float64(0.6555141738939674), np.float64(0.6563855625733159), np.float64(0.6623674235714079), np.float64(0.6523773542721666), np.float64(0.652631671116098), np.float64(0.639330545702239), np.float64(0.645228793484061), np.float64(0.6691902434137291), np.float64(0.6420742727361282), np.float64(0.6627090034843296), np.float64(0.6737307846467611), np.float64(0.6585053236129665), np.float64(0.643330839118803), np.float64(0.6597747872654413), np.float64(0.6546539793567266), np.float64(0.6553582837916203), np.float64(0.6555199739621155), np.float64(0.6410346110514696), np.float64(0.6348713094313733), np.float64(0.6589917556421037), np.float64(0.6515489263930764), np.float64(0.636780264190844), np.float64(0.665464729985708), np.float64(0.6584449728276245), np.float64(0.6581083715933419), np.float64(0.6400154331836169), np.float64(0.6686965481752081), np.float64(0.6585391198753784), np.float64(0.6654357461718559), np.float64(0.6657644506627808), np.float64(0.6711436507253796), np.float64(0.658638942373423), np.float64(0.657636110665234), np.float64(0.657337535899364), np.float64(0.6518038231597174), np.float64(0.6587663837512444), np.float64(0.6550030746968778), np.float64(0.6598607364414568), np.float64(0.6398195125219562), np.float64(0.667319323081117), np.float64(0.6489190297900862), np.float64(0.6519247722525231), np.float64(0.6594119129033581), np.float64(0.64041161127374), np.float64(0.6341488704614956), np.float64(0.6509811632474133), np.float64(0.6493893222570783), np.float64(0.648850538397578), np.float64(0.6561776727216827), np.float64(0.6501798254947543), np.float64(0.675690716155075), np.float64(0.6533451230118662), np.float64(0.6427219754986094), np.float64(0.6560700200280187), np.float64(0.6510332506591767), np.float64(0.6351133536461636), np.float64(0.661938055368819), np.float64(0.6479709405090899), np.float64(0.6376362608499591), np.float64(0.6572697020942705), np.float64(0.6663827280996238), np.float64(0.6510875079594863), np.float64(0.6317082663376693), np.float64(0.6539849465811362), np.float64(0.641718879478967), np.float64(0.6468081195877724), np.float64(0.6574610631895126), np.float64(0.658409486799694), np.float64(0.6525258872474629), np.float64(0.6529408416666751), np.float64(0.672793508360513), np.float64(0.6395926889412096), np.float64(0.6542780189763404), np.float64(0.6534416614790731), np.float64(0.6294823911361455), np.float64(0.6638185429135629), np.float64(0.6677448802031627), np.float64(0.659488179976611), np.float64(0.6777383428950551), np.float64(0.6611545774316389), np.float64(0.6545205030390613), np.float64(0.6665558724787195), np.float64(0.6596206095431134), np.float64(0.6638744350784688), np.float64(0.6685394351411257), np.float64(0.6623711544712237), np.float64(0.6676669103859314), np.float64(0.6686135469309631), np.float64(0.6540858856347117), np.float64(0.6599376827174793), np.float64(0.6613497235120878), np.float64(0.6539946739389642), np.float64(0.6501964844730845), np.float64(0.6484381945523985), np.float64(0.6584526683700396), np.float64(0.6479639238652483), np.float64(0.6677203782225876), np.float64(0.6623825639507697), np.float64(0.6894920770700642), np.float64(0.6624911306821009), np.float64(0.6679512324689258), np.float64(0.6643881235319831), np.float64(0.6891061134719448), np.float64(0.6750204909269947), np.float64(0.6581614174586682), np.float64(0.6680628917956261), np.float64(0.6435489421176487), np.float64(0.6609301414798961), np.float64(0.6585917611025541), np.float64(0.6640032977020388), np.float64(0.6808989472399132), np.float64(0.6665477866478569), np.float64(0.6660699323558382), np.float64(0.6621939231780158), np.float64(0.6702132431823021), np.float64(0.6470512029022468), np.float64(0.6308199435741925), np.float64(0.6553411904500592), np.float64(0.6529167605647079), np.float64(0.6614789704208698), np.float64(0.6743613645475323), np.float64(0.649082123464098), np.float64(0.6399968980800808), np.float64(0.624590067863131), np.float64(0.6304201875825182), np.float64(0.6430298450866947), np.float64(0.6547316661797757), np.float64(0.6545662344644633), np.float64(0.6532669257965921), np.float64(0.647362092036882), np.float64(0.6435733610974114), np.float64(0.644386848007603), np.float64(0.6368163032981153), np.float64(0.6268178723439524), np.float64(0.6491625599639117), np.float64(0.6702992057122584), np.float64(0.6518692487850997), np.float64(0.6534855763569168), np.float64(0.6479781725156157), np.float64(0.6558819022432593), np.float64(0.6494487684866694), np.float64(0.6489328212828621), np.float64(0.6434722688084764), np.float64(0.6515300824329495), np.float64(0.6411137166770996), np.float64(0.6409558536682205), np.float64(0.6437923744038361), np.float64(0.6281659022856293), np.float64(0.6321068237324762), np.float64(0.6395104674795454), np.float64(0.627951064103931), np.float64(0.6438885509035042), np.float64(0.651650090528509), np.float64(0.6308776213891841), np.float64(0.6468235350671787), np.float64(0.6394736635683187), np.float64(0.6398711665612858), np.float64(0.6344406889670089), np.float64(0.6432897225458032), np.float64(0.6407099037947355), np.float64(0.6280261877918447), np.float64(0.6558059602886923), np.float64(0.6598090569760829), np.float64(0.6219108385809504), np.float64(0.6345454864911174), np.float64(0.6520310643954099), np.float64(0.6474152050782654), np.float64(0.6454502255579246), np.float64(0.638261443945772), np.float64(0.6429506832253282), np.float64(0.6514910548810012), np.float64(0.6583455867634654), np.float64(0.6406137179282313), np.float64(0.6545239114549076), np.float64(0.6334664167223586), np.float64(0.6417965814137695), np.float64(0.6662481397390811), np.float64(0.6513135416592115), np.float64(0.6442895660634936), np.float64(0.6573918311056062), np.float64(0.6678126570557181), np.float64(0.6582313851188246), np.float64(0.6515394957664805), np.float64(0.6438519442594037), np.float64(0.6721998523330587), np.float64(0.6702291679224611), np.float64(0.6566651431240467), np.float64(0.6494462374589242), np.float64(0.6513323328909785), np.float64(0.6696851311926337), np.float64(0.661895378087604), np.float64(0.6745944268249985), np.float64(0.6680946649808891), np.float64(0.6668997395783515), np.float64(0.638331507326867), np.float64(0.6509849326728693), np.float64(0.6452064324327609), np.float64(0.6643055927190427), np.float64(0.6557815004815255), np.float64(0.6430613244740195), np.float64(0.6408376180992988), np.float64(0.6409658037280647), np.float64(0.6328070912785557), np.float64(0.6573148233591979), np.float64(0.645287672424489), np.float64(0.6554385736052994), np.float64(0.6585563867638178), np.float64(0.651576939497651), np.float64(0.6595157848480911), np.float64(0.6581751229829615), np.float64(0.6612812593062164), np.float64(0.6628711904850939), np.float64(0.6556335924238509), np.float64(0.6550912519465049), np.float64(0.6779501955652601), np.float64(0.6611290169821298), np.float64(0.6880251470850475), np.float64(0.6677306644010766), np.float64(0.6761613203250734), np.float64(0.6664148739833954), np.float64(0.6666070764957995), np.float64(0.6516624259987563), np.float64(0.6877145312421102), np.float64(0.6674521369489251), np.float64(0.6600856383160575), np.float64(0.6495664392180925), np.float64(0.6613809063930997), np.float64(0.6835305932654671), np.float64(0.6892490290852205), np.float64(0.6712169736943915), np.float64(0.6749631461630424), np.float64(0.694687398598151), np.float64(0.6669729359785185), np.float64(0.6837825098008906), np.float64(0.6801458707622619), np.float64(0.665675215351145), np.float64(0.6662666654545001), np.float64(0.6606422958419521), np.float64(0.669004210404571), np.float64(0.6619506325422103), np.float64(0.6735513237326305), np.float64(0.6708235913944706), np.float64(0.6740112341382358), np.float64(0.6724084365321304), np.float64(0.6812773462883344), np.float64(0.6530361416816249), np.float64(0.677640761146175), np.float64(0.6631510680865401), np.float64(0.6419993700059313), np.float64(0.6752926024532001), np.float64(0.6587221718513772), np.float64(0.65301687886002), np.float64(0.6545006359562954), np.float64(0.6641190077280418), np.float64(0.6597446878031806), np.float64(0.6763816408734054), np.float64(0.6824154061489176), np.float64(0.6724187966284936), np.float64(0.6708100099897928), np.float64(0.6515121768376857), np.float64(0.6537649968989249), np.float64(0.6698357830681482), np.float64(0.6609310048576094), np.float64(0.6643263735955329), np.float64(0.6749008388469415), np.float64(0.6552803853099758), np.float64(0.670196164078142), np.float64(0.6542337114593209), np.float64(0.6605617291937496), np.float64(0.6811798846789917), np.float64(0.6756921623215297), np.float64(0.6741060547406991), np.float64(0.6689551802998014), np.float64(0.6700360473722762), np.float64(0.6559954691936108), np.float64(0.6657786921609987), np.float64(0.665960045219138), np.float64(0.6782089469715444), np.float64(0.6735344372240925), np.float64(0.6715175681806685), np.float64(0.6714998852975229), np.float64(0.6684436118763597), np.float64(0.6590411230579055), np.float64(0.6715195495031391), np.float64(0.6607437164066127), np.float64(0.6759856305578112), np.float64(0.667768624474143), np.float64(0.6787773939453339), np.float64(0.6853338165456453), np.float64(0.6791934905570787), np.float64(0.6682274081250683), np.float64(0.6874553189509526), np.float64(0.6777154452742713), np.float64(0.6586679388456391), np.float64(0.6650074516912146), np.float64(0.6599099760364715), np.float64(0.6785102371338301), np.float64(0.6807087609076495), np.float64(0.662195073341164), np.float64(0.6501470663035617), np.float64(0.6730844212130924), np.float64(0.6547122740801283), np.float64(0.6478721522578808), np.float64(0.6664455847551013), np.float64(0.6542229742013557), np.float64(0.6641284416146442), np.float64(0.6488882416630191), np.float64(0.6534908011811541), np.float64(0.6670372797437861), np.float64(0.6778439718762674), np.float64(0.6860189864801899), np.float64(0.6690654563579788), np.float64(0.6649754948819346), np.float64(0.682640457667566), np.float64(0.6698298198423333), np.float64(0.6561311057379657), np.float64(0.660518268239327), np.float64(0.6553524001602881), np.float64(0.6608094947834423), np.float64(0.656192638188365), np.float64(0.6463324471449827), np.float64(0.6656866863817138), np.float64(0.6627581751472273), np.float64(0.647627115385524), np.float64(0.6435775769879154), np.float64(0.6577605938113079), np.float64(0.6674701716514547), np.float64(0.6532722274110053), np.float64(0.675193374661355), np.float64(0.6645253692660232), np.float64(0.6607066366012531), np.float64(0.6497215806151921), np.float64(0.6657456915285712), np.float64(0.6772418854412111), np.float64(0.6388765912987304), np.float64(0.6265941503004034), np.float64(0.6673768406767678), np.float64(0.6732004560966646), np.float64(0.6663008215150571), np.float64(0.6740157055054838), np.float64(0.6651291753461002), np.float64(0.6818515240152566), np.float64(0.6643605098160942), np.float64(0.6434741793941898), np.float64(0.6742085335346643), np.float64(0.6538943127691422), np.float64(0.6811884119031242), np.float64(0.6807988573659204), np.float64(0.6816433700529775), np.float64(0.6657393686767092), np.float64(0.687054994620291), np.float64(0.6591123358143536), np.float64(0.6744640934953772), np.float64(0.67589285933013), np.float64(0.6707817482724698), np.float64(0.6792598576490082), np.float64(0.6821064024459318), np.float64(0.6756756189800924), np.float64(0.6606456953424283), np.float64(0.6813586952983092), np.float64(0.68255886289609), np.float64(0.6829919948893597), np.float64(0.6696687309538732), np.float64(0.6935327119049667), np.float64(0.6877171255151373), np.float64(0.6869834577679483), np.float64(0.6755430806746531), np.float64(0.6831823735137162), np.float64(0.672247509502562), np.float64(0.6762059168639846), np.float64(0.6759077980653962), np.float64(0.6745656219161084), np.float64(0.6709047369527767), np.float64(0.6626558790066969), np.float64(0.6694594725406044), np.float64(0.6756799960193248), np.float64(0.6728024498033662), np.float64(0.6861331776837151), np.float64(0.6773956051819698), np.float64(0.6894716799289236), np.float64(0.690696071469836), np.float64(0.6766467949286781), np.float64(0.688015473856722), np.float64(0.6815400574334511), np.float64(0.6978446612302656), np.float64(0.6910786957686), np.float64(0.6879070825196171), np.float64(0.6899064253926568), np.float64(0.7008237418756573), np.float64(0.6940589032198466), np.float64(0.6847532313835725), np.float64(0.6782799689546527), np.float64(0.6825932369893175), np.float64(0.6852678780829105), np.float64(0.6842396438249597), np.float64(0.6820066696212466), np.float64(0.6880103601867696), np.float64(0.69161938038599), np.float64(0.6877050337561927), np.float64(0.6873728101032422), np.float64(0.6971941607025993), np.float64(0.6875486600275614), np.float64(0.6991962220839941), np.float64(0.6843631405558702), np.float64(0.6838122480246394), np.float64(0.6863763879273569), np.float64(0.685911623173254), np.float64(0.6809771251934297), np.float64(0.6931293399827186), np.float64(0.6836836840662353), np.float64(0.6895065214270771), np.float64(0.6703853023846226), np.float64(0.6757308737786588), np.float64(0.6799236016350824), np.float64(0.6707749062272491), np.float64(0.674101416871778), np.float64(0.6848529025171394), np.float64(0.6918991599005698), np.float64(0.7167698342811063), np.float64(0.7107539948224381), np.float64(0.6941808722855333), np.float64(0.6928147169526558), np.float64(0.7054414452141439), np.float64(0.709516406028307), np.float64(0.6997091998135377), np.float64(0.6863964219970395), np.float64(0.6953725892534204), np.float64(0.6655086667130029), np.float64(0.6988743769552505), np.float64(0.6915663901260811), np.float64(0.7027932242657946), np.float64(0.7052120715442818), np.float64(0.7059416394069267), np.float64(0.6971762900711076), np.float64(0.6660479542705485), np.float64(0.6913350039627478), np.float64(0.671487480779318), np.float64(0.6903443326806753), np.float64(0.6896203651025904), np.float64(0.684232144225349), np.float64(0.6598613946496067), np.float64(0.7074000782214845), np.float64(0.6970183703490627), np.float64(0.6694722908229531), np.float64(0.6805894544877557), np.float64(0.6712333621184454), np.float64(0.6729126059152347), np.float64(0.6757687529945202), np.float64(0.6810365530962937), np.float64(0.6768008420703886), np.float64(0.6687525389057661), np.float64(0.688280052231751), np.float64(0.6632760461410754), np.float64(0.6714213354724163), np.float64(0.6570693473640186), np.float64(0.6681475479675238), np.float64(0.689068292646938), np.float64(0.6957907179001206), np.float64(0.6958580021939953), np.float64(0.677622652662062), np.float64(0.6665544293479552), np.float64(0.6863595341164911), np.float64(0.6616830270453531), np.float64(0.6599450765233281), np.float64(0.6777544114161606), np.float64(0.6909911223100953), np.float64(0.6914380240954557), np.float64(0.6989970192604027), np.float64(0.7007868261114325), np.float64(0.6954388271098801), np.float64(0.6680832571505668), np.float64(0.6869996503339656), np.float64(0.6853817685050164), np.float64(0.6768039960663333), np.float64(0.6842133861530485), np.float64(0.6938493006140662), np.float64(0.7118485821698403), np.float64(0.6893772370512387), np.float64(0.6787548626758284), np.float64(0.7144631562143873), np.float64(0.6886605613677365), np.float64(0.6986675226262434), np.float64(0.6875172959493518), np.float64(0.6734810357916569), np.float64(0.6994018914982906), np.float64(0.6958873469889382), np.float64(0.6960685737994402), np.float64(0.6933566313747239), np.float64(0.6849230249044795), np.float64(0.6704111525759704), np.float64(0.6807383416496623), np.float64(0.6877795442003265), np.float64(0.6803435992607189), np.float64(0.6679146288109695), np.float64(0.7025213695135134), np.float64(0.6999357488410203), np.float64(0.6967304693632449), np.float64(0.6676581675097598), np.float64(0.7018352575963548), np.float64(0.681777094014962), np.float64(0.6776172254421741), np.float64(0.6812651538013687), np.float64(0.6917322265914153), np.float64(0.7178887989675966), np.float64(0.6869414843163629), np.float64(0.6949429959907085), np.float64(0.6794455915594823), np.float64(0.6737017758426227), np.float64(0.6838029465461203), np.float64(0.7025557005770665), np.float64(0.6766676866485286), np.float64(0.6901743227614542), np.float64(0.6943096814211831), np.float64(0.7138246454019658), np.float64(0.7069669754442115), np.float64(0.6938622542691965), np.float64(0.6839057858688871), np.float64(0.6855728311793626), np.float64(0.6869742175235359), np.float64(0.707181808376416), np.float64(0.6891974526891148), np.float64(0.7061574404633106), np.float64(0.6925095420650905), np.float64(0.6939244825060631), np.float64(0.6991222772462808), np.float64(0.7061399060982431), np.float64(0.6953554290846568), np.float64(0.6850698719261376), np.float64(0.685773043261884), np.float64(0.6945542599810798), np.float64(0.7076376712772333), np.float64(0.685410863348064), np.float64(0.704269356729958), np.float64(0.6987367427708134), np.float64(0.6837721710381772), np.float64(0.710254475237238), np.float64(0.7002245443073077), np.float64(0.6977318393803563), np.float64(0.7001681464074804), np.float64(0.6534118087578663), np.float64(0.7094649352607572), np.float64(0.7108614468961295), np.float64(0.699968660085617), np.float64(0.6918344924005384), np.float64(0.6765297919218383), np.float64(0.6720165855229051), np.float64(0.6774279054044382), np.float64(0.6688798602109198), np.float64(0.6916398367566011), np.float64(0.6934992842507436), np.float64(0.7018683625658944), np.float64(0.6780058323405519), np.float64(0.6857678402824692), np.float64(0.7061847060336485), np.float64(0.7187222372695082), np.float64(0.7158131441030735), np.float64(0.7180896551581236), np.float64(0.7045378909074137), np.float64(0.6946629597395283), np.float64(0.6898642156806243), np.float64(0.6970568060499873), np.float64(0.7038932931350067), np.float64(0.686671769411272), np.float64(0.7129808160562318), np.float64(0.7049380785651499), np.float64(0.7167270519812601), np.float64(0.7133782645980536), np.float64(0.6946805309213354), np.float64(0.7105412483032536), np.float64(0.6997676871311549), np.float64(0.6917094445496453), np.float64(0.7060726295825606), np.float64(0.7070867616595559), np.float64(0.7222557673169798), np.float64(0.7196997582145663), np.float64(0.6966692237889225), np.float64(0.695845638736092), np.float64(0.7087750245506792), np.float64(0.6982565111552081), np.float64(0.6983367983255369), np.float64(0.7035356263742607), np.float64(0.6968665562368277), np.float64(0.6967588629594701), np.float64(0.6896996201313076), np.float64(0.6927462788195851), np.float64(0.7173031160115722), np.float64(0.7055261427615176), np.float64(0.7119965816842155), np.float64(0.7026421170688723), np.float64(0.7001846183028321), np.float64(0.6876842884890106), np.float64(0.7007836802302863), np.float64(0.7088113417500671), np.float64(0.7082790866034521), np.float64(0.6817539711633843), np.float64(0.695175304210192), np.float64(0.6888377584451619), np.float64(0.706741924462974), np.float64(0.6847885714802943), np.float64(0.6655527445347958), np.float64(0.671743054383619), np.float64(0.6924501073904228), np.float64(0.6841348189185082), np.float64(0.6671114565661154), np.float64(0.6632234768756973), np.float64(0.6821669537849085), np.float64(0.6961758870809542), np.float64(0.6791998819784624), np.float64(0.6898932585615073), np.float64(0.6906878580023004), np.float64(0.6904987729800517), np.float64(0.6531900013865823), np.float64(0.693251512457121), np.float64(0.6743461992057798), np.float64(0.6811363036110933), np.float64(0.6946089728722625), np.float64(0.7031167238481796), np.float64(0.7122484335977936), np.float64(0.6945495903946014), np.float64(0.7083924137927468), np.float64(0.6811348419353596), np.float64(0.6737446164143053), np.float64(0.6923612773925745), np.float64(0.7106543320027363), np.float64(0.685897322509911), np.float64(0.7141036175363612), np.float64(0.712543282334593), np.float64(0.714812915367211), np.float64(0.7008049195719671), np.float64(0.7015188368533241), np.float64(0.6943492212082141), np.float64(0.7132023029699466), np.float64(0.6958792848932198), np.float64(0.6903144508498731), np.float64(0.698710207433038), np.float64(0.7218093567747293), np.float64(0.7128674129759299), np.float64(0.7183416256157817), np.float64(0.7193990347415025), np.float64(0.7137172363481448), np.float64(0.7525406855746851), np.float64(0.7529604864502052), np.float64(0.7572811695056193), np.float64(0.7695125694887422), np.float64(0.7413236233328435), np.float64(0.7523342143060853), np.float64(0.7378319414495561), np.float64(0.7481679254681181), np.float64(0.7357249260816436), np.float64(0.7463957636561037), np.float64(0.7329634880694139), np.float64(0.7617836598333229), np.float64(0.7311589237582621), np.float64(0.7499441147783623), np.float64(0.7361562343717214), np.float64(0.751730326310086), np.float64(0.7449502676387495), np.float64(0.7329078089577121), np.float64(0.7454359964256414), np.float64(0.7465129916915422), np.float64(0.74045773527254), np.float64(0.7516547395139762), np.float64(0.7666424970899867), np.float64(0.7381075603475791), np.float64(0.7391767884632294), np.float64(0.7640845312883989), np.float64(0.7619686926892806), np.float64(0.7601028453859849), np.float64(0.7638269406631208), np.float64(0.7403758599365743), np.float64(0.7639137813940095), np.float64(0.7714416596924484), np.float64(0.7437572492123707), np.float64(0.7209804168809164), np.float64(0.7440017345762325), np.float64(0.7515102455915953), np.float64(0.7600503438038273), np.float64(0.7687518763061393), np.float64(0.7406220663279316), np.float64(0.7514457357156508), np.float64(0.7719954054810191), np.float64(0.7593848625767688), np.float64(0.7729631922976214), np.float64(0.7466902536745007), np.float64(0.7391838086976258), np.float64(0.7592124310700263), np.float64(0.7666235415496908), np.float64(0.7481037861339563), np.float64(0.7656490110984269), np.float64(0.7478573224641765), np.float64(0.7482916396113596), np.float64(0.7558053707195692), np.float64(0.7401758933777131), np.float64(0.7343567617557797), np.float64(0.7616670595097247), np.float64(0.758234805382196), np.float64(0.7586047290008195), np.float64(0.7395413173906671), np.float64(0.754067507192048), np.float64(0.7483556707366625), np.float64(0.7644647391423656), np.float64(0.7490115325297875), np.float64(0.7515001623238345), np.float64(0.7538126624598732), np.float64(0.7669668758369544), np.float64(0.7754239183210605), np.float64(0.7676470662022118), np.float64(0.7577491534808749), np.float64(0.7718092935965993), np.float64(0.7357645742734076), np.float64(0.7709148464619531), np.float64(0.7145573098632655), np.float64(0.6921621307954912), np.float64(0.6773302073912708), np.float64(0.6880704274207222), np.float64(0.6985384117071393), np.float64(0.6850908895796439), np.float64(0.7043296576181032), np.float64(0.6910755274720027), np.float64(0.7142409599874885), np.float64(0.6945811788252151), np.float64(0.6585493003084621), np.float64(0.6887630652557049), np.float64(0.6634653711291096), np.float64(0.6992125557504695), np.float64(0.712999022384807), np.float64(0.6877503473294813), np.float64(0.6976505475057436), np.float64(0.7117725315645421), np.float64(0.7097752234885247), np.float64(0.7238004251226199), np.float64(0.7221058681823845), np.float64(0.7036009164995308), np.float64(0.695091949963584), np.float64(0.6981806340209389), np.float64(0.6986707184600606), np.float64(0.6951446849237665), np.float64(0.7213191151177437), np.float64(0.7029041910211469), np.float64(0.7263784876553714), np.float64(0.6974828887515), np.float64(0.7006545690899817), np.float64(0.7352847545042155), np.float64(0.7118036386543478), np.float64(0.7129392238807567), np.float64(0.7195372746453481), np.float64(0.7199722850930087), np.float64(0.6956132009501895), np.float64(0.718357348026397), np.float64(0.6949440606761508), np.float64(0.7083475964460473), np.float64(0.6923370133893937), np.float64(0.7275850011725551), np.float64(0.7323885691126941), np.float64(0.6856433639834199), np.float64(0.7174627614579838), np.float64(0.7129231899447203), np.float64(0.694542252452073), np.float64(0.7128448299131872), np.float64(0.7063336104820827), np.float64(0.7141182681217514), np.float64(0.719142025124158), np.float64(0.7182572050714426), np.float64(0.6911086844854145), np.float64(0.7167722262362715), np.float64(0.6923215343826205), np.float64(0.6998523797406552), np.float64(0.6916601921028253), np.float64(0.6934431964022937), np.float64(0.6626016341495027), np.float64(0.6862741260394081), np.float64(0.7135880823141799), np.float64(0.6946440098239643), np.float64(0.6941141296411452), np.float64(0.6881505786151725), np.float64(0.6928094974051973), np.float64(0.6994537640481064), np.float64(0.6928268936553373), np.float64(0.6853499879238707), np.float64(0.6815641541946074), np.float64(0.7014886612227538), np.float64(0.7173296050939026), np.float64(0.7022906703618798), np.float64(0.6872254357513741), np.float64(0.7225964073768043), np.float64(0.7001356206837623)]
        list_of_rms_standard_errors = [np.float64(0.026311714702306033), np.float64(0.015643284740140786), np.float64(0.019325273790648505), np.float64(0.021735882161416406), np.float64(0.02462370063959173), np.float64(0.02535123227882637), np.float64(0.025124971713676118), np.float64(0.02714143325265193), np.float64(0.028980435875035782), np.float64(0.0295629922342832), np.float64(0.029582161627414203), np.float64(0.029895208286430414), np.float64(0.02996486637713207), np.float64(0.030561937456552262), np.float64(0.031346716685524904), np.float64(0.03154867873710461), np.float64(0.031121978969953678), np.float64(0.032511656208727396), np.float64(0.03155254736658851), np.float64(0.03392625848933874), np.float64(0.03271829845302452), np.float64(0.03461886985180699), np.float64(0.03361481866638358), np.float64(0.034301151665256144), np.float64(0.03554302023174955), np.float64(0.03550037839504403), np.float64(0.03409251895423903), np.float64(0.035223009521157964), np.float64(0.03499803114853457), np.float64(0.03633513056081145), np.float64(0.03786362507468496), np.float64(0.036543902058414166), np.float64(0.03716982503638162), np.float64(0.03644160666746225), np.float64(0.036756333622623066), np.float64(0.03850000851729215), np.float64(0.037817348178319526), np.float64(0.03820501110442451), np.float64(0.0376418940371615), np.float64(0.03764747267171946), np.float64(0.03797305467146848), np.float64(0.03862870162399217), np.float64(0.03705083498774385), np.float64(0.03828072052915814), np.float64(0.03869215875883294), np.float64(0.03873065784063241), np.float64(0.040231563058085874), np.float64(0.03807919642989264), np.float64(0.03992183357824489), np.float64(0.039391423329859517), np.float64(0.0399428688114906), np.float64(0.040988659366296844), np.float64(0.040689011146714034), np.float64(0.041032152874163084), np.float64(0.0410157700401292), np.float64(0.04161766760243469), np.float64(0.04197063827029609), np.float64(0.04163494822562733), np.float64(0.042301938500232054), np.float64(0.042571768163903595), np.float64(0.042336963759743336), np.float64(0.04104721373766674), np.float64(0.04254303814759024), np.float64(0.04229452553989982), np.float64(0.04138775153589814), np.float64(0.042726678847091436), np.float64(0.04287166734809137), np.float64(0.04303553190361965), np.float64(0.04424267266776563), np.float64(0.04384763081046431), np.float64(0.04422680615449956), np.float64(0.0438021614878518), np.float64(0.04299377552478874), np.float64(0.04193237903193813), np.float64(0.04244338196155651), np.float64(0.04250134170094923), np.float64(0.0443562209303408), np.float64(0.04290734276707733), np.float64(0.04292746811919332), np.float64(0.04436043452678266), np.float64(0.042619715691129406), np.float64(0.04291382257248887), np.float64(0.04423365604027289), np.float64(0.04456933080394905), np.float64(0.04337419949594567), np.float64(0.044757877567761714), np.float64(0.044109859057501116), np.float64(0.04448137923109799), np.float64(0.04639848630852593), np.float64(0.04465301838757097), np.float64(0.04685864371435366), np.float64(0.0452451571911202), np.float64(0.044830211753402446), np.float64(0.045714212807063545), np.float64(0.044730007046916084), np.float64(0.04437025697476748), np.float64(0.04439534614824611), np.float64(0.04416341975076607), np.float64(0.044803060845564596), np.float64(0.04402818029371547), np.float64(0.04392796711266767), np.float64(0.043896206075586676), np.float64(0.04469434611036009), np.float64(0.0436542985073487), np.float64(0.0440976378513545), np.float64(0.044778428191020216), np.float64(0.04400012099132488), np.float64(0.04416705408279913), np.float64(0.0457299350121519), np.float64(0.04440358054856603), np.float64(0.0438355143083516), np.float64(0.04362648328961573), np.float64(0.044380881675558925), np.float64(0.043750660211033895), np.float64(0.04428966160869881), np.float64(0.04533302682029369), np.float64(0.04627666843735055), np.float64(0.0457715097424931), np.float64(0.04556376195502205), np.float64(0.04653391858163642), np.float64(0.04538538410438579), np.float64(0.04535017364074748), np.float64(0.045201152834447554), np.float64(0.04608266161457766), np.float64(0.04584542528233718), np.float64(0.046422352078519594), np.float64(0.04550569230720059), np.float64(0.046595719462767964), np.float64(0.04542892619796934), np.float64(0.04695439160249166), np.float64(0.04585812496800507), np.float64(0.04632382422751888), np.float64(0.04687578795465233), np.float64(0.04571962846629385), np.float64(0.04650765654940372), np.float64(0.0473730364217242), np.float64(0.04812106031550745), np.float64(0.04644834313044339), np.float64(0.04600900137334488), np.float64(0.046074072034419905), np.float64(0.04657108112078811), np.float64(0.04740257231134489), np.float64(0.04740317565951339), np.float64(0.04573763382891783), np.float64(0.0479919549583253), np.float64(0.0470853396378315), np.float64(0.04736916658076674), np.float64(0.04681282194945932), np.float64(0.04749780516414691), np.float64(0.04658291441018933), np.float64(0.045717737212815884), np.float64(0.04757339669650202), np.float64(0.046189548673295), np.float64(0.04723876337058944), np.float64(0.04745206016266298), np.float64(0.04750091328888766), np.float64(0.04649739324337062), np.float64(0.04720668932859219), np.float64(0.04709338688378776), np.float64(0.04694801690629976), np.float64(0.04643494372214369), np.float64(0.04637943090344747), np.float64(0.04701381290318637), np.float64(0.04787621882719199), np.float64(0.04818154104754616), np.float64(0.047693346331212916), np.float64(0.04824319984752102), np.float64(0.04634303344134771), np.float64(0.04696949665690749), np.float64(0.0475054784496952), np.float64(0.04642194640226753), np.float64(0.04686628398445263), np.float64(0.04560423077398934), np.float64(0.04821203042732172), np.float64(0.04648979895182827), np.float64(0.0473872602900747), np.float64(0.04786975114333047), np.float64(0.04773727669950067), np.float64(0.04750304021782815), np.float64(0.048264665310417953), np.float64(0.04756816583116439), np.float64(0.047051744927499015), np.float64(0.047372936658823174), np.float64(0.049238169249448065), np.float64(0.04742235124818672), np.float64(0.047152223262262594), np.float64(0.047609579103595694), np.float64(0.04659766028594049), np.float64(0.047109581383723634), np.float64(0.04706555243896792), np.float64(0.04650316695952001), np.float64(0.047126632938614785), np.float64(0.04703266561159639), np.float64(0.04565352235025527), np.float64(0.04580557981513331), np.float64(0.047740583525994654), np.float64(0.047267536893894124), np.float64(0.04827402574912142), np.float64(0.04740009950806339), np.float64(0.04904603958313003), np.float64(0.047934501186818745), np.float64(0.046862435836354396), np.float64(0.04657861010788418), np.float64(0.046239584519004055), np.float64(0.04589523887966211), np.float64(0.046998394386095356), np.float64(0.048283608568290844), np.float64(0.04734629704124715), np.float64(0.047553197587862965), np.float64(0.04759079206063807), np.float64(0.04826843000739034), np.float64(0.0463074059716809), np.float64(0.04742321516210482), np.float64(0.047328483050497035), np.float64(0.04806121438639061), np.float64(0.047114864542705595), np.float64(0.04786971953477714), np.float64(0.04856449452603532), np.float64(0.047098769619757036), np.float64(0.04794818698782276), np.float64(0.048350165756370606), np.float64(0.048364617020465614), np.float64(0.04725847271298852), np.float64(0.0467254700186281), np.float64(0.046982116760627704), np.float64(0.04745257116517061), np.float64(0.04778473541915864), np.float64(0.04753054830420225), np.float64(0.047858372378364704), np.float64(0.04614430540373862), np.float64(0.047375994747494564), np.float64(0.0482846336663042), np.float64(0.04661979998856723), np.float64(0.046251219024073226), np.float64(0.045398447851294356), np.float64(0.0466785896947119), np.float64(0.048065254502414625), np.float64(0.04670448091306876), np.float64(0.04694401398869675), np.float64(0.04599452245112476), np.float64(0.04621742126677934), np.float64(0.04510528836497351), np.float64(0.04693951991684146), np.float64(0.047234770318484776), np.float64(0.0481023101355548), np.float64(0.047810809791530115), np.float64(0.04796240970128632), np.float64(0.04737238951850598), np.float64(0.04751220642922264), np.float64(0.04808038869334545), np.float64(0.0474597842587415), np.float64(0.04746313694819605), np.float64(0.04805750418186123), np.float64(0.048586039948323945), np.float64(0.047644826662140975), np.float64(0.04678825343233207), np.float64(0.047524578865189314), np.float64(0.04780769133851051), np.float64(0.04717373506539213), np.float64(0.04724634972674663), np.float64(0.0471081490011983), np.float64(0.049536518068226856), np.float64(0.04741249606204671), np.float64(0.04779235128917965), np.float64(0.048234262135745797), np.float64(0.04800045768284377), np.float64(0.04826266427301861), np.float64(0.04658236232906264), np.float64(0.04781690814409504), np.float64(0.04709472999328437), np.float64(0.0490416169505587), np.float64(0.04872247071440719), np.float64(0.04875852742458514), np.float64(0.04750240059534796), np.float64(0.04927727544180216), np.float64(0.049660528437031955), np.float64(0.048033249772070935), np.float64(0.049170653735322305), np.float64(0.04852772381002719), np.float64(0.048346871489842805), np.float64(0.04770633559033976), np.float64(0.04917216218086657), np.float64(0.04959123311784163), np.float64(0.04867280276973604), np.float64(0.048365137894001244), np.float64(0.04794129244740781), np.float64(0.0489652313445048), np.float64(0.0480673807471129), np.float64(0.049734955980729266), np.float64(0.04837587594697836), np.float64(0.0499088319023301), np.float64(0.049005307677927344), np.float64(0.04679022567320398), np.float64(0.04811529570313955), np.float64(0.048967244870805074), np.float64(0.04852738105835952), np.float64(0.04907571262854783), np.float64(0.04907021955084796), np.float64(0.04884187454102645), np.float64(0.04792302948241198), np.float64(0.04823370004229929), np.float64(0.04959768844835789), np.float64(0.04976936674912855), np.float64(0.04910867925021731), np.float64(0.04959797751115808), np.float64(0.04954215532419755), np.float64(0.04984378219814721), np.float64(0.050267158086140115), np.float64(0.05044542513589954), np.float64(0.050614462662955335), np.float64(0.049342874549250584), np.float64(0.04981849221273842), np.float64(0.05084427079922132), np.float64(0.05033317491125249), np.float64(0.0498360841794115), np.float64(0.05039184603667096), np.float64(0.05027088786981539), np.float64(0.04882502430922665), np.float64(0.04999732016208735), np.float64(0.05003768157459181), np.float64(0.051024976471333215), np.float64(0.05004392479663944), np.float64(0.050703510582732224), np.float64(0.05118595968969655), np.float64(0.05099547247962779), np.float64(0.04955812202661203), np.float64(0.05096180942280559), np.float64(0.04873844491926156), np.float64(0.0495472890715988), np.float64(0.04973889033107529), np.float64(0.0509707879256183), np.float64(0.04909484701312509), np.float64(0.04915444956994256), np.float64(0.049234408634667444), np.float64(0.048267430651773274), np.float64(0.04878458204186655), np.float64(0.04952999276670662), np.float64(0.04944448670549947), np.float64(0.04954876496416468), np.float64(0.05035332942074527), np.float64(0.04992355233941911), np.float64(0.05022322127404727), np.float64(0.04989513372530521), np.float64(0.04965511167581077), np.float64(0.050062389564123615), np.float64(0.049822266248132234), np.float64(0.04974831582666442), np.float64(0.05008090228130897), np.float64(0.050863705712078815), np.float64(0.04890576732678909), np.float64(0.05047587283788648), np.float64(0.04988190845412399), np.float64(0.04941965783533655), np.float64(0.04979423841290814), np.float64(0.050933769560877994), np.float64(0.04888028117057961), np.float64(0.049843034273299835), np.float64(0.0491180394978595), np.float64(0.04858522948569635), np.float64(0.05093519681148573), np.float64(0.04999697572467253), np.float64(0.050510481287368784), np.float64(0.05042592326451277), np.float64(0.051497565158558516), np.float64(0.04958143101972615), np.float64(0.050642035797924256), np.float64(0.049547090150197835), np.float64(0.0491313658838886), np.float64(0.04953315903795977), np.float64(0.04940541101187649), np.float64(0.05011274058326699), np.float64(0.04773838850332554), np.float64(0.0506202077427188), np.float64(0.05039965896485986), np.float64(0.05046328526384288), np.float64(0.0501970405496067), np.float64(0.05070082987763191), np.float64(0.04954399072593518), np.float64(0.050207440399839584), np.float64(0.051242818834953616), np.float64(0.05015619136748268), np.float64(0.05073108211511251), np.float64(0.05145933138069027), np.float64(0.05007561954929882), np.float64(0.04977680011405101), np.float64(0.050590737419257494), np.float64(0.05118358250071386), np.float64(0.050493084767608554), np.float64(0.05012942461990953), np.float64(0.050856520030429855), np.float64(0.05168742610882094), np.float64(0.051565005626651644), np.float64(0.05098705415985595), np.float64(0.050860368555663726), np.float64(0.04993608343818327), np.float64(0.04970283404287738), np.float64(0.04930910158204603), np.float64(0.0499401481155537), np.float64(0.0507187686033241), np.float64(0.051817438328246715), np.float64(0.05039731153803731), np.float64(0.05125701387500595), np.float64(0.05053824121115598), np.float64(0.05071991966888315), np.float64(0.051239088630877956), np.float64(0.05054238474008207), np.float64(0.05040808597499364), np.float64(0.05096087184423651), np.float64(0.050325883465918174), np.float64(0.05017453490857447), np.float64(0.05025205366248717), np.float64(0.05139025771530298), np.float64(0.05076822874362203), np.float64(0.05190058613362413), np.float64(0.05251737215369069), np.float64(0.05075621248337087), np.float64(0.04978832827843502), np.float64(0.050115314785188686), np.float64(0.051700173313754635), np.float64(0.051719918045828456), np.float64(0.05143489958170256), np.float64(0.052674728707265736), np.float64(0.05192173854254494), np.float64(0.053152651636504906), np.float64(0.05274631398561054), np.float64(0.05213009705111184), np.float64(0.05105060489741177), np.float64(0.05224593185344223), np.float64(0.05224415898418655), np.float64(0.05295344468990416), np.float64(0.05311654910689097), np.float64(0.05154210467991131), np.float64(0.05207533892248321), np.float64(0.05232230371311849), np.float64(0.05298534306101854), np.float64(0.05074766425698428), np.float64(0.05249949923011714), np.float64(0.049836112146864), np.float64(0.05310572516438553), np.float64(0.05294717196209384), np.float64(0.05213992289675222), np.float64(0.05221329563763353), np.float64(0.052420153347518685), np.float64(0.051947715961452604), np.float64(0.052242211297300924), np.float64(0.05116782509777748), np.float64(0.05246606915856192), np.float64(0.051362910123761714), np.float64(0.05061203968561925), np.float64(0.05127417659577793), np.float64(0.05129286977575198), np.float64(0.051376888143728594), np.float64(0.05128748293815355), np.float64(0.052969947420567355), np.float64(0.050916994155747665), np.float64(0.049288832672651016), np.float64(0.05049934467738523), np.float64(0.051011005200327315), np.float64(0.05145484836786548), np.float64(0.05084370157103932), np.float64(0.05025155225502179), np.float64(0.051232129844414935), np.float64(0.05112310647500596), np.float64(0.05017996941623754), np.float64(0.05051441929701044), np.float64(0.05093193637489347), np.float64(0.05262399916157202), np.float64(0.051383511659635187), np.float64(0.051084285108293485), np.float64(0.05194957673417664), np.float64(0.05159975735976709), np.float64(0.05031340123043023), np.float64(0.05173670336508219), np.float64(0.05064255959690936), np.float64(0.05211961966510963), np.float64(0.05251809548019009), np.float64(0.05167752676312531), np.float64(0.05270119213160343), np.float64(0.052484967610666435), np.float64(0.052706976637927304), np.float64(0.05224000773523884), np.float64(0.051456031012722156), np.float64(0.052404621187906335), np.float64(0.052164227141685816), np.float64(0.05407692162140586), np.float64(0.052894900379946604), np.float64(0.05312229911611884), np.float64(0.05086575680525532), np.float64(0.051801430467248735), np.float64(0.05111126704999609), np.float64(0.0511694147217186), np.float64(0.05224850184823555), np.float64(0.05217083660485466), np.float64(0.05196037186041129), np.float64(0.05066914785741829), np.float64(0.05304358616337166), np.float64(0.0509612091097832), np.float64(0.05184025965377705), np.float64(0.05220858673139169), np.float64(0.05271951971773086), np.float64(0.052556875756021255), np.float64(0.05259747204712525), np.float64(0.05149126185748635), np.float64(0.05241013206086938), np.float64(0.05075993223154437), np.float64(0.05087805607186477), np.float64(0.05176832816084782), np.float64(0.05264585309508509), np.float64(0.05193373339566508), np.float64(0.052661601662321554), np.float64(0.054120325767352186), np.float64(0.05259754369461529), np.float64(0.05219234737106877), np.float64(0.051744887133172025), np.float64(0.05263191961480409), np.float64(0.05081674897111829), np.float64(0.05154448606804689), np.float64(0.05213151905993257), np.float64(0.05314276358039486), np.float64(0.05187580443894631), np.float64(0.05123442652683088), np.float64(0.05106370328387121), np.float64(0.0519986595230659), np.float64(0.0526353449809632), np.float64(0.05387940502340445), np.float64(0.05240846849032113), np.float64(0.053146816941922595), np.float64(0.05203967362235584), np.float64(0.05329697829599672), np.float64(0.05331748267767119), np.float64(0.054183962181317546), np.float64(0.0524982069588699), np.float64(0.05156271964361674), np.float64(0.052670797173777956), np.float64(0.052471236744202496), np.float64(0.05242606313041373), np.float64(0.0527061973339006), np.float64(0.05374346673537342), np.float64(0.05230312464446712), np.float64(0.050609047412283795), np.float64(0.05173891849526757), np.float64(0.05176772297077278), np.float64(0.051776717175684224), np.float64(0.05108081454640804), np.float64(0.05326305728970922), np.float64(0.053133518376584504), np.float64(0.05272314793435606), np.float64(0.05263168979496387), np.float64(0.052449127581985985), np.float64(0.051829824525544126), np.float64(0.05282901393927542), np.float64(0.054197038511075546), np.float64(0.05307907429710534), np.float64(0.05321486290287786), np.float64(0.053935914584881534), np.float64(0.05237970508930428), np.float64(0.0526779172281233), np.float64(0.052945578494686125), np.float64(0.05096428049674601), np.float64(0.05386504678055479), np.float64(0.052353892753889734), np.float64(0.05171463340302091), np.float64(0.05246976115912233), np.float64(0.05307786370318923), np.float64(0.05419313096143801), np.float64(0.05343469680429548), np.float64(0.05495782909081583), np.float64(0.05218515212154913), np.float64(0.05221291446697148), np.float64(0.05176906220776075), np.float64(0.051284246486879494), np.float64(0.05191819505156817), np.float64(0.05367891910539621), np.float64(0.052881093561043806), np.float64(0.05316279259888868), np.float64(0.051756936109637074), np.float64(0.05259158960328099), np.float64(0.05256237466991045), np.float64(0.05104201974717903), np.float64(0.054652210028325834), np.float64(0.053998994492465466), np.float64(0.053807560949031136), np.float64(0.054398107261220054), np.float64(0.05289994374713386), np.float64(0.05324583366724426), np.float64(0.052450478057889466), np.float64(0.05513652334756549), np.float64(0.05606113562710685), np.float64(0.05492287096412813), np.float64(0.05435498875576505), np.float64(0.054315378169093556), np.float64(0.05363068687515715), np.float64(0.05386487387395687), np.float64(0.05376579549543112), np.float64(0.05174302949816335), np.float64(0.053663788054161325), np.float64(0.052865080182815764), np.float64(0.05315396176762249), np.float64(0.05275221422017599), np.float64(0.05317239338254203), np.float64(0.05195515736796091), np.float64(0.051710615745654505), np.float64(0.052864835236081466), np.float64(0.051729195669866675), np.float64(0.052466672264947306), np.float64(0.05324085165241639), np.float64(0.053376920432907746), np.float64(0.05282519201846641), np.float64(0.05168280212675544), np.float64(0.05280493418504457), np.float64(0.05239792212816535), np.float64(0.05334962621346256), np.float64(0.051754656539140534), np.float64(0.05242474945409344), np.float64(0.051260049887466186), np.float64(0.05193580098929576), np.float64(0.05174694668429201), np.float64(0.051196402197652244), np.float64(0.052126326352862545), np.float64(0.0526201157052245), np.float64(0.05267331622451083), np.float64(0.051452771017261985), np.float64(0.05390386078018206), np.float64(0.051007167611107705), np.float64(0.0511853577959961), np.float64(0.05213108334837635), np.float64(0.051962987659046965), np.float64(0.05455406587350579), np.float64(0.05371925937280433), np.float64(0.054580515082731845), np.float64(0.053513877387429194), np.float64(0.05322970601209578), np.float64(0.052732706948727226), np.float64(0.052416211651423864), np.float64(0.05344263837522679), np.float64(0.0528280679037998), np.float64(0.054509929932793175), np.float64(0.05337182666799893), np.float64(0.054740933108810605), np.float64(0.05182915897266456), np.float64(0.05237214560398277), np.float64(0.05204946121249841), np.float64(0.05313899062988451), np.float64(0.05250182745657422), np.float64(0.05332198768114796), np.float64(0.05227323104252608), np.float64(0.05236407834504854), np.float64(0.05354312504544486), np.float64(0.05283667010008114), np.float64(0.052510362442113497), np.float64(0.05221151393080283), np.float64(0.05282982081792984), np.float64(0.05334907290484973), np.float64(0.05352295556538052), np.float64(0.0535473326452402), np.float64(0.05401609300660942), np.float64(0.05357799974511029), np.float64(0.05338240818385922), np.float64(0.05417516930407282), np.float64(0.05352250817340869), np.float64(0.053593656761143464), np.float64(0.054082073666396134), np.float64(0.05326638792378995), np.float64(0.05328715280714421), np.float64(0.05220112046485352), np.float64(0.052682710379585766), np.float64(0.05463915457375026), np.float64(0.05242514483907052), np.float64(0.054109963549497576), np.float64(0.055009888212983436), np.float64(0.053766734525269304), np.float64(0.05252774305458679), np.float64(0.05387038579845505), np.float64(0.053452273583549724), np.float64(0.05350977979985207), np.float64(0.05352298174698994), np.float64(0.05234025681798596), np.float64(0.05183702534798249), np.float64(0.05380645153413374), np.float64(0.05319874707070775), np.float64(0.05199289085140784), np.float64(0.054334967676132266), np.float64(0.05376180690428052), np.float64(0.0537343235285877), np.float64(0.05225704129354009), np.float64(0.054598844526322986), np.float64(0.05376949397854), np.float64(0.05433260115764104), np.float64(0.054359439766938626), np.float64(0.054798649612862425), np.float64(0.05377764445137537), np.float64(0.05369576358527711), np.float64(0.053671385057728706), np.float64(0.053219559304552924), np.float64(0.05378804999630134), np.float64(0.05348077709871487), np.float64(0.05387740351929006), np.float64(0.05224104443850215), np.float64(0.054486394568273665), np.float64(0.052984016912254246), np.float64(0.053229434756627135), np.float64(0.05384075723086035), np.float64(0.052289389099142505), np.float64(0.05177803845309906), np.float64(0.053152389403986686), np.float64(0.053022416131387744), np.float64(0.053335185712548895), np.float64(0.05393747244391545), np.float64(0.05344445243885574), np.float64(0.05554143473376296), np.float64(0.05370463829202245), np.float64(0.052831420945435026), np.float64(0.053928623447005865), np.float64(0.05351460355524177), np.float64(0.052205996081769256), np.float64(0.054410972980755666), np.float64(0.053262883211497136), np.float64(0.052413377776453975), np.float64(0.05402723670539599), np.float64(0.05477632282867968), np.float64(0.05351906347785419), np.float64(0.051926099629167874), np.float64(0.05375723146544345), np.float64(0.05274896696053541), np.float64(0.0531672999205561), np.float64(0.0540429664905297), np.float64(0.05412092643105489), np.float64(0.05363729752700634), np.float64(0.05367140657015453), np.float64(0.05530328572004341), np.float64(0.05257419517492263), np.float64(0.053781321867932885), np.float64(0.05371257370514486), np.float64(0.051743133814668586), np.float64(0.05456554810475379), np.float64(0.054888291041868556), np.float64(0.05420959446400403), np.float64(0.05570974861497902), np.float64(0.05434657149103144), np.float64(0.053801253935107374), np.float64(0.054790555208970186), np.float64(0.054220480107315165), np.float64(0.05457014241240822), np.float64(0.05495360305544835), np.float64(0.05444657350767475), np.float64(0.05488188196237194), np.float64(0.05495969500704664), np.float64(0.05376552860453717), np.float64(0.054246543361698873), np.float64(0.05436261240003262), np.float64(0.053758031049331544), np.float64(0.05344582179843933), np.float64(0.053301291257264335), np.float64(0.05412447593427239), np.float64(0.0532623064469262), np.float64(0.05488627699147174), np.float64(0.05444751136111631), np.float64(0.05667589961269294), np.float64(0.05445643549145924), np.float64(0.05490525309362832), np.float64(0.054612367343182346), np.float64(0.056644173600359164), np.float64(0.05548634255938152), np.float64(0.054100535256831776), np.float64(0.05491443143374471), np.float64(0.05289939718278774), np.float64(0.05432812296944076), np.float64(0.05413590928645584), np.float64(0.05458073485481646), np.float64(0.05596954869176281), np.float64(0.05478989055776752), np.float64(0.05475061117692383), np.float64(0.05443200518511369), np.float64(0.05509118922890478), np.float64(0.05318728124592129), np.float64(0.05185307994780996), np.float64(0.053868714024738515), np.float64(0.053669427115766215), np.float64(0.054373236430490676), np.float64(0.055432162704736616), np.float64(0.053718418756541673), np.float64(0.05296652015383244), np.float64(0.05169144181260797), np.float64(0.052173945953710145), np.float64(0.053217528634081704), np.float64(0.05418597823848225), np.float64(0.054172287012306794), np.float64(0.054064755461842584), np.float64(0.05357606794276714), np.float64(0.053262510339183576), np.float64(0.05332983499490116), np.float64(0.05270329225675119), np.float64(0.05187581622330624), np.float64(0.0537250757286321), np.float64(0.05547435697729552), np.float64(0.05394908288933504), np.float64(0.05408285111096967), np.float64(0.053627055125972455), np.float64(0.05428117245242439), np.float64(0.05374876251451239), np.float64(0.053706062420104506), np.float64(0.05325414388182209), np.float64(0.05392101328230951), np.float64(0.05305894871857178), np.float64(0.053045883882997684), np.float64(0.05328063600939147), np.float64(0.05198737997508062), np.float64(0.05231353295467364), np.float64(0.052926262870892436), np.float64(0.0519695998406536), np.float64(0.05328859563313853), np.float64(0.05393094522296604), np.float64(0.05221180344491439), np.float64(0.05353149665081974), np.float64(0.052923216957527004), np.float64(0.05295611453929867), np.float64(0.052506685015803994), np.float64(0.05323903624562901), np.float64(0.053025528926637984), np.float64(0.05197581712130274), np.float64(0.05427488745154523), np.float64(0.05460618609065811), np.float64(0.05146970721952315), np.float64(0.052515358120610325), np.float64(0.0539624748445311), np.float64(0.05358046360934823), np.float64(0.053417840747146506), np.float64(0.05282289294773713), np.float64(0.053210977151819765), np.float64(0.05391778333912854), np.float64(0.05448506843408251), np.float64(0.05301756852768607), np.float64(0.05416878433511364), np.float64(0.05242605367112656), np.float64(0.05311546300629047), np.float64(0.05513908837184368), np.float64(0.05390309224036269), np.float64(0.05332178388392117), np.float64(0.05440613505420935), np.float64(0.055268568746250635), np.float64(0.05447561703872322), np.float64(0.053921792335335686), np.float64(0.05328556604571662), np.float64(0.055631655610848145), np.float64(0.05546856060856944), np.float64(0.05434599392893778), np.float64(0.05374855304516572), np.float64(0.05390464741376916), np.float64(0.05542353580844732), np.float64(0.05477885125439917), np.float64(0.0558298320058708), np.float64(0.05529190788820097), np.float64(0.05519301515824293), np.float64(0.0528286914344141), np.float64(0.05387589636714827), np.float64(0.053397664284547855), np.float64(0.05497832203657663), np.float64(0.05427286314350073), np.float64(0.05322013388671362), np.float64(0.0530360986376895), np.float64(0.05304670735580961), np.float64(0.05237148750914688), np.float64(0.05439976184776977), np.float64(0.053404387754102285), np.float64(0.05461988113377494), np.float64(0.0548796988969848), np.float64(0.054298078291470896), np.float64(0.05495964873734092), np.float64(0.05484792691524681), np.float64(0.05510677160885135), np.float64(0.05523926587375783), np.float64(0.05463613270198758), np.float64(0.05459093766220873), np.float64(0.05689231659719611), np.float64(0.05548071464804187), np.float64(0.05773778774731092), np.float64(0.0560347125928446), np.float64(0.056742197521807806), np.float64(0.05592429391385025), np.float64(0.05594042319039297), np.float64(0.05468629598005518), np.float64(0.05771172144481248), np.float64(0.056011339088282694), np.float64(0.05539315625542062), np.float64(0.05451040467669826), np.float64(0.055501852737845664), np.float64(0.057360613169385656), np.float64(0.05784049364324092), np.float64(0.056327277169665256), np.float64(0.05704479747646412), np.float64(0.05871180106315122), np.float64(0.056369501463101546), np.float64(0.0577901697467166), np.float64(0.057482817651075734), np.float64(0.056259824052134186), np.float64(0.05630981070926519), np.float64(0.055834464718444145), np.float64(0.05654117548547148), np.float64(0.05594503935132818), np.float64(0.056925476703065706), np.float64(0.05669494124393006), np.float64(0.05696434622667486), np.float64(0.05682888510504799), np.float64(0.05757844478657386), np.float64(0.055191627363371965), np.float64(0.057271097246016055), np.float64(0.05604649467210997), np.float64(0.05425884990935398), np.float64(0.057072641615001495), np.float64(0.055672184622420046), np.float64(0.0551899993578082), np.float64(0.055315399720098295), np.float64(0.056128300502743506), np.float64(0.05575860299313947), np.float64(0.057164682160441425), np.float64(0.05767462839990464), np.float64(0.056829760693000025), np.float64(0.05669379340573526), np.float64(0.05506282882618079), np.float64(0.055253226872170214), np.float64(0.05661145620294529), np.float64(0.055858865083740286), np.float64(0.05614582611726307), np.float64(0.05703953154713377), np.float64(0.055381300568486794), np.float64(0.05664191395733729), np.float64(0.055292840482667395), np.float64(0.05582765559389233), np.float64(0.05757020777719298), np.float64(0.057106410587275246), np.float64(0.05697236003023561), np.float64(0.056537031685305904), np.float64(0.05662838162581571), np.float64(0.05544173618716097), np.float64(0.056268569438745523), np.float64(0.05628389656961499), np.float64(0.057319117712794763), np.float64(0.05692404953261187), np.float64(0.056753592987283864), np.float64(0.05675209851089123), np.float64(0.05649379626234024), np.float64(0.055699140919344865), np.float64(0.05675376043959545), np.float64(0.055843036320618135), np.float64(0.05713121318012734), np.float64(0.05643674941485308), np.float64(0.057367160250643354), np.float64(0.05792127909628846), np.float64(0.05740232683282275), np.float64(0.05647552371031103), np.float64(0.05810057877471471), np.float64(0.05727740920687817), np.float64(0.05566760109686546), np.float64(0.05620338772229981), np.float64(0.05577257239850172), np.float64(0.057344581379051136), np.float64(0.057530390551207615), np.float64(0.05596569836945469), np.float64(0.05494745592857183), np.float64(0.056886016237972604), np.float64(0.05533328640618031), np.float64(0.054755190600086906), np.float64(0.056324932150080824), np.float64(0.05529193301874204), np.float64(0.0561290978116836), np.float64(0.054841065828480653), np.float64(0.05523005310441994), np.float64(0.05637493950379965), np.float64(0.0572882716873305), np.float64(0.05797918652482775), np.float64(0.05654635171328225), np.float64(0.0562006868789668), np.float64(0.05769364872475652), np.float64(0.05661095221837835), np.float64(0.05545319956741678), np.float64(0.055823982472837874), np.float64(0.05538738693420129), np.float64(0.055848595608122775), np.float64(0.05545840000864267), np.float64(0.05462506176736759), np.float64(0.05626079353116332), np.float64(0.05601328915817863), np.float64(0.05473448120456313), np.float64(0.05439223274392063), np.float64(0.05559091644523822), np.float64(0.05641152554756001), np.float64(0.055211580249238626), np.float64(0.05706425533595497), np.float64(0.056162644321021804), np.float64(0.055839902505091694), np.float64(0.05491149582458668), np.float64(0.056265780376255084), np.float64(0.05723738609610023), np.float64(0.053994926938257345), np.float64(0.05295687121143856), np.float64(0.05640363764052453), np.float64(0.05689582297552738), np.float64(0.05631269742325567), np.float64(0.05696472412618029), np.float64(0.05621367524878262), np.float64(0.05762697166146173), np.float64(0.056148711154467845), np.float64(0.05438349405229402), np.float64(0.056981021069109516), np.float64(0.05526415605796197), np.float64(0.057570928459168676), np.float64(0.05753800509171204), np.float64(0.05802533704059997), np.float64(0.056671498536925526), np.float64(0.05848600511609397), np.float64(0.056107368036562175), np.float64(0.05741419643502187), np.float64(0.057535821059797695), np.float64(0.05710073439307112), np.float64(0.0578224389905976), np.float64(0.058064752975445444), np.float64(0.057517328333126366), np.float64(0.05623789626778969), np.float64(0.05800110391619457), np.float64(0.058103268966756196), np.float64(0.058140139610551346), np.float64(0.057005988066940766), np.float64(0.05903742503039286), np.float64(0.058542369440939504), np.float64(0.05847991549481927), np.float64(0.05750604592331828), np.float64(0.05815634571528136), np.float64(0.05722550827503597), np.float64(0.05756247028681744), np.float64(0.05753709272939341), np.float64(0.05742283910814241), np.float64(0.05711120388480315), np.float64(0.05640901446501298), np.float64(0.05698817480784967), np.float64(0.05751770093144956), np.float64(0.057272747930560086), np.float64(0.058407534847351554), np.float64(0.058086193935230665), np.float64(0.05912170880181953), np.float64(0.05922669951028302), np.float64(0.05802198398574082), np.float64(0.0589968401612903), np.float64(0.05844157778389053), np.float64(0.05983968602512199), np.float64(0.059259509273222606), np.float64(0.05898754568081268), np.float64(0.059158987917783235), np.float64(0.06009514008297701), np.float64(0.05951505995388257), np.float64(0.058717105177021826), np.float64(0.058162027503113875), np.float64(0.05853188718575662), np.float64(0.05876123576741418), np.float64(0.058673065407197376), np.float64(0.058481589448899134), np.float64(0.05899640166770944), np.float64(0.059305872596668), np.float64(0.05897022014228412), np.float64(0.05937996672513232), np.float64(0.060228402193067215), np.float64(0.059395157844858305), np.float64(0.06040135395441351), np.float64(0.059119970875791454), np.float64(0.05907238100942954), np.float64(0.059293888959502654), np.float64(0.059253739399863385), np.float64(0.05882746369977689), np.float64(0.06032916538915579), np.float64(0.05950702656870302), np.float64(0.060013839508096356), np.float64(0.05834955101313338), np.float64(0.05881482328214221), np.float64(0.05917975340079543), np.float64(0.05838346167497128), np.float64(0.05867299726272692), np.float64(0.05960879397231985), np.float64(0.0602220919566039), np.float64(0.06286487311536144), np.float64(0.06233724909135259), np.float64(0.060883690088757575), np.float64(0.060763870339725634), np.float64(0.06187130766766055), np.float64(0.06222870537370287), np.float64(0.06136855648793678), np.float64(0.06020094863361556), np.float64(0.06098821058110053), np.float64(0.05836897130589678), np.float64(0.06129533767968062), np.float64(0.06065438483432929), np.float64(0.06163904332569582), np.float64(0.06185119026600347), np.float64(0.0619151776004596), np.float64(0.06114640560775888), np.float64(0.058416269953602606), np.float64(0.060634090925318476), np.float64(0.05889334798817755), np.float64(0.06054720330607461), np.float64(0.06048370714906686), np.float64(0.06001112891604863), np.float64(0.05787367278085582), np.float64(0.06204309114625725), np.float64(0.06113255513189623), np.float64(0.05871660413701188), np.float64(0.05969164389142824), np.float64(0.05887106030724016), np.float64(0.05901833973107471), np.float64(0.05926884040109554), np.float64(0.05973085703343219), np.float64(0.059359360601158415), np.float64(0.058653477717923834), np.float64(0.060835936536062764), np.float64(0.05862587375311947), np.float64(0.05934582241823419), np.float64(0.058077273903864586), np.float64(0.059056457750125024), np.float64(0.060905607803910786), np.float64(0.06149979186422889), np.float64(0.0615057390117872), np.float64(0.059893946597870056), np.float64(0.05891564462773353), np.float64(0.0606661851132263), np.float64(0.05848506942747636), np.float64(0.05833145485254005), np.float64(0.05990559253643302), np.float64(0.061075563540646464), np.float64(0.06111506445101552), np.float64(0.06178319154352684), np.float64(0.06194138961369899), np.float64(0.061468688818726885), np.float64(0.059050775191045214), np.float64(0.06072276392799175), np.float64(0.060579762026440706), np.float64(0.05982158689408221), np.float64(0.06047649064092877), np.float64(0.061328193198218695), np.float64(0.06291911995379042), np.float64(0.06093291488932209), np.float64(0.05999402077017778), np.float64(0.06315021783339213), np.float64(0.06086956910985763), np.float64(0.06175406788047777), np.float64(0.060768517768603134), np.float64(0.0595278759261026), np.float64(0.061818977531642366), np.float64(0.06150833274972427), np.float64(0.06152435108805416), np.float64(0.061284646978216104), np.float64(0.060539214437595044), np.float64(0.05925653402119471), np.float64(0.06016933719927011), np.float64(0.060791697458180444), np.float64(0.06013444657176466), np.float64(0.05903587041074153), np.float64(0.062094703038933226), np.float64(0.0618661643000462), np.float64(0.061582854943254495), np.float64(0.05901320222009183), np.float64(0.0620340587402737), np.float64(0.06026115080445475), np.float64(0.05989346689487187), np.float64(0.06021590125488049), np.float64(0.06114106852350742), np.float64(0.06345300473598171), np.float64(0.0607176227298066), np.float64(0.06142486312539071), np.float64(0.060055073154876855), np.float64(0.05954738677446727), np.float64(0.06044021256226301), np.float64(0.06209773750491367), np.float64(0.05980953872987357), np.float64(0.061003367978182156), np.float64(0.06136888549704876), np.float64(0.0630937809177266), np.float64(0.06248764280144318), np.float64(0.06132933815039166), np.float64(0.06044930236007565), np.float64(0.060596649740523416), np.float64(0.06072051596390182), np.float64(0.06250663152934119), np.float64(0.06091702404662096), np.float64(0.06241608934211783), np.float64(0.06120977415382703), np.float64(0.06133483840142539), np.float64(0.06179426288992831), np.float64(0.06241453950856248), np.float64(0.061461317405080304), np.float64(0.06055219400319648), np.float64(0.060614346155676735), np.float64(0.0613905033918282), np.float64(0.06254692449789855), np.float64(0.06058233367154274), np.float64(0.06274130097350426), np.float64(0.062248416547589946), np.float64(0.0609152665389393), np.float64(0.06327449770858717), np.float64(0.06238096044304416), np.float64(0.06215889263821144), np.float64(0.06237593611879449), np.float64(0.058210550496290286), np.float64(0.06320415989703594), np.float64(0.06332857104170748), np.float64(0.062358164464752515), np.float64(0.06163351521227316), np.float64(0.06027006412659485), np.float64(0.05986799574419954), np.float64(0.06035007443484082), np.float64(0.059588554043393055), np.float64(0.061616173909225226), np.float64(0.061781826658077056), np.float64(0.0625274034128002), np.float64(0.06040156025840096), np.float64(0.061093054885831326), np.float64(0.06291193385137105), np.float64(0.06402886590757499), np.float64(0.06376970329007492), np.float64(0.06397251101400825), np.float64(0.0627652238994252), np.float64(0.061885495110182265), np.float64(0.06145798901124809), np.float64(0.06209875588947586), np.float64(0.06270779856569599), np.float64(0.06117358329301946), np.float64(0.06351737945297409), np.float64(0.06280087544955999), np.float64(0.06385112067491411), np.float64(0.06355278698326865), np.float64(0.06188706047547282), np.float64(0.06330004548384392), np.float64(0.06234026037095243), np.float64(0.061622375064297144), np.float64(0.06290194928753544), np.float64(0.06299229535365579), np.float64(0.06434365778384901), np.float64(0.06411595039483221), np.float64(0.06206422731733071), np.float64(0.06199085652931741), np.float64(0.06314269776597455), np.float64(0.06220563411492581), np.float64(0.06271249578983452), np.float64(0.06317936433077269), np.float64(0.06258046415263696), np.float64(0.06257079301083396), np.float64(0.061936854290717665), np.float64(0.06221045231765801), np.float64(0.06441572139800258), np.float64(0.06335811797923396), np.float64(0.06393918054771243), np.float64(0.06309912482644366), np.float64(0.06287843492239119), np.float64(0.061755872166568276), np.float64(0.06293223227159772), np.float64(0.0636531375575122), np.float64(0.06360533963433061), np.float64(0.061223313949376405), np.float64(0.062428585237113345), np.float64(0.06185945682649493), np.float64(0.0634672983990783), np.float64(0.06149582910260101), np.float64(0.05976840084846049), np.float64(0.06032430708347911), np.float64(0.062183855338164756), np.float64(0.06143712038944584), np.float64(0.05990837732103943), np.float64(0.05955922643775371), np.float64(0.06126040088361176), np.float64(0.06251844023146634), np.float64(0.060993949969645786), np.float64(0.06195424353508771), np.float64(0.06202560067136469), np.float64(0.062008620335099865), np.float64(0.05865819373995333), np.float64(0.06225582363768987), np.float64(0.06055807638948091), np.float64(0.061167845765735716), np.float64(0.0623777271816153), np.float64(0.06314174577341439), np.float64(0.06396179751721454), np.float64(0.062372394477697594), np.float64(0.06361551671636118), np.float64(0.06116771450337998), np.float64(0.060504052660002465), np.float64(0.0621758781688685), np.float64(0.06381864296799371), np.float64(0.06159539788437442), np.float64(0.06412839795301539), np.float64(0.06398827571542709), np.float64(0.0641920947785837), np.float64(0.06293413962078642), np.float64(0.06299825128525657), np.float64(0.062354400793578624), np.float64(0.0640474575155543), np.float64(0.06249180456871144), np.float64(0.06199206771916938), np.float64(0.0627460289176007), np.float64(0.06482039376463872), np.float64(0.06401738350629381), np.float64(0.06450898231356349), np.float64(0.06460394045625892), np.float64(0.06409369989801175), np.float64(0.06758014856003877), np.float64(0.06761784779155822), np.float64(0.06800585658412302), np.float64(0.06966839456775863), np.float64(0.06711628729738198), np.float64(0.06811313936551347), np.float64(0.0668001652731354), np.float64(0.06773594129733948), np.float64(0.0666094050648784), np.float64(0.06757549730558067), np.float64(0.06635939621429103), np.float64(0.06954101573825884), np.float64(0.06674537261060001), np.float64(0.0684602180887254), np.float64(0.0672015625689086), np.float64(0.06862327614679224), np.float64(0.06800434430087562), np.float64(0.06690502325630435), np.float64(0.06804868506977739), np.float64(0.06814700083668744), np.float64(0.06759423408132646), np.float64(0.06861637604791239), np.float64(0.06998456486637909), np.float64(0.06737969344457909), np.float64(0.06747730017092256), np.float64(0.06975105560456964), np.float64(0.06955790684977342), np.float64(0.06997313864546313), np.float64(0.07031596940413978), np.float64(0.06815712243622546), np.float64(0.07032396374140538), np.float64(0.07101696111022239), np.float64(0.0684684045502786), np.float64(0.06637162717823232), np.float64(0.06849091125232051), np.float64(0.06918212571821582), np.float64(0.06996830548306981), np.float64(0.07076934647891435), np.float64(0.06817978757169023), np.float64(0.06917618710277593), np.float64(0.07106793754199504), np.float64(0.06990704297043795), np.float64(0.07115702953988508), np.float64(0.06873840949648058), np.float64(0.06804738522484445), np.float64(0.06989116936359469), np.float64(0.07057341738339656), np.float64(0.06886853570685555), np.float64(0.07048370458361633), np.float64(0.06884584688699229), np.float64(0.06888582902117374), np.float64(0.06957752403556677), np.float64(0.06813871402237383), np.float64(0.0676030195354377), np.float64(0.07011713622739045), np.float64(0.06980117162419251), np.float64(0.06983522585357907), np.float64(0.0680802965676936), np.float64(0.06941754073028611), np.float64(0.06889172356404463), np.float64(0.07037468351327249), np.float64(0.06895210053600467), np.float64(0.06918119747817918), np.float64(0.0693940803710916), np.float64(0.07060502386642756), np.float64(0.0713835577317631), np.float64(0.07066764046498705), np.float64(0.06975646374284344), np.float64(0.07105080455430642), np.float64(0.06831402571919754), np.float64(0.07157764655973121), np.float64(0.06692442379799186), np.float64(0.064826923101721), np.float64(0.06343778619983506), np.float64(0.06444369996912), np.float64(0.06542411652497182), np.float64(0.0641646406824176), np.float64(0.06596651640013079), np.float64(0.06472515337616132), np.float64(0.06689479491747703), np.float64(0.06505348770793842), np.float64(0.06167879308959577), np.float64(0.06450857144600096), np.float64(0.062139225307530695), np.float64(0.06548725589956642), np.float64(0.06677847680372292), np.float64(0.0644137217219174), np.float64(0.06592178038315187), np.float64(0.06725618244880892), np.float64(0.06706745457521168), np.float64(0.06839271156133142), np.float64(0.06823259098110637), np.float64(0.06648403740338862), np.float64(0.0656800156402355), np.float64(0.06597186885074313), np.float64(0.0660181774774337), np.float64(0.06568499862559417), np.float64(0.06815824980424004), np.float64(0.06641820303382119), np.float64(0.06863631557297546), np.float64(0.06590593811997854), np.float64(0.06620563374190055), np.float64(0.06947787868697622), np.float64(0.06725912179251023), np.float64(0.0673664245104244), np.float64(0.06798988170545615), np.float64(0.06803098632910702), np.float64(0.06572926922885002), np.float64(0.06787838911977574), np.float64(0.06566604141608846), np.float64(0.06693255649952075), np.float64(0.06541969860262609), np.float64(0.06875032038440072), np.float64(0.06920421489066533), np.float64(0.06478720818506678), np.float64(0.06779385863455298), np.float64(0.06736490944587831), np.float64(0.06562807410767235), np.float64(0.06735750511887252), np.float64(0.06674225271364125), np.float64(0.06747783371922493), np.float64(0.0679525341362102), np.float64(0.06786892649997704), np.float64(0.06530363243090748), np.float64(0.06772860918926071), np.float64(0.0654182359739667), np.float64(0.06612983397323254), np.float64(0.06535574500240134), np.float64(0.06552422307250677), np.float64(0.06260996936659242), np.float64(0.06484680959708822), np.float64(0.06742773586938597), np.float64(0.06563768927553279), np.float64(0.06558762030451845), np.float64(0.06502411769931964), np.float64(0.06546434414563575), np.float64(0.06609216833069068), np.float64(0.06546598793676484), np.float64(0.06475948675313148), np.float64(0.0644017590905361), np.float64(0.06628444804025162), np.float64(0.06839469426153505), np.float64(0.06696078809662605), np.float64(0.06552437433669357), np.float64(0.06889686415570509), np.float64(0.06675531217202096)]
        
        ## Here is an example popt_k and popt_m value found during fit model development. These numbers will change depending on the fitting; the calculated values above will not.
        ## (popt_k, popt_m) = (array([ 9.50000000e-01, 2.87401654e-02,  6.95964092e-04]), array([ 7.21652034e-03, 1.93964230e-01,  1.09845293e-06]))
    
    # Conversions into numpy.
    k        = np.array(list_of_slopes)
    m        = np.array(list_of_offsets)
    k_err    = np.array(list_of_slopes_err)
    m_err    = np.array(list_of_offsets_err)
    n        = np.array(list_of_n_samples)
    rms_dev  = np.array(list_of_rms_deviations)
    rms_serr = np.array(list_of_rms_standard_errors)
    
    ## Plot!
    
    # Get time axis, i.e., x-axis.
    time = np.array(take_relaxation_data_at_this_time_interval_s)
    
    # How many datapoints must we have in the fitted point to qualify
    # as a reliable metric? → enforce_n
    if enforce_n != -1:
        # Check descending order
        for i in range(1, len(n)):
            if n[i] > n[i - 1]:
                print(f"Warning: 'n' is not in descending order at index {i}: "
                      f"n[{i-1}] = {n[i-1]}, n[{i}] = {n[i]}")
    # Enforce n-threshold for reported data accuracy.
    for i in range(len(n)):
        if n[i] < enforce_n:
            k[i] = None
            m[i] = None
            k_err[i] = None
            m_err[i] = None
            time[i]  = None
    
    # Plotting
    if plot_RMS_deviation:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25.31, 10), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25.31, 10), sharex=True)

    # Top plot: slope
    ax1.errorbar(time/3600, k, yerr=k_err, fmt='o', markersize=5, ecolor='gray', capsize=2, label='Slope $k(t\')$', color="#EE1C1C")
    #ax1.set_xlabel('Relaxation time [h]', fontsize=33)
    ax1.set_xlabel('Time after manipulation [h]', fontsize=33)
    ax1.set_ylabel('Linear fit slope [-]', fontsize=33)
    #ax1.set_title('Slope and Offset vs Time')
    ax1.tick_params(axis='both', labelsize=26)
    ax1.set_ylim(0,1.31)
    ax1.grid(True)
    ax1.legend(fontsize=26, loc='lower right')

    # Bottom plot: offset
    ax2.errorbar(time/3600, m, yerr=m_err, fmt='o', markersize=5, ecolor='gray', capsize=2, label='Offset $m(t\')$', color="#1CEE70")
    ax2.tick_params(axis='both', labelsize=26)
    ax2.set_ylim(0,4.1)
    #ax2.set_xlabel('Relaxation time [h]', fontsize=33)
    ax2.set_xlabel('Time after manipulation [h]', fontsize=33)
    ax2.set_ylabel('Linear fit offset [%]', fontsize=33)
    ax2.grid(True)
    ax2.legend(fontsize=26, loc='lower right')
    
    # Third plot: RMS deviation
    if plot_RMS_deviation:
        ##ax3.plot(time/3600, rms_dev, 'o-', color="#1C70EE", label='RMS deviation')
        ax3.errorbar(time/3600, rms_dev, yerr=rms_serr, fmt='o', markersize=5, ecolor='gray', capsize=2, label='RMS deviation', color="#1C70EE")
        ax3.tick_params(axis='both', labelsize=26)
        ax3.set_xlabel('Relaxation time [h]', fontsize=33)
        ax3.set_ylabel('Deviation from fit [%]', fontsize=33)
        ax3.set_ylim(0,4.1)
        ax3.grid(True)
        ax3.legend(fontsize=26, loc='lower right')
    
    ## Perform ln-fits and add these to plots.
    def log_growth_tau(t, a, b, tau):
        return a + b * np.log(1 + t / tau)
    #def log_growth(t, a, b):
    #    return a + b * np.log(t)
    ####def log_decay(t, a, b, c):
    ####    return a + b * np.log(t + c)
    #def power_growth(t, a, c, alpha):
    #    return c * (t**alpha) + a

    # After k, m, etc. are computed and converted to numpy arrays:
    time_hours = time / 3600  # Use hours for plotting

    # Fit slope (k)
    mask_k = ~np.isnan(k) & (time_hours > 0)
    p0 = [k[mask_k][0], 0.1, 0.1]  # start value + small slope
    popt_k, pcov_k = curve_fit(
        log_growth_tau,
        time_hours[mask_k],
        k[mask_k],
        p0=p0,
        bounds=([1.0, -np.inf, 1e-10], [1.05, np.inf, np.inf]),
        maxfev=20000
    )
    #fit_k = power_growth(time_hours, *popt_k)
    fit_k = log_growth_tau(time_hours, *popt_k)
    ax1.plot(
        time_hours,
        fit_k,
        '-.',
        color='black',
        lw=2,
        label=(
            f"Fit, $k(t') = a + b\\cdot \\ln(t'/\\tau)$:\n"
            f"$a={popt_k[0]:.2f},\\ "
            f"b={popt_k[1]:.2f},\\ "
            f"\\tau={popt_k[2]*3600:.1f}\\,\\mathrm{{s}}$"
        ),
        zorder=10
    )
    print("Optimal k parameters:", popt_k)
    
    # Fit offset (m)
    mask_m = ~np.isnan(m) & (time_hours > 0)
    if np.sum(mask_m) > 4:  # Need enough points
        popt_m, pcov_m = curve_fit(
            log_growth_tau,
            time_hours[mask_m],
            m[mask_m],
            p0=[0.05, 0.1, 0.1],
            bounds=([0, -1, 1e-8], [0.1, np.inf, np.inf]),
            maxfev=20000
        )
        #fit_m = power_growth(time_hours, *popt_m)
        fit_m = log_growth_tau(time_hours, *popt_m)
        ax2.plot(
            time_hours,
            fit_m,
            '-.',
            color='black',
            lw=2,
            label=(
                f"Fit, $m(t') = a + b\\cdot \\ln(t'/\\tau)$:\n"
                f"$a={popt_m[0]:.2f}\\,\\mathrm{{\\%}},\\ "
                f"b={popt_m[1]:.2f}\\,\\mathrm{{\\%}},\\ "
                f"\\tau={popt_m[2]*3600000:.1f}\\,\\mathrm{{ms}}$"
            ),
            zorder=10
        )
        print("Optimal m parameters:", popt_m)
    
    # Calculate deviation from latest fit.
    deviation_k = k[mask_k] - log_growth_tau(time_hours[mask_k], *popt_k)
    deviation_m = m[mask_m] - log_growth_tau(time_hours[mask_m], *popt_m)
    mean_devk = np.mean(deviation_k)
    mean_devm = np.mean(deviation_m)
    variance_devk = np.var(deviation_k)
    variance_devm = np.var(deviation_m)
    
    print("Mean and variance of k-fit-deviation: "+str(mean_devk)+", "+str(variance_devk))
    print("Mean and variance of m-fit-deviation: "+str(mean_devm)+", "+str(variance_devm))
    
    # Plot latest fit error.
    '''fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(25.31, 10), sharex=True)
    ax4.plot(time_hours[mask_k], deviation_k)
    ax5.plot(time_hours[mask_m], deviation_m)
    ax4.set_xlabel("Time [h]")
    ax5.set_xlabel("Time [h]")
    ax4.set_ylabel("Linear fit slope deviation")
    ax5.set_ylabel("Linear fit offset deviation")'''
    
    # Update legends to include fits
    ax1.legend(fontsize=26, loc='lower right')
    ax2.legend(fontsize=26, loc='lower right')
    
    # Tight layout!
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.135)
    
    # Save plot?
    if savepath != '':
        fig.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show!
    plt.show()
    
    # Return!
    return list_of_slopes, list_of_offsets, list_of_slopes_err, list_of_offsets_err, list_of_n_samples, list_of_rms_deviations, list_of_rms_standard_errors, (popt_k, popt_m), (deviation_k, deviation_m)


def plot_active_vs_total_resistance_gain(
    title_voltage_V,
    title_junction_size_nm,
    folder_path,
    take_relaxation_data_at_this_time_s,
    filename_tags = [],
    outlier_threshold_in_std_devs = 2.0,
    highlight_outliers = False,
    plot_ideal_curve = False,
    colourise = False,
    savepath = '',
    plot = True,
    legend_string = 'Manipulated junctions'
    ):
    ''' Given some set of data, plot the active gain on the X axis in percent,
        and the total gain on the Y axis in percent relative to the
        initial resistance point. Also, make a fit.
        
        folder_path: path to folder containing measurement files.
        
        take_relaxation_data_at_this_time_s: time in seconds where the datapoint
        will be taken for the resistance manipulation. The point closest
        in time to the user-supplied value will be chosen.
        
        filename_tags: defines substrings that must be found in the filename
        for the file to be considered for data inclusion. If blank, then
        consider all files in the folder.
        
        Datapoints that fall outside of outlier_threshold_in_std_devs are
        considered outliers, this limit was set to 2 standard deviations
        from the mean. The argument highlight_outliers sets whether
        to plot a second trace that fits to data that is not an outlier.
        For the paper, all data was included. This highlight_outliers
        is just for in-situ analysis of the experiments themselves.
    '''
    
    # Counter to keep track of colourised patterns.
    colourise_counter = 0
    
    # Detect special junction sizes for colourisation?
    dot_colour = "#D63834"
    if title_junction_size_nm == 200:
        dot_colour = "#EE1C1C"
    elif title_junction_size_nm == 350:
        dot_colour = "#1CEE70"
    elif title_junction_size_nm == 318:
        dot_colour = "#1C70EE"
    elif title_junction_size_nm == 354:
        dot_colour = "#C41CEE"
    
    # Acquire dataset.
    ## Rework voltage into something that will be written in the title.
    ## Append this number to the filename_tags to look for.
    filename_tags.append(str(title_voltage_V).replace('.',"p"))
    (active_gain_percent, total_gain_percent) = acquire_relaxation_data_from_folder(
        folder_path = folder_path,
        take_relaxation_data_at_this_time_s = take_relaxation_data_at_this_time_s,
        filename_tags = filename_tags,
        verbose = plot,
    )
    
    ##  Here are the two datasets for R01 and R02, i.e., the 354x354nm^2  ##
    ##  and 318x318nm^2 junction sizes from the JJTest100W3 wafer.        ##
    ##  That is, 354 is 'High-dose 2', and 318 is 'High-dose 1'.
    #active_gain_percent  = [2.45,  0.71,  10.05,  1.79,  5.02, 0.61, 3.52, 0.77, 1.01,  8.26, 2.39, 4.22, 4.15,  9.02,  7.71, 2.58, 1.77, 16.51, 5.13, 6.07, 11.03, 7.01, 0.28, 8.11, 2.75, 12.53, 13.16, 5.42, 1.60, 1.94, 4.97522988,  6.17057683, 7.01116346] ## THICK354
    #total_gain_percent   = [3.587, 2.180, 11.969, 2.999, 7.30, 2.76, 6.76, 2.17, 3.06, 10.55, 4.61, 5.70, 6.10, 10.85, 11.78, 6.45, 5.91, 19.07, 6.61, 8.23, 13.13, 8.93, 1.77, 9.74, 5.66, 14.68, 15.84, 7.13, 2.75, 3.36, 6.99326582, 10.37398798, 9.57568379] ## THICK354
    #active_gain_percent = [7.0942, 10.0738, 11.8119, 5.0343, 15.0296, 18.0029, 20.0640, 6.0434,  9.0186,  7.6048, 4.040, 10.007, 4.060, 6.020, 21.129, 3.577, 20.741, 3.043, 1.017, 0.78152682, 1.08007167, 2.1641953,  3.4114392 ] ## THICK318
    #total_gain_percent  = [9.7949, 12.2327, 14.0164, 6.9078, 17.4735, 20.3925, 22.1454, 8.2411, 11.1680, 10.2548, 5.986, 12.063, 5.804, 7.871, 23.058, 5.378, 23.186, 4.797, 2.859, 2.51228217, 4.02749426, 4.87928191, 6.21816964] ## THICK318
    
    # Sort lists together based on the active gain list.
    sorted_active, sorted_total = zip(*sorted(zip(active_gain_percent, total_gain_percent)))
    
    # Convert to numpy arrays!
    sorted_active = np.array(sorted_active, dtype=np.float64)
    sorted_total  = np.array(sorted_total, dtype=np.float64)
    
    # Report!
    if plot:
        print("Sorted active:", sorted_active)
        print("Sorted total:",  sorted_total)
    
    # Ensure the data sets are of identical length.
    if (len(sorted_active) != len(sorted_total)):
        raise ValueError("Error! The provided data sets do not have matching lengths; len(x) != len(y)")
        
    # Initial linear regression
    slope, intercept, r_value, p_value, std_err = linregress(sorted_active, sorted_total)
    
    # Compute residuals
    y_predicted = slope * sorted_active + intercept
    residuals   = sorted_total - y_predicted
    
    # Set threshold for outliers.
    threshold = outlier_threshold_in_std_devs * np.std(residuals)
    mask = np.abs(residuals) < threshold
    
    # Filter out outliers?
    active_filtered = sorted_active[mask]
    total_filtered  = sorted_total[mask]
    
    # Create figure for plotting.
    if plot:
        if colourise:
            fig, ax = plt.subplots(figsize=(12.8, 10), facecolor=get_colourise(-2))
            #plt.figure(figsize=(10, 5), facecolor=get_colourise(-2))
        else:
            fig, ax = plt.subplots(figsize=(12.5, 9.14))
            #plt.figure(figsize=(10, 5))
    
    # Let's fit the data and see what we get.
    def linear_func(x, k, m):
        return k * x + m
    
    def linear_fitter(
        sorted_active,
        sorted_total
        ):
        ## Let's guess initial guessing values.
        k_guess = (sorted_total[-1] - sorted_total[0]) / (sorted_active[-1] - sorted_active[0])
        m_guess = sorted_total[0]
        ## Fit!
        optimal_vals, covariance_mtx_of_opt_vals = curve_fit(
            f     = linear_func,
            xdata = sorted_active,
            ydata = sorted_total,
            p0    = (k_guess, m_guess)
        )
        
        # Extract parameters.
        optimal_k = optimal_vals[0]
        optimal_m = optimal_vals[1]
        
        # Get the fit error.
        fit_err = np.sqrt(np.diag(covariance_mtx_of_opt_vals))
        err_k = fit_err[0]
        err_m = fit_err[1]
        
        # Get a fit curve! Extend to zero, i.e., insert zero.
        fitted_curve = linear_func(
            x = sorted_active,
            k = optimal_k,
            m = optimal_m
        )
        
        # Return!
        return fitted_curve, optimal_k, optimal_m, err_k, err_m
    
    # Plot the fit curve!
    ## Extend to zero!
    original_fitted_curve, optimal_k, optimal_m, err_k, err_m = linear_fitter(sorted_active, sorted_total)
    if plot:
        if colourise:
            plt.plot(np.insert(sorted_active, 0, 0), np.insert(original_fitted_curve, 0, optimal_m), color=get_colourise(colourise_counter), label=f"Linear fit: {optimal_k:.2f} · x + {optimal_m:.2f} [%]\nSlope error: ±{err_k:.2f}\nOffset error: ±{err_m:.2f}")
            colourise_counter += 1
        else:
            if savepath == '':
                plt.plot(np.insert(sorted_active, 0, 0), np.insert(original_fitted_curve, 0, optimal_m), color="#34D2D6", label=f"Linear fit: {optimal_k:.2f} · x + {optimal_m:.2f} [%]\nSlope error: ±{err_k:.2f}\nOffset error: ±{err_m:.2f}")
            else:
                plt.plot(np.insert(sorted_active, 0, 0), np.insert(original_fitted_curve, 0, optimal_m), color="#222222", label=f"Linear fit: {optimal_k:.2f} · x + {optimal_m:.2f} [%]\nSlope error: ±{err_k:.2f}\nOffset error: ±{err_m:.2f}")
        
    if highlight_outliers:
        # Do the same with the filtered data.
        ## Remember to extend to 0.
        filtered_fitted_curve, filtered_k, filtered_m, filtered_err_k, filtered_err_m = linear_fitter(active_filtered, total_filtered)
        if plot:
            if colourise:
                plt.plot(np.insert(active_filtered, 0, 0), np.insert(filtered_fitted_curve, 0, filtered_m), color=get_colourise(colourise_counter), label=f"Trimmed fit: {filtered_k:.2f} · x + {filtered_m:.2f} [%]\nSlope error: ±{filtered_err_k:.2f}\nOffset error: ±{filtered_err_m:.2f}")
                colourise_counter += 1
            else:
                plt.plot(active_filtered, filtered_fitted_curve, color="#81D634", label=f"Trimmed fit: {filtered_k:.2f} · x + {filtered_m:.2f} [%]\nSlope error: ±{filtered_err_k:.2f}\nOffset error: ±{filtered_err_m:.2f}")
        
    # Plot an ideal curve?
    if plot:
        if plot_ideal_curve:
            if highlight_outliers:
                plt.plot(np.insert(sorted_active, 0, 0), 1.00 * np.insert(sorted_active, 0, filtered_m) + filtered_m, color="#000000", label=f"Ideal trend: 1.0000 · x + {optimal_m:.2f} [%]")
            else:
                plt.plot(np.insert(sorted_active, 0, 0), 1.00 * np.insert(sorted_active, 0, optimal_m) + optimal_m, color="#000000", label=f"Ideal trend: 1.0000 · x + {optimal_m:.2f} [%]")
        
    
    # Insert datapoints!
    if plot:
        if (not colourise):
            if highlight_outliers:
                plt.scatter(sorted_active, sorted_total, color="#8934D6", label=str(outlier_threshold_in_std_devs)+"σ outliers")
                ## This is a weird way to do this plotting, but it ensures that
                ## only the outliers are highlighted in the plot legend.
                ## Even though this dataset is the original! → Result = Good legend!
                plt.scatter(active_filtered, total_filtered, color="#D63834")
            else:
                plt.scatter(sorted_active, sorted_total, color=dot_colour, label=legend_string)
        else:
            if highlight_outliers:
                plt.scatter(sorted_active, sorted_total, color=get_colourise(3), label=str(outlier_threshold_in_std_devs)+"σ outliers")
                ## This is a weird way to do this plotting, but it ensures that
                ## only the outliers are highlighted in the plot legend.
                ## Even though this dataset is the original! → Result = Good legend!
                plt.scatter(active_filtered, total_filtered, color=get_colourise(2))
            else:
                plt.scatter(sorted_active, sorted_total, color=get_colourise(2))
        
        # Labels and such.
        plt.grid()
        if colourise:
            fig.patch.set_alpha(0)
            ax.grid(color=get_colourise(-1))
            ax.set_facecolor(get_colourise(-2))
            ax.spines['bottom'].set_color(get_colourise(-1))
            ax.spines['top'].set_color(get_colourise(-1))
            ax.spines['left'].set_color(get_colourise(-1))
            ax.spines['right'].set_color(get_colourise(-1))
            ax.tick_params(axis='both', colors=get_colourise(-1))
        
        # Bump up the size of the ticks' numbers on the axes.
        ax.tick_params(axis='both', labelsize=26)
        
        # Extend axes to include the origin?
        if np.all(sorted_active >= 0):
            ax.set_xlim(xmin=0, xmax=26.35)
        if np.all(sorted_total >= 0):
            ax.set_ylim(ymin=0, ymax=36.0)
        
        # Fancy colours?
        if (not colourise):
            plt.xlabel("Active manipulation [%]", fontsize=33)
            plt.ylabel("Total manipulation [%]", fontsize=33)
            #plt.title(f"Active vs. total manipulation\n30 minutes after stopping\n±{title_voltage_V:.2f} V, {title_junction_size_nm}x{title_junction_size_nm} nm", fontsize=38)
            if savepath == '':
                plt.title(f"Active vs. total manipulation\n±{title_voltage_V:.2f} V, {title_junction_size_nm}x{title_junction_size_nm} nm", fontsize=38)
        else:
            plt.xlabel("Active manipulation [%]", color=get_colourise(-1), fontsize=33)
            plt.ylabel("Total manipulation [%]",  color=get_colourise(-1), fontsize=33)
            print("WARNING: CHANGE BACK")
            if savepath == '':
                plt.title(f"Active vs. total manipulation\n±{title_voltage_V:.2f} V, {title_junction_size_nm}x{title_junction_size_nm} nm", color=get_colourise(-1), fontsize=38)
        
        ## # Bump up legend.
        ## plt.legend(fontsize=26, loc='lower right')
        
        # Get current legend handles and labels.
        handles, labels = plt.gca().get_legend_handles_labels()

        # Reorder them so the scatter entry (like 'Low-dose 1') comes first.
        order = np.argsort([0 if legend_string in l else 1 for l in labels])
        handles = [handles[i] for i in order]
        labels = [labels[i] for i in order]

        # Now draw the legend.
        plt.legend(handles, labels, fontsize=26, loc='lower right')
        
        # Save plots?
        if savepath != '':
            plt.tight_layout()
            # Fix path name.
            base, _ = os.path.splitext(savepath)
            new_savepath = f"{base}_{title_junction_size_nm}nm.png"
            plt.savefig(new_savepath, dpi=164, bbox_inches='tight')
            
        # Show shits.
        plt.show()
    
    # How many samples were in the set?
    n_samples = len(sorted_active)
    
    # Let's calculate the RMS deviation.
    ## TODO: filtered_xx currently not supported here.
    
    ## Non-weighted RMS version here.
    residuals = sorted_total - (optimal_k * sorted_active + optimal_m)
    rms_deviation = np.sqrt(np.mean(residuals**2))
    rms_se = rms_deviation / np.sqrt(2 * n_samples)
    return optimal_k, optimal_m, err_k, err_m, n_samples, rms_deviation, rms_se
    ## Weighted RMS version here.
    ##residuals = sorted_total - (optimal_k * sorted_active + optimal_m)
    ##weighted_rms = np.sqrt(np.sum(n_samples * residuals**2) / np.sum(n_samples))
    ##return optimal_k, optimal_m, err_k, err_m, n_samples, weighted_rms
    
    
def plot_ac_voltage_for_biased_junction(
    dc_current,
    ac_current,
    ac_freq,
    critical_current
    ):
    
    time_axis = np.linspace(-25e-9, 25e-9, 20000)
    
    def ac_voltage_for_biased_junction(
        t,
        dc_current,
        ac_current,
        ac_freq,
        critical_current
        ):
        ''' Calculated by Christian Križan. '''
        
        omega = 2*np.pi*ac_freq
        
        mfq = (2.067833848e-15)/(2*np.pi)
        
        top_part = omega * (ac_current / critical_current) * np.cos(omega * t)
        
        inner_part_of_bottom = (dc_current + ac_current*np.sin( omega * t )) / critical_current
        
        bottom_part = np.sqrt( 1 - (inner_part_of_bottom)**2 )
        
        return mfq * (top_part / bottom_part)
    
    y_axis = ac_voltage_for_biased_junction(
        t = time_axis,
        dc_current = dc_current,
        ac_current = ac_current,
        ac_freq = ac_freq,
        critical_current = critical_current
        )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plt.grid()
    
    plt.xlabel("Time [µs]", fontsize=33)
    plt.ylabel("Voltage [nV]", fontsize=33)
    plt.title(f"Voltage over junction", fontsize=38)
    
    ax.tick_params(axis='both', labelsize=23)
    
    print(np.max(y_axis))
    
    plt.plot(time_axis*(1e6), y_axis*(1e9), color='green')
    plt.show()
    
def plot_critical_current_of_double_ScS_junction(  ):
    
    phi_phi0_axis = np.linspace(-5.5, 5.5, 20000)
    
    def function_cos(
        phi_phi0,
        ):
        ''' Calculated by Christian Križan. '''
        
        return np.abs(np.cos(np.pi * phi_phi0))
    
    y_axis = 2 * function_cos( phi_phi0_axis )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plt.grid()
    
    plt.xlabel("Φ/Φ₀ [-]", fontsize=33)
    plt.ylabel("I_c / I_s [-]", fontsize=33)
    plt.title(f"Critical current dependency on B", fontsize=38)
    
    ax.tick_params(axis='both', labelsize=23)
    
    print(np.max(y_axis))
    
    plt.plot(phi_phi0_axis, y_axis, color='purple')
    plt.show()
    
def plot_critical_current_of_triple_ScS_junction(  ):
    
    raise NotImplementedError("Halted! There is an error in the calculation below, do not proceed.")
    
    phi_phi0_axis = np.linspace(-5.5, 5.5, 20000)
    
    def function_triple_cos(
        phi_phi0,
        ):
        ''' Calculated by Christian Križan. '''
        
        return 2*np.abs(np.cos(np.pi * phi_phi0))+1
    
    y_axis = function_triple_cos( phi_phi0_axis )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    plt.grid()
    
    plt.xlabel("Φ/Φ₀ [-]", fontsize=33)
    plt.ylabel("I_c / I_s [-]", fontsize=33)
    plt.title(f"Critical current dependency on B, 3 shorts", fontsize=38)
    
    ax.tick_params(axis='both', labelsize=23)
    
    print(np.max(y_axis))
    
    plt.plot(phi_phi0_axis, y_axis, color='orange')
    plt.show()

def plot_barplot_comparing_qubit_quality_factors(
    qubit_identifiers = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    ):
    raise NotImplementedError("Halted! Not done.")

    # Redefine the data since execution state was reset
    quarters = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8"]
    ch3_values_updated = [1676534, 1190290, 2487886, 1500422, 1014434, 1123330, 1186250, 1122872]
    ch5_values_updated = [840215, None, 1337651, None, 1297704, None, 1212784, None]
    ch3_values_twice_updated = [1302120, None, 1375136, None, None, 365889, 1397712, 1116143]
    ch5_values_twice_updated = [None, None, 1341931, None, 1860994, None, 163470, None]

    # Create two subplots: one for Ch3 comparison and one for Ch5 comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # First subplot: Ch3 (Reference vs Manipulated Reference)
    axes[0].bar(x - 0.2, reference, width=0.4, label="Reference", color="blue", alpha=0.7)
    axes[0].bar(x + 0.2, [v if v is not None else 0 for v in manipulated_reference], width=0.4, label="Manipulated reference", color="green", alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(quarters)
    axes[0].set_title("Ch3 Comparison")
    axes[0].set_ylabel("Qubit quality factor")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].legend()
    for i in range(len(quarters)):
        if manipulated_reference[i] is None:
            axes[0].text(x[i] + 0.2, 500000, "N/A", ha="center", fontsize=9, color="green")

    # Second subplot: Ch5 (Manipulated vs Twice Manipulated)
    axes[1].bar(x - 0.2, [v if v is not None else 0 for v in manipulated], width=0.4, label="Manipulated", color="red", alpha=0.7)
    axes[1].bar(x + 0.2, [v if v is not None else 0 for v in twice_manipulated], width=0.4, label="Twice manipulated", color="orange", alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(quarters)
    axes[1].set_title("Ch5 Comparison")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].legend()
    for i in range(len(quarters)):
        if manipulated[i] is None:
            axes[1].text(x[i] - 0.2, 500000, "N/A", ha="center", fontsize=9, color="red")
        if twice_manipulated[i] is None:
            axes[1].text(x[i] + 0.2, 500000, "N/A", ha="center", fontsize=9, color="orange")

    # Adjust layout and display
    plt.suptitle("Side-by-Side Comparisons: Ch3 and Ch5 Variants")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_quality_factor_vs_manipulation(
    savepath = '',
    plot_difference_plot = False
    ):
    ''' Given a set of quality factors, plot these against how much the
        qubits were manipulated.
    '''
    qubits = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8"]
    
    ch3_quality_reference = [1676534, 1190290, 2487886, 1500422, 1014434, 1123330, 1186250, 1122872]
    ch5_quality_single    = [840215,  None,    1337651, None,    1297704, None,    1212784, None   ]
    ch3_quality_single    = [1302120, None,    1375136, None,    None,    365889,  1397712, 1116143]
    ch5_quality_twice     = [None,    None,    1341931, None,    1860994, None,    163470,  None   ]
    
    ch3_manipulated_first_percent  = [0, 0, 0, 0, 0, 0, 0, 0]
    ch5_manipulated_first_percent  = [-8.004, 7.679, 9.126, 1.780, 6.760, 6.682, 6.677, -0.510]
    ch3_manipulated_second_percent = [3.912, 0.145, 1.765, 2.207, 11.123, 9.152, 4.108, 4.010]
    ch5_manipulated_second_percent = [0, 0, 1.315, 0, 0.545, 0, 0.517, 0]
    
    # Ch3 round 1:
    ## No manipulation!
    
    # Ch5 round 1:
    # Q1: 12-02-2025_17-01_JJTest100W3_Ch5_Q1_0p70V_MANIPULATE_7855p085_KriK.csv
    # Q2: 14-02-2025_14-03_JJTest100W3_Ch5_Q2_0p85V_MANIPULATE_AND_CREEP_9754p432_KriK.csv
    # Q3: 14-02-2025_17-26_JJTest100W3_Ch5_Q3_0p85V_MANIPULATE_AND_CREEP_8494p915_KriK.csv
    # Q4: 13-02-2025_17-32_JJTest100W3_Ch5_Q4_0p85V_MANIPULATE_AND_CREEP_KriK.csv
    # Q5: 16-02-2025_17-53_JJTest100W3_Ch5_Q5_0p85V_MANIPULATE_AND_CREEP_6692p460_KriK.csv
    # Q6: 17-02-2025_08-52_JJTest100W3_Ch5_Q6_0p85V_MANIPULATE_AND_CREEP_6203p625_KriK.csv
    # Q7: 17-02-2025_11-40_JJTest100W3_Ch5_Q7_0p85V_MANIPULATE_AND_CREEP_7341p045_KriK.csv
    # Q8: 17-02-2025_10-55_JJTest100W3_Ch5_Q8_0p85V_MANIPULATE_AND_CREEP_6836p235_KriK.csv
    
    # Ch5 round 2:
    # Q1: -
    # Q2: -
    # Q3: 2025-04-02_18-17_Ch5_Q3_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q4: -
    # Q5: 2025-04-02_19-56_Ch5_Q5_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q6: -
    # Q7: 2025-04-02_20-15_Ch5_Q7_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q8: -
    
    # Ch3 round 2:
    # Q1: 2025-04-02_21-00_Ch3_Q1_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q2: 2025-04-03_18-47_Ch3_Q2_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q3: 2025-04-03_12-52_Ch3_Q3_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q4: 2025-04-03_18-09_Ch3_Q4_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q5: 2025-04-03_15-36_Ch3_Q5_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q6: 2025-04-03_17-52_Ch3_Q6_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q7: 2025-04-03_16-24_Ch3_Q7_ROUND2_0p85V_MANIPULATE_KriK.csv
    # Q8: 2025-04-03_16-58_Ch3_Q8_ROUND2_0p85V_MANIPULATE_KriK.csv
    
    # Convert all data to numpy arrays, replacing None with np.nan
    def to_np_array(data):
        return np.array([np.nan if x is None else x for x in data])
    
    ref_y = to_np_array(ch3_quality_reference)
    single_ch5_y = to_np_array(ch5_quality_single)
    single_ch3_y = to_np_array(ch3_quality_single)
    twice_ch5_y = to_np_array(ch5_quality_twice)
    
    ref_x = np.array(ch3_manipulated_first_percent)
    single_ch5_x = np.array(ch5_manipulated_first_percent)
    single_ch3_x = np.array(ch3_manipulated_second_percent)
    twice_ch5_x = np.array(ch3_manipulated_second_percent)
    
    # Compute qubit quality factor difference plots
    ch3_difference_quality = single_ch3_y - ref_y
    ch5_difference_quality = twice_ch5_y - single_ch5_y
    
    # Start figure with two subplots side by side
    if plot_difference_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 11), sharey=False)
    else:
        fig, ax1 = plt.subplots(1, figsize=(12.6, 11), sharey=False)
    
    symbol_list = ['s', '^', 'o', 'v', 'd', '*', 'x', 'p']
    for ii in range(len(qubits)):
        if ii == 0:
            label_to_use_1 = "Quantum device 1, unmanipulated"
            label_to_use_2 = "Quantum device 1, once manipulated"
            label_to_use_3 = "Quantum device 2, once manipulated"
            label_to_use_4 = "Quantum device 2, twice manipulated"
        else:
            label_to_use_1 = None
            label_to_use_2 = None
            label_to_use_3 = None
            label_to_use_4 = None
        
        if (ii == 6):
            bump_up_size = 140
        else:
            bump_up_size = 0
        
        ax1.scatter(ref_x[ii],        ref_y[ii]/1e6,        s=130+bump_up_size, label=label_to_use_1,  marker=symbol_list[ii], color="#EE1C1C")
        ax1.scatter(single_ch3_x[ii], single_ch3_y[ii]/1e6, s=130+bump_up_size, label=label_to_use_2,  marker=symbol_list[ii], color="#1CEE70")
        ax1.scatter(single_ch5_x[ii], single_ch5_y[ii]/1e6, s=130+bump_up_size, label=label_to_use_3,  marker=symbol_list[ii], color="#1C70EE")
        ax1.scatter(twice_ch5_x[ii],  twice_ch5_y[ii]/1e6,  s=130+bump_up_size, label=label_to_use_4,  marker=symbol_list[ii], color="#C41CEE")
    
    # Bump up the size of the ticks' numbers on the axes.
    ax1.tick_params(axis='both', labelsize=26)
    
    # Set up additional plot stuff.
    ##ax1.set_xlim(xmin=-13.0, xmax=13.0)
    
    ax1.set_xlim(xmin=-0.5, xmax=13.0)
    ax1.set_ylim(ymin=0)
    ax1.set_xlabel("Active manipulation [%]", fontsize=33)
    ax1.set_ylabel("Qubit quality factor [10⁶]", fontsize=33)
    ax1.legend(fontsize=26)
    ax1.grid(True)
    
    # --- Difference Plot (right) ---
    if plot_difference_plot:
        ax2.scatter(single_ch3_x, ch3_difference_quality/1e6, s=120, label="Sample 1: once manip. - no manip.",    marker='o', color="#EE1C1C")
        ax2.scatter(twice_ch5_x,  ch5_difference_quality/1e6, s=120, label="Sample 2: twice manip. - once manip.", marker='s', color="#1CEE70")
        
        ax2.axhline(0, color='gray', linewidth=1.2, linestyle='--')  # Zero-line
        ax2.tick_params(axis='both', labelsize=26)
        ax2.set_xlim(xmin=-13.0, xmax=13.0)
        ax2.set_xlabel("Active manipulation [%]", fontsize=33)
        ax2.set_ylabel("Δ Qubit quality factor [10⁶]", fontsize=33)
        ax2.legend(fontsize=26)
        ax2.grid(True)
    
    # Tight layout.    
    plt.tight_layout()
    
    # Save plots?
    if savepath != '':
        plt.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show shit.
    plt.show()
    

def perform_stepped_manipulation_analysis(
    filepath_list,
    normalise_time = 1,
    study_relaxation_here_idx = 30, # 30 → 30 minutes.
    savepath = ''
    ):
    ''' Perform various analyses on a stepped active manipulation experiment.
        
        normalise_time:
            0 = X axis shows the UNIX timestamp.
            1 = X axis is now a duration, with t₀ set at the timestamp of the
                very first datapoint.
        
        study_relaxation_here_idx:
            Sets which time the relaxation portions are analysed at.
    '''
    
    def unit_multiplier(unit_str):
        ''' Analyse the unit prefix for each datapoint.
        '''
        # Begin with the most common (and expected) resistance unit prefix.
        # Then, test a few other not-impossible alternatives.
        if 'kOhm' in unit_str:
            return 1e3
        elif 'MOhm' in unit_str:
            return 1e6
        elif 'GOhm' in unit_str:
            return 1e9
        elif 'TOhm' in unit_str:
            return 1e12
        elif 'TOhm' in unit_str:
            return 1e12
        
        elif 'hOhm' in unit_str:
            return 1e2
        elif 'daOhm' in unit_str:
            return 1e1
        elif 'dOhm' in unit_str:
            return 1e-1
        elif 'cOhm' in unit_str:
            return 1e-2
        elif 'mOhm' in unit_str:
            return 1e-3
        elif (('uOhm' in unit_str) or ('µOhm' in unit_str)):
            return 1e-6
        elif 'nOhm' in unit_str:
            return 1e-9
        
        elif 'Ohm' in unit_str:
            return 1
        else:
            raise ValueError("Halted! Cannot identify resistance prefix from datapoint: "+str(unit_str))
    
    def dump_csv(filepath):
        ''' For some .csv file containing resistance manipulation data,
            analyse this file and dump its contents.
        '''
        
        # Get CSV data.
        df = pd.read_csv(filepath, sep=';', header=None, dtype=str)
        
        resistances = []
        errorbars = []
        timestamps = []
        resistance_added = []
        annotations = {
            "START_MANIPULATION": [],
            "STOP_MANIPULATION": [],
            "START_CREEP": [],
            "STOP_CREEP": [],
            "SHORTED": []
        }
        
        for start_row in range(3, len(df), 6):  # 0-indexed, so row 4 is index 3
            # Extract timestamp
            try:
                timestamp_value = float(df.iat[start_row + 1, 1])  # B5, B11, etc.
            except (ValueError, TypeError):
                timestamp_value = np.nan
            timestamps.append(timestamp_value)
            
            # Extract unit and apply multiplier
            unit_cell = str(df.iat[start_row, 0])  # A4, A10, A16, etc.
            multiplier = unit_multiplier(unit_cell)
            
            # Extract resistance value and apply multiplier
            try:
                resistance_value = float(df.iat[start_row, 1]) * multiplier
            except (ValueError, TypeError):
                print("WARNING! Failed reading resistance value at data row "+str(start_row)+" (0-indexed).")
                resistance_value = np.nan  # Set to NaN
            resistances.append(resistance_value)
            
            # Extract error bar and apply multiplier
            try:
                errorbar_value = float(df.iat[start_row, 2]) * multiplier
            except (ValueError, TypeError):
                print("WARNING! Failed reading resistance error bar value at data row "+str(start_row)+" (0-indexed).")
                errorbar_value = np.nan  # Set to NaN
            errorbars.append(errorbar_value)
            
            # Check annotation string
            try:
                annotation = str(df.iat[start_row + 1, 2])  # C5, C11, etc.
            except IndexError:
                annotation = ""
            for keyword in annotations:
                if keyword in annotation:
                    annotations[keyword].append(len(resistances) - 1)
                    ##
                    ## Note: there is no guarantee that the target resistance
                    ##       was the resistance that was reached. Hence,
                    ##       this code snippet has been commented out.
                    ##
                    ##if keyword == 'START_MANIPULATION':
                    ##    try:
                    ##        suffix = annotation.strip().split('_')[-1]
                    ##        addition = float(suffix)
                    ##    except (IndexError, ValueError):
                    ##        print(f"WARNING! Failed to parse resistance addition from annotation: '{annotation}'")
                    ##        addition = np.nan
                    ##    resistance_added.append(addition)
        
        dumped_resistances = np.array(resistances)
        dumped_resistance_errorbars = np.array(errorbars)
        dump_timestamps = np.array(timestamps)
        ##resistance_added = np.array(resistance_added)
        
        return dumped_resistances, dumped_resistance_errorbars, dump_timestamps, annotations
    
    # Go through the entries.
    ## Let's instantiate the figure already at this point.
    if len( filepath_list ) == 2:
        fig1, axs1 = plt.subplots(1, 2, figsize=(31.1, 13))
        '''fig2, axs2 = plt.subplots(2, 2, figsize=(31.1, 13))''' # Removed, useful for research but co-authors don't want this in the paper.
        fig3, axs3 = plt.subplots(1, 2, figsize=(25.4, 13))
    else:
        fig1, axs1 = plt.subplots(1, 2, figsize=(31.1, 13))
        fig3, axs3 = plt.subplots(1, 2, figsize=(25.4, 13))
    
    # During the process, we will try to figure out the y limit for plots
    # showing resistance increase values.
    y_lim_top = 0
    y_lim_top_relaxation = 0
    
    # Go through the selected files.
    for kk in range(len( filepath_list )):
    
        # Set filepath.
        filepath = filepath_list[kk]
    
        # Dump data!
        dumped_resistances, dumped_resistance_errorbars, dump_timestamps, annotations = dump_csv(filepath)
        
        ## Plot 1 logic goes here.
        
        # Check whether to update the resistances.
        resistance_added_ohm = []
        time_taken_for_manipulation = []

        # Number of valid manipulation intervals is the minimum of the lengths.
        n_intervals = min(len(annotations['START_MANIPULATION']), len(annotations['STOP_MANIPULATION']))

        ## Define list for calculating where, in time, that the relaxation
        ## is analysed.
        relaxation_sample_period_list = []
        
        # Loop through each valid pair, that is, we are on the lookout for
        # erroneous entries that could happen if the device broke during
        # manipulation.
        for i in range(n_intervals):
            start_idx = annotations['START_MANIPULATION'][i]
            stop_idx  = annotations['STOP_MANIPULATION' ][i]
            
            # Get resistance at start and stop indices
            start_res = dumped_resistances[start_idx]
            stop_res  = dumped_resistances[stop_idx ]
            
            # Get timestamp at start and stop indices
            start_time = dump_timestamps[start_idx]
            stop_time  = dump_timestamps[stop_idx ]
            
            # Get differences.
            delta_res  = stop_res  - start_res
            delta_time = stop_time - start_time
            
            # Store.
            resistance_added_ohm.append(delta_res)
            time_taken_for_manipulation.append(delta_time)
            
            # Get step size held during the relaxation, as well.
            start_relax_idx = annotations['START_CREEP'][i]
            relaxation_sample_period_list.append(
                dump_timestamps[start_relax_idx+study_relaxation_here_idx] - dump_timestamps[start_relax_idx]
            )
        
        # Numpy conversions.
        resistance_added_ohm = np.array(resistance_added_ohm)
        time_taken_for_manipulation = np.array(time_taken_for_manipulation)
        relaxation_sample_period_list = np.array(relaxation_sample_period_list)
        
        # Calculate at what time the relaxation was studied.
        mean_relaxation_time_selected = np.mean(relaxation_sample_period_list)
        print("Relaxation time selected, mean: "+str(mean_relaxation_time_selected/60)+str(" min"))
        
        # Calculate resistance as percentages of increase relative
        # to the original resistance value.
        resistance_added_percent = 100 * (resistance_added_ohm/dumped_resistances[0])
        ## Important! The math here has been checked.
        ## There is not supposed to be a -1 anywhere;
        ## the statement here is correct as-is.
        
        # Set new y axis limits?
        ##if verbose:
        ##    print("Debug: weird Y limits found for MPW005A, see code.")
        if kk < 1:
            if np.max(resistance_added_percent) > y_lim_top:
                y_lim_top = np.max(resistance_added_percent)*1.1
        
        ## Plot 2 logic goes here.
        relaxation_trace_data_xy = []
        relaxation_stack_xy = []
        
        # Number of valid manipulation intervals is the minimum of the lengths.
        n_intervals = min(len(annotations['START_CREEP']), len(annotations['STOP_CREEP']))
        
        # Loop through each valid pair, that is, we are on the lookout for
        # erroneous entries that could happen if the device broke during
        # manipulation.
        for i in range(n_intervals):
            start_relaxation_idx = annotations['START_CREEP'][i]
            stop_relaxation_idx  = annotations['STOP_CREEP' ][i]
            
            # Get time snippet from these indices.
            ## Here, we want to normalise the time. So, subtract time[0]
            ## for all entries.
            relaxation_x = dump_timestamps[start_relaxation_idx:stop_relaxation_idx] - dump_timestamps[start_relaxation_idx]
            
            # Get resistance snippet from these indices.
            ## Here, we want to perform a percent conversion.
            ## Then, we want to move all entries down to y = 0.
            
            relaxation_y = 100*(dumped_resistances[start_relaxation_idx:stop_relaxation_idx]/dumped_resistances[0])
            relaxation_y -= relaxation_y[0] # Move down to y = 0
            
            # Append!
            relaxation_trace_data_xy.append( (relaxation_x,relaxation_y) )
            
            ## Plot 2b logic goes here!
            relaxation_stack_xy.append( (dump_timestamps[start_relaxation_idx +study_relaxation_here_idx] - dump_timestamps[0], relaxation_y[study_relaxation_here_idx]) )
            
            # Figure out y axis for the plot later.
            if np.max(relaxation_y) > y_lim_top_relaxation:
                y_lim_top_relaxation = np.max(relaxation_y)*1.1
        
        ## Plot 3 logic goes here.
        # Here, we want to find the deepest trench of the drop,
        # for each manipulation.
        resistance_trench_depth_percent = []
        resistance_increase_at_start_of_trench_percent = []
        
        # Number of valid manipulation intervals is the minimum of the lengths.
        n_intervals = min(len(annotations['START_MANIPULATION']), len(annotations['STOP_MANIPULATION']))
        
        # Loop through each valid pair, that is, we are on the lookout for
        # erroneous entries that could happen if the device broke during
        # manipulation.
        for i in range(n_intervals):
            start_idx = annotations['START_MANIPULATION'][i]
            stop_idx  = annotations['STOP_MANIPULATION' ][i]
            
            # For the drop, we are interested in how much "resistance percent"
            # had been added at the beginning of the drop.
            start_res = dumped_resistances[start_idx]
            
            # Grab the lowest point in the resistance interval, and compare
            # it to the starting value.
            drop_value = dumped_resistances[start_idx:stop_idx].min()
            drop_value_percent = ((start_res - drop_value) / dumped_resistances[0])*100
            
            # Store!
            resistance_increase_at_start_of_trench_percent.append( ((start_res / dumped_resistances[0])-1)*100 )
            resistance_trench_depth_percent.append(drop_value_percent)
        
        # Numpy conversion.
        resistance_increase_at_start_of_trench_percent = np.array(resistance_increase_at_start_of_trench_percent)
        resistance_trench_depth_percent = np.array(resistance_trench_depth_percent)
        
        
        ## Plot 4 logic goes here.
        # Compute time derivative of the resistances.
        # The gradient analysis will yield better results for evenly spaced data.
        # Thus, let's try to interpolate the data given the timestamps they were
        # collected on.
        
        # Array conversion.
        dump_timestamps    = np.array(dump_timestamps)
        dumped_resistances = np.array(dumped_resistances)
        
        # Calculate median time step.
        dt = np.median(np.diff(dump_timestamps))
        uniform_time = np.arange(dump_timestamps[0], dump_timestamps[-1], dt)
        
        # Interpolate resistance onto uniform time grid
        interp_func = interp1d(dump_timestamps, dumped_resistances, kind='linear', fill_value="extrapolate")
        uniform_resistance = interp_func(uniform_time)
        
        # Compute time derivatives
        dR_dt   = np.gradient(uniform_resistance, dt)
        d2R_dt2 = np.gradient(dR_dt, dt)
        d3R_dt3 = np.gradient(d2R_dt2, dt)
        
        ## Figure stuff!
        # Figure out colours and label tags.
        if ('mpw005a' in filepath.lower()):
            #label_tag = '350 nm medium_dose-oxide'
            label_tag = 'Medium-dose 1'
            colour_list = interpolate_hsv_colours("#C4EE1C", "#EE1C1C", len(relaxation_trace_data_xy))
        elif ('jjanneal01c' in filepath.lower()):
            #label_tag = '200 nm low_dose-oxide'
            label_tag = 'Low-dose 1'
            colour_list = interpolate_hsv_colours("#1C70EE", "#1CEE70", len(relaxation_trace_data_xy))
        else:
            label_tag = None
            colour_list = interpolate_hsv_colours("#1C70EE", "#1CEE70", len(relaxation_trace_data_xy))
        
        ## Subplot 1: Time needed to add X resistance.
        
        axs1[0].scatter(time_taken_for_manipulation/60, resistance_added_percent, s=190, color=colour_list[-1], label=label_tag)
        axs1[0].plot(time_taken_for_manipulation/60, resistance_added_percent, '--', color="#000000")
        #axs1[0].set_title("Time derivatives of resistance")
        axs1[0].set_xlabel("Time needed for step [min]", fontsize=33)
        axs1[0].set_ylabel("Resistance increase during step [%]", fontsize=33)
        axs1[0].legend(fontsize=24)
        axs1[0].grid(True)
        
        # Set limits, scales, ticks.
        axs1[0].set_ylim(-0.5, y_lim_top)
        axs1[0].set_xscale('log')
        axs1[0].tick_params(axis='both', labelsize=26)
        
        '''
        ## Subplot 2:
        
        # Plot!
        for ii in range(len(relaxation_trace_data_xy)):
            axs2[0,kk].scatter(relaxation_trace_data_xy[ii][0]/60, relaxation_trace_data_xy[ii][1], color=colour_list[ii])
        
        # Grid.
        axs2[0,kk].grid(True)
        
        # Axis labels.
        axs2[0,kk].set_xlabel("Duration [min]", fontsize=33)
        axs2[0,kk].set_ylabel("Resistance increase [%]", fontsize=33)
        axs2[0,kk].tick_params(axis='both', labelsize=26)'''
        
        ## Subplot 3:
        
        # Plot!
        axs1[1].scatter(resistance_increase_at_start_of_trench_percent, resistance_trench_depth_percent, s=190, color=colour_list[-1], label=label_tag)
        #axs1[1].plot(   resistance_increase_at_start_of_trench_percent, resistance_trench_depth_percent,       color=colour_list[-1]                 )
        
        # Axis labels.
        axs1[1].set_xlabel("Resistance increase at manipulation start [%]", fontsize=33)
        axs1[1].set_ylabel("Largest resistance drop [%]", fontsize=33)
        
        # Grid and stuff!
        axs1[1].grid(True)
        axs1[1].tick_params(axis='both', labelsize=26)
        axs1[1].legend(fontsize=24)
        
        ## Subplot 4: Does the relaxation effect stack?
        
        # Figure out time axis.
        if normalise_time == 1:
            time_axis = uniform_time - uniform_time[0]
        else:
            time_axis = uniform_time
        
        '''# Plot!
        axs2[1,kk].plot(time_axis, dR_dt,   label=r"$\frac{dR}{dt}$",     color='tab:red')
        axs2[1,kk].plot(time_axis, d2R_dt2, label=r"$\frac{d^2R}{dt^2}$", color='tab:orange')
        axs2[1,kk].plot(time_axis, d3R_dt3, label=r"$\frac{d^3R}{dt^3}$", color='tab:purple')
        #axs2[1,kk].set_title("Time derivatives of resistance")
        axs2[1,kk].set_xlabel("Time [s]", fontsize=33)
        axs2[1,kk].set_ylabel("Rate of resistance change [$\Omega / s$, $\Omega^2 / s^2$, $\Omega^3 / s^3$]", fontsize=33)
        axs2[1,kk].legend(fontsize=24)
        axs2[1,kk].grid(True)
        axs2[1,kk].tick_params(axis='both', labelsize=26)
        
        # Set axis limits.
        axs2[1,kk].set_ylim(-15.0, +15.0)
        
        # Insert vertical lines for manipulation events
        plotted_labels = set()
        for idx in annotations.get("START_MANIPULATION", []):
            ts = dump_timestamps[idx]
            
            # Plot duration instead?
            if normalise_time == 1:
                ts -= dump_timestamps[0]
            
            label = "Start manipulation" if "start" not in plotted_labels else None
            axs2[1,kk].axvline(ts, color='cyan', linestyle='--', linewidth=1.5, label=label)
            plotted_labels.add("start")
        for idx in annotations.get("STOP_MANIPULATION", []):
            ts = dump_timestamps[idx]
            
            # Plot duration instead?
            if normalise_time == 1:
                ts -= dump_timestamps[0]
            
            label = "Stop manipulation" if "stop" not in plotted_labels else None
            axs2[1,kk].axvline(ts, color='magenta', linestyle='--', linewidth=1.5, label=label)
            plotted_labels.add("stop")
        '''
        
        ## Fig 3 i.e. subplot 5
        # Plot!
        for ii in range(len(relaxation_trace_data_xy)):
            if kk < 1:
                if ii == 0:
                    axs3[0].scatter(relaxation_trace_data_xy[ii][0]/60, relaxation_trace_data_xy[ii][1], color=colour_list[ii], label=label_tag)
                else:
                    axs3[0].scatter(relaxation_trace_data_xy[ii][0]/60, relaxation_trace_data_xy[ii][1], color=colour_list[ii], label="_something")
            
            # Add line to show where the relaxation is being studied.
            if ii == 0:
                ##axs3[0].axvline(relaxation_trace_data_xy[ii][0][study_relaxation_here_idx]/60, color="#EE1C1C", linestyle='--', linewidth=1.5, label="_something")
                axs3[0].axvline(mean_relaxation_time_selected/60, color="#EE1C1C", linestyle='--', linewidth=1.5, label="_something")
            
            # Right plot.
            axs3[1].plot(relaxation_stack_xy[ii][0]/3600, relaxation_stack_xy[ii][1], color=colour_list[ii])
            if ii == len(relaxation_stack_xy)-1:
                axs3[1].scatter(relaxation_stack_xy[ii][0]/3600, relaxation_stack_xy[ii][1], s=90, color=colour_list[ii], label=label_tag)
            else:
                axs3[1].scatter(relaxation_stack_xy[ii][0]/3600, relaxation_stack_xy[ii][1], s=90, color=colour_list[ii], label="_something")
            
            # Limits.
            ## Use same axes for the (a) subplot.
            axs3[0].set_ylim(-0.5, y_lim_top)
            axs3[1].set_ylim(-0.5, y_lim_top)
            ##axs3[0].set_ylim(-0.5,y_lim_top_relaxation)
            ##axs3[1].set_ylim(-0.5,y_lim_top_relaxation)
            
            # Grid.
            axs3[0].grid(True)
            axs3[1].grid(True)
            
            # Axis labels.
            axs3[0].set_xlabel("Duration [min]", fontsize=33)
            axs3[0].set_ylabel("Resistance increase [%]", fontsize=33)
            axs3[0].tick_params(axis='both', labelsize=26)
            axs3[1].set_xlabel("Time since experiment start [h]", fontsize=33)
            axs3[1].set_ylabel("Resistance increase [%]", fontsize=33)
            axs3[1].tick_params(axis='both', labelsize=26)
            
            # Legends!
            axs3[0].legend(fontsize=24)
            axs3[1].legend(fontsize=24)
    
    # Tight!
    plt.tight_layout()
    
    # Save plots?
    if savepath != '':
        plt.figure(1)
        plt.savefig(savepath, dpi=164, bbox_inches='tight')
        
        #plt.figure(3)
        plt.figure(2)
        plt.savefig(savepath.replace(".png","_excerpt3.png"), dpi=164, bbox_inches='tight')
    
    # Plot shits!
    plt.show()
    
    
    # Return shits.
    return dumped_resistances, dumped_resistance_errorbars, dump_timestamps, annotations
    

def compare_junction_oxidation_dose_to_known_dataset(
    path_to_reference_data_file,
    list_of_normal_resistances_times_area_ohm_micrometer_squared = [],
    list_of_oxidation_times_in_minutes = [],
    list_of_oxidation_pressures_in_mbar = [],
    user_label_list = [],
    plot_reference_fit = False
    ):
    ''' Plots your values for some oxidation process that you have obtained,
        against the reference dataset.
        
        The reference fit comes from J. Phys. d: Appl. Phys. 48 (2015) 395308.
        
        The format of the dataset should be an .xlsx file.
        
        Format of file:
            A1: blank   B1: "R_sg"  C1: "R_n"   D1: blank   E1: "Area"  F1: "R_n * A"   G1: "G/A"       H1: "p*t"       I1: "t_ox"  J1: "p_ox"  K1: blank   L1: "t"
            A2: blank   B2: blank   C2: blank   D2: blank   E2: "µm^2"  F2: "µm^2"      G2: "mS/µm^2"   H2: "mbar s"    I2: "sec."  J2: "mbar"  K2: blank   L2: "min."
            C3:
                Insert cell data here! Remember that column K is blank.
    '''
    # User input sanitation.
    if not ((len(list_of_normal_resistances_times_area_ohm_micrometer_squared) == len(list_of_oxidation_times_in_minutes)) and (len(list_of_normal_resistances_times_area_ohm_micrometer_squared) == len(list_of_oxidation_pressures_in_mbar))):
        raise ValueError("Halted! The lengths of the input datasets do not match.\n"+\
        "len(list_of_normal_resistances_times_area_ohm_micrometer_squared): "+str(len(list_of_normal_resistances_times_area_ohm_micrometer_squared))+"\n"+\
        "len(list_of_oxidation_times_in_minutes): "+str(len(list_of_oxidation_times_in_minutes))+"\n"+\
        "len(list_of_oxidation_pressures_in_mbar): "+str(len(list_of_oxidation_pressures_in_mbar))
        )
    if not (len(user_label_list) == len(list_of_oxidation_pressures_in_mbar)):
        raise ValueError("Halted! The user-provided label list has an unexpected size. Did you forget something?")
    
    ## Calculate effective oxygen dose, using an empirical model:
    ## Fig. 4 in L. J. Zeng et al. 2015, J. Phys. D: Appl. Phys. 48 395308
    def effective_oxygen_dose(t_minutes, p_mbar):
        ''' t_minutes:  oxidation time in minutes
            p_mbar:     oxidation pressure in mbar
        '''
        return (t_minutes**0.65) * (p_mbar**0.43)
    
    # Open datafile, extract data.
    df = pd.read_excel(path_to_reference_data_file)

    # Extract data from columns I, J, and F.
    # Recall that Excel columns are 0-indexed in pandas as 8, 9, 5.
    # Extract data (skip first two rows: header and units)
    t_ox_seconds = df.iloc[2:, 8].dropna().to_numpy()       # Column I (t in seconds)
    p_ox_mbar = df.iloc[2:, 9].dropna().to_numpy()          # Column J (p in mbar)
    normal_state_resistance = df.iloc[2:, 5].dropna().to_numpy()  # Column F (R_N·A in Ω·µm^2)
    
    # Ensure all lists are the same length (safe fallback)
    min_len = min(len(t_ox_seconds), len(p_ox_mbar), len(normal_state_resistance))
    t_ox_seconds = t_ox_seconds[:min_len]
    p_ox_mbar = p_ox_mbar[:min_len]
    normal_state_resistance = normal_state_resistance[:min_len]
    
    # Convert time to minutes.
    t_ox_minutes = t_ox_seconds / 60
    
    # Calculate effective oxygen dose.
    dose = effective_oxygen_dose(t_ox_minutes, p_ox_mbar)
    
    # Plot setup.
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.loglog(dose, normal_state_resistance, 'o', color="#000000", markerfacecolor='none', label="J. Phys. D: Appl. Phys. 48 (2015) 395308")
    plt.xlabel("D = t₀^0.65 · p₀^0.43 [min^0.65 · mbar^0.43]", fontsize=33)
    plt.ylabel("Normal resistance [Ω · µm^2]", fontsize=33)
    plt.title("R_N vs. oxygen dose", fontsize=38)
    
    ## Now, plot the user-added data?
    if len(list_of_normal_resistances_times_area_ohm_micrometer_squared) != 0:
        
        for ii in range(len(list_of_normal_resistances_times_area_ohm_micrometer_squared)):
            normal_resistances_ohm_micrometer_squared = np.array(list_of_normal_resistances_times_area_ohm_micrometer_squared[ii])
            oxidation_times_in_minutes = np.array(list_of_oxidation_times_in_minutes[ii])
            oxidation_pressures_in_mbar = np.array(list_of_oxidation_pressures_in_mbar[ii])
            
            # Get user dose.
            user_dose = effective_oxygen_dose(oxidation_times_in_minutes, oxidation_pressures_in_mbar)
            
            # Plot user things.
            if ii == 0:
                marker_string = 'p'
            elif ii == 1:
                marker_string = '*'
            elif ii == 2:
                marker_string = 'H'
            elif ii == 3:
                marker_string = '^'
            elif ii == 4:
                marker_string = 'x'
            elif ii == 5:
                marker_string = 'v'
            else:
                marker_string = 'o'
            plt.loglog(user_dose, normal_resistances_ohm_micrometer_squared, marker_string, markersize = 8, label=user_label_list[ii])
    
    # Plot reference fit line?
    if plot_reference_fit:
        x_fit = np.linspace(1e-1, 1e2, 1000)
        def fit_func(x):
            return 57.25 * (x**(1.0139))
        plt.plot(x_fit, fit_func(x_fit), ':', color="red")
    
    # Axes adjustments.
    #plt.ylim(2e0, 3e10)#3e3)
    #plt.xlim(1e-1, 3e6)#1e2)
    ax.tick_params(axis='both', labelsize=23)
    
    # Grid and legend.
    plt.grid(True, which="both", ls="--")
    plt.legend(fontsize=24)
    plt.tight_layout()
    
    # Show stuff!
    plt.show()

def compare_aging_vs_junction_sizes(
    savepath = '',
    logarithmise_absolute_resistance = False,
    override_set_xlog = False,
    plot_fit = True,
    ):
    ''' Reconstruct plots from Maurizio Toselli's thesis and his raw data,
        but adjust for his later reported inaccuracy, that is, the x-axis
        in the thick-oxide plot is incorrect due to him being given the
        wrong junction size numbers.
        
        Important: the Y data is extracted from the rastered plot,
        this Y data is thus accurate when viewed in print, but less accurate
        when worked with digitally.
    '''
    
    def transform(data):
        ''' Changes reference from 1.00 being the start, into [%] of increase.
            That is, 20 means that the sample has increased its resistance
            by 20 %'''
        return [(x, (y - 1) * 100) for x, y in data]

    def shift_data(data, reference_day_timestamp, shift_with_this_many_days):
        ''' Accounts for the thick-oxide being measured
            <shift_with_this_many_days> days after the wafer's manufacture.
            AND, also convert the time entry into decimal-days.
        '''
        def convert_string_into_timestamp(string_format):
            dt = datetime.strptime(string_format, "%Y_%m_%d__%H_%M_%S")
            return dt.timestamp()
        
        # Get reference.
        reference = convert_string_into_timestamp(reference_day_timestamp)
        
        # Fix the number! Remember that the X of the double (i.e., entry 0)
        # is supposed to be a number in days, not seconds.
        # Hence, divide by 86400
        return [((convert_string_into_timestamp(x) -reference)/86400 +shift_with_this_many_days, y) for x, y in data]
    
    def percentise_resistance_data(data_to_percentise):
        ''' Takes a list of doubles, and normalises the content in the
            second index of the doubles, to the second index of the first
            double.
        '''
        reference_resistance = data_to_percentise[0][1]
        percentised_list = []
        for i in range(len(data_to_percentise)):
            timestamp_string, resistance = data_to_percentise[i]
            resistance_percent = resistance / reference_resistance
            percentised_list.append( (timestamp_string, resistance_percent) )
        return percentised_list
    
    def unpack(data):
        return zip(*data)
    
    ## Junction size data!
    thin_x  = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550])
    thin_y  = np.array([33.247058823529414, 28.2, 22.41176470588235, 21.24705882352941, 18.03529411764706, 16.129411764705882, 15.494117647058824, 13.905882352941177, 12.882352941176471, 12.458823529411765])
    thick_x = np.array([150, 200, 250, 300, 350, 400, 450, 500, 550, 600])
    thick_y = np.array([3.2823529411764705, 3.0352941176470587, 2.788235294117647, 2.611764705882353, 2.2588235294117647, 1.9058823529411764, 1.9058823529411764, 1.7647058823529411, 1.2352941176470589, 1.2352941176470589])
    
    ## Time data!
    
    # Thin 1!
    reference_day_thin1 = "2024_05_14__15_40_00" # Estimate from fabrication logs.
    aging_vs_time_100nm_daily_25nA_res = [ #C12R0
        ("2024_05_16__18_53_31", 33190.88588599869),
        ("2024_05_17__14_42_13", 35267.930373783944),
        ("2024_05_18__10_57_36", 37104.75851743472),
        ("2024_05_19__12_44_11", 38297.77209702591),
        ("2024_05_20__09_59_14", 38885.089396954085),
        ("2024_05_21__17_59_17", 40810.04743003749),
        ("2024_05_22__14_15_12", 41668.73488659467),
        ("2024_05_23__18_25_15", 42140.55895649924),
        ("2024_05_24__18_00_44", 42675.41039559225),
        ("2024_05_25__15_39_32", 43352.39858097007),
        ("2024_05_26__17_24_21", 43641.79837278881),
        ("2024_05_27__16_37_56", 44084.176051536335),
        ("2024_05_28__13_35_14", 44382.789861807054),
        ("2024_05_29__17_43_22", 44724.86303231583),
        ("2024_05_30__09_50_49", 45067.837904053195)
    ]
    aging_vs_time_100nm_daily_25nA = transform(
        percentise_resistance_data(aging_vs_time_100nm_daily_25nA_res)
    )
    
    aging_vs_time_100nm_biweekly_25nA_res = [ #C16R0
        ("2024_05_16__18_59_23", 34788.81359984259),
        ("2024_05_19__12_46_39", 39636.572069205526),
        ("2024_05_24__18_05_19", 44457.96909138682),
        ("2024_05_27__16_35_34", 45623.40668277429),
        ("2024_05_30__09_56_09", 46458.066109563144)
    ]
    aging_vs_time_100nm_biweekly_25nA = transform(
        percentise_resistance_data(aging_vs_time_100nm_biweekly_25nA_res)
    )
    
    aging_vs_time_100nm_weekly_25nA_res = [ #C20R0
        ("2024_05_16__19_02_33", 35996.30515883294),
        ("2024_05_24__18_44_14", 45703.79169948271),
        ("2024_05_30__11_00_02", 47601.51407567098),
    ]
    aging_vs_time_100nm_weekly_25nA = transform(
        percentise_resistance_data(aging_vs_time_100nm_weekly_25nA_res)
    )
    
    aging_vs_time_100nm_bimonthly_25nA_res = [ #C8R1
        ("2024_05_16__19_06_03", 30632.500928136786),
        ("2024_05_30__09_53_56", 41940.8037054309),
    ]
    aging_vs_time_100nm_bimonthly_25nA = transform(
        percentise_resistance_data(aging_vs_time_100nm_bimonthly_25nA_res)
    )
    
    
    # Thin 2!
    reference_day_thin2 = "2024_05_14__15_40_00" # Estimate from fabrication logs.
    aging_vs_time_500nm_daily_1uA_res = [ #C12R3
        ("2024_05_16__19_28_35", 913.956910733973),
        ("2024_05_17__14_48_41", 931.8648093402484),
        ("2024_05_18__11_08_32", 947.653916962966),
        ("2024_05_19__13_22_13", 956.5285413916414),
        ("2024_05_20__10_08_57", 965.2218924466715),
        ("2024_05_21__18_14_36", 992.9263503372712),
        ("2024_05_22__14_23_06", 998.36958195838),
        ("2024_05_23__18_53_42", 1004.0520504388213),
        ("2024_05_24__18_22_49", 1008.682828022061),
        ("2024_05_25__15_48_05", 1013.698497395419),
        ("2024_05_26__17_29_55", 1017.9934898945415),
        ("2024_05_27__16_50_06", 1021.2514349653823),
        ("2024_05_28__13_46_16", 1025.0283143325032),
        ("2024_05_30__10_15_26", 1029.5197316585088) # MT has a longer description regarding *this* datapoint.
    ]
    aging_vs_time_500nm_daily_1uA = transform(
        percentise_resistance_data(aging_vs_time_500nm_daily_1uA_res)
    )
    
    aging_vs_time_500nm_biweekly_1uA_res = [ #C17R3
        ("2024_05_16__19_32_35", 921.1435199549404),
        ("2024_05_24__18_57_36", 1013.6179528175744),
        ("2024_05_27__17_17_59", 1023.8096909003256),
        ("2024_05_30__10_48_25", 1032.3456340550931),
    ]
    aging_vs_time_500nm_biweekly_1uA = transform(
        percentise_resistance_data(aging_vs_time_500nm_biweekly_1uA_res)
    )
    
    aging_vs_time_500nm_weekly_1uA_res = [ #C20R3
        ("2024_05_16__19_30_39", 926.8692850514876),
        ("2024_05_24__18_55_53", 1022.575459429151),
        ("2024_05_30__10_46_12", 1043.1617901078193)
    ]
    aging_vs_time_500nm_weekly_1uA = transform(
        percentise_resistance_data(aging_vs_time_500nm_weekly_1uA_res)
    )
    
    aging_vs_time_500nm_bimonthly_1uA_res = [ #C16R3
        ("2024_05_16__19_34_48", 913.6422867354229),
        ("2024_05_30__10_19_11", 1026.837954296748)
    ]
    aging_vs_time_500nm_bimonthly_1uA = transform(
        percentise_resistance_data(aging_vs_time_500nm_bimonthly_1uA_res)
    )
    
    
    # Thick 1
    reference_day_thick1 = "2023_11_14__10_40_00" # Estimate from deposition start +2h 20' = (1h 10' per leg)
    aging_vs_time_200nm_daily_25nA_res = [ #C0R1
        ("2023_12_03__17_14_08", 34258.96858796486),
        ("2023_12_04__11_38_57", 34398.37243440831),
        ("2023_12_05__12_58_10", 34455.748048776535),
        ("2023_12_06__10_34_38", 34319.683914323614),
        ("2023_12_07__11_34_43", 34498.86086360684),
        ("2023_12_08__14_51_25", 34622.40441532264),
        ("2023_12_09__12_11_53", 35020.784432286666),
        ("2023_12_10__16_34_42", 35152.74811384351),
        ("2023_12_11__18_15_21", 35129.81439468518),
        ("2023_12_12__17_12_24", 35105.6601415939),
        ("2023_12_13__11_53_18", 35028.24514629035),
        ("2023_12_14__13_41_19", 35149.43762948493),
        ("2023_12_15__13_37_40", 35336.532091305104),
        ("2023_12_16__12_26_33", 35371.684626901435)
    ]
    aging_vs_time_200nm_daily_25nA = transform(
        percentise_resistance_data(aging_vs_time_200nm_daily_25nA_res)
    )
    
    aging_vs_time_200nm_biweekly_25nA_res = [ # C2R1
        ("2023_12_03__18_09_07", 31455.78375195259),
        ("2023_12_06__11_28_30", 31567.80197708531),
        ("2023_12_10__17_06_36", 32256.821931018996),
        ("2023_12_13__11_56_26", 32199.850114897494),
        ("2023_12_16__12_31_19", 32386.519847622763)
    ]
    aging_vs_time_200nm_biweekly_25nA = transform(
        percentise_resistance_data(aging_vs_time_200nm_biweekly_25nA_res)
    )
    
    aging_vs_time_200nm_weekly_25nA_res = [ # C4R1
        ("2023_12_03__17_19_20", 30889.453600541423),
        ("2023_12_10__17_01_03", 31616.378726497733),
        ("2023_12_16__12_28_58", 31679.78311859156)
    ]
    aging_vs_time_200nm_weekly_25nA = transform(
        percentise_resistance_data(aging_vs_time_200nm_weekly_25nA_res)
    )
    
    aging_vs_time_200nm_bimonthly_25nA_res = [ #C6R1
        ("2023_12_03__17_21_27", 31155.69961318053),
        ("2023_12_16__12_30_08", 31995.569742742515),
    ]
    aging_vs_time_200nm_bimonthly_25nA = transform(
        percentise_resistance_data(aging_vs_time_200nm_bimonthly_25nA_res)
    )
    
    
    # Thick 2
    reference_day_thick2 = "2023_11_14__10_40_00" # Estimate from deposition start +2h 20' = (1h 10' per leg)
    aging_vs_time_600nm_daily_1uA_res = [ # C0R4
        ("2023_12_03__17_43_31", 3165.061639366152),
        ("2023_12_04__11_47_28", 3171.054024758447),
        ("2023_12_05__13_09_55", 3179.666601344485),
        ("2023_12_06__11_16_50", 3167.945212004684),
        ("2023_12_07__12_06_06", 3177.509773126098),
        ("2023_12_08__14_57_22", 3183.42527992468),
        ("2023_12_09__12_23_49", 3200.8131770125256),
        ("2023_12_10__16_42_54", 3215.1531558823863),
        ("2023_12_11__18_22_06", 3221.0044254146364),
        ("2023_12_12__17_18_23", 3216.9527142333613),
        ("2023_12_13__12_05_27", 3214.1380001229454),
        ("2023_12_14__13_48_46", 3213.9587012655247),
        ("2023_12_15__13_42_58", 3215.209684764905),
        ("2023_12_16__12_42_15", 3226.9205173098403)
    ]
    aging_vs_time_600nm_daily_1uA = transform(
        percentise_resistance_data(aging_vs_time_600nm_daily_1uA_res)
    )
    
    aging_vs_time_600nm_biweekly_1uA_res = [ # C2R4
        ("2023_12_03__17_44_58", 3219.0579010482093),
        ("2023_12_06__11_19_26", 3227.393849170229),
        ("2023_12_10__16_45_43", 3272.032901574661),
        ("2023_12_13__12_06_50", 3267.990711836602),
        ("2023_12_16__12_43_23", 3280.163474304789)
    ]
    aging_vs_time_600nm_biweekly_1uA = transform(
        percentise_resistance_data(aging_vs_time_600nm_biweekly_1uA_res)
    )
    
    aging_vs_time_600nm_weekly_1uA_res = [ # C4R4
        ("2023_12_03__17_47_19", 3109.172075494075),
        ("2023_12_10__16_49_49", 3151.2306213508036),
        ("2023_12_16__12_44_32", 3153.3247656814087)
    ]
    aging_vs_time_600nm_weekly_1uA = transform(
        percentise_resistance_data(aging_vs_time_600nm_weekly_1uA_res)
    )
    
    aging_vs_time_600nm_bimonthly_1uA_res = [ # C6R4
        ("2023_12_03__17_49_24", 3112.9178083456177),
        ("2023_12_16__12_45_43", 3150.3727358462743)
    ]
    aging_vs_time_600nm_bimonthly_1uA = transform(
        percentise_resistance_data(aging_vs_time_600nm_bimonthly_1uA_res)
    )
    
    ## Account for the old medium_dose-oxide devices being much older
    ## than the old low_dose-oxide devices.
    aging_vs_time_100nm_daily_25nA     = shift_data( aging_vs_time_100nm_daily_25nA,     reference_day_thin1,  shift_with_this_many_days = 0)
    aging_vs_time_100nm_weekly_25nA    = shift_data( aging_vs_time_100nm_weekly_25nA,    reference_day_thin1,  shift_with_this_many_days = 0)
    aging_vs_time_100nm_biweekly_25nA  = shift_data( aging_vs_time_100nm_biweekly_25nA,  reference_day_thin1,  shift_with_this_many_days = 0)
    aging_vs_time_100nm_bimonthly_25nA = shift_data( aging_vs_time_100nm_bimonthly_25nA, reference_day_thin1,  shift_with_this_many_days = 0)
    aging_vs_time_500nm_daily_1uA      = shift_data( aging_vs_time_500nm_daily_1uA,      reference_day_thin2,  shift_with_this_many_days = 0)
    aging_vs_time_500nm_weekly_1uA     = shift_data( aging_vs_time_500nm_weekly_1uA,     reference_day_thin2,  shift_with_this_many_days = 0)
    aging_vs_time_500nm_biweekly_1uA   = shift_data( aging_vs_time_500nm_biweekly_1uA,   reference_day_thin2,  shift_with_this_many_days = 0)
    aging_vs_time_500nm_bimonthly_1uA  = shift_data( aging_vs_time_500nm_bimonthly_1uA,  reference_day_thin2,  shift_with_this_many_days = 0)
    aging_vs_time_200nm_daily_25nA     = shift_data( aging_vs_time_200nm_daily_25nA,     reference_day_thick1, shift_with_this_many_days = 0)#20)
    aging_vs_time_200nm_weekly_25nA    = shift_data( aging_vs_time_200nm_weekly_25nA,    reference_day_thick1, shift_with_this_many_days = 0)#20)
    aging_vs_time_200nm_biweekly_25nA  = shift_data( aging_vs_time_200nm_biweekly_25nA,  reference_day_thick1, shift_with_this_many_days = 0)#20)
    aging_vs_time_200nm_bimonthly_25nA = shift_data( aging_vs_time_200nm_bimonthly_25nA, reference_day_thick1, shift_with_this_many_days = 0)#20)
    aging_vs_time_600nm_daily_1uA      = shift_data( aging_vs_time_600nm_daily_1uA,      reference_day_thick2, shift_with_this_many_days = 0)#20)
    aging_vs_time_600nm_weekly_1uA     = shift_data( aging_vs_time_600nm_weekly_1uA,     reference_day_thick2, shift_with_this_many_days = 0)#20)
    aging_vs_time_600nm_biweekly_1uA   = shift_data( aging_vs_time_600nm_biweekly_1uA,   reference_day_thick2, shift_with_this_many_days = 0)#20)
    aging_vs_time_600nm_bimonthly_1uA  = shift_data( aging_vs_time_600nm_bimonthly_1uA,  reference_day_thick2, shift_with_this_many_days = 0)#20)
    
    # Define helper function for preparing the absolute-resistance data,
    # if the user needs it.
    def prepare_absolute_data(datasets_res, reference_day):
        '''Convert raw (timestamp_str, resistance) pairs into (days_since_ref, resistance).'''
        converted = []
        for data in datasets_res:
            converted.append(shift_data(data, reference_day, shift_with_this_many_days=0))
        
        return converted
    
    def subtract_initial_resistance(datasets):
        '''Given lists of (x, y) pairs, subtract the first y from all y-values in each dataset.'''
        adjusted = []
        for data in datasets:
            if not data:
                adjusted.append([])
                continue
            initial_res = data[0][1]
            adjusted.append([(x, y - initial_res) for (x, y) in data])
        return adjusted
    
    # Create figure.
    ##fig = plt.figure(figsize=(31.09, 19.75))
    fig = plt.figure(figsize=(30.9, 19.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.21, 1], wspace=0.12)

    # Swap: ax2_3 is left, ax1 is right
    ax2_3 = fig.add_subplot(gs[0])
    ax1   = fig.add_subplot(gs[1])
    
    # --- Subplot 1 (junction size vs resistance) ---
    ax1.scatter(thin_x, thin_y, s=190, color="#1C70EE", label="Aging, Low-dose")
    ax1.scatter(thick_x, thick_y, s=190, color="#EE1C1C", label="Aging, Medium-dose")
    ax1.set_xlim(-25, 675)
    ax1.set_ylim(-1, 42)
    ax1.tick_params(axis='both', labelsize=26)
    ax1.grid(True, which="both", ls="--")
    ax1.set_xlabel("Electrode width [nm]", fontsize=33)
    ax1.set_ylabel("Resistance increase [%]", fontsize=33)
    ax1.legend(fontsize=26)
    
    thin_colour_large_HSV  = ['#1cee70', '#1ceea4', '#1ceecb', '#1cddee', '#1cc0ee', '#1c9fee', '#1c84ee', '#1c70ee']
    thick_colour_large_HSV = ['#ee1c1c', '#ee5d1c', '#ee9d1c', '#eec01c', '#e6dd1c', '#d1ec1c', '#b9ee1c', '#c4ee1c']
    
    # Define model to be fitted to, if plotting the logarithmic version.
    def deltaR_model(t, A, tau, R_init):
        '''Model for ΔR(t) = R_init * A * log10(1 + t / tau)'''
        return R_init * A * np.log10(1 + t / tau)
    
    ## --- Subplot 2+3 combined (log-time aging) ---
    # --- Subplot 2+3 combined (lin-time aging) ---
    # Plot thin oxide datasets
    if logarithmise_absolute_resistance:
        # Use absolute-resistance datasets and convert timestamps to days
        thin_sets = prepare_absolute_data([
            aging_vs_time_100nm_daily_25nA_res, aging_vs_time_100nm_weekly_25nA_res,
            aging_vs_time_100nm_biweekly_25nA_res, aging_vs_time_100nm_bimonthly_25nA_res,
            aging_vs_time_500nm_daily_1uA_res, aging_vs_time_500nm_weekly_1uA_res,
            aging_vs_time_500nm_biweekly_1uA_res, aging_vs_time_500nm_bimonthly_1uA_res
        ], reference_day_thin1)

        thick_sets = prepare_absolute_data([
            aging_vs_time_200nm_daily_25nA_res, aging_vs_time_200nm_weekly_25nA_res,
            aging_vs_time_200nm_biweekly_25nA_res, aging_vs_time_200nm_bimonthly_25nA_res,
            aging_vs_time_600nm_daily_1uA_res, aging_vs_time_600nm_weekly_1uA_res,
            aging_vs_time_600nm_biweekly_1uA_res, aging_vs_time_600nm_bimonthly_1uA_res
        ], reference_day_thick1)
        
        # Subtract the initial resistance from each dataset to show ΔR (in ohms)
        thin_sets = subtract_initial_resistance(thin_sets)
        thick_sets = subtract_initial_resistance(thick_sets)
    else:
        thin_sets = [
            aging_vs_time_100nm_daily_25nA, aging_vs_time_100nm_weekly_25nA,
            aging_vs_time_100nm_biweekly_25nA, aging_vs_time_100nm_bimonthly_25nA,
            aging_vs_time_500nm_daily_1uA, aging_vs_time_500nm_weekly_1uA,
            aging_vs_time_500nm_biweekly_1uA, aging_vs_time_500nm_bimonthly_1uA
        ]
        thick_sets = [
            aging_vs_time_200nm_daily_25nA, aging_vs_time_200nm_weekly_25nA,
            aging_vs_time_200nm_biweekly_25nA, aging_vs_time_200nm_bimonthly_25nA,
            aging_vs_time_600nm_daily_1uA, aging_vs_time_600nm_weekly_1uA,
            aging_vs_time_600nm_biweekly_1uA, aging_vs_time_600nm_bimonthly_1uA
        ]
    
    markers = ['o', '^', 's', '*']
    #labels  = ['daily', 'weekly', 'biweekly', 'bimonthly']
    labels  = ['1/d', '1/w', '2/w', '2/m']
    sizes_thin  = ['25 nA, 100 nm,','25 nA, 100 nm,','25 nA, 100 nm,','25 nA, 100 nm,','1 µA, 500 nm,','1 µA, 500 nm,','1 µA, 500 nm,','1 µA, 500 nm,']
    sizes_thick = ['25 nA, 200 nm,','25 nA, 200 nm,','25 nA, 200 nm,','25 nA, 200 nm,','1 µA, 600 nm,','1 µA, 600 nm,','1 µA, 600 nm,','1 µA, 600 nm,']
    thin_colours  = thin_colour_large_HSV
    thick_colours = thick_colour_large_HSV
    
    # Prepare def that calculates current density.
    def current_density(s: str) -> str:
        # Parse current and length.
        parts = s.strip().split(',')
        current_str = parts[0].strip()
        length_str = parts[1].strip().rstrip(',')

        # Convert current to amperes.
        if 'nA' in current_str:
            I = float(current_str.split()[0]) * 1e-9
        elif 'µA' in current_str or 'uA' in current_str:
            I = float(current_str.split()[0]) * 1e-6
        else:
            raise ValueError("Unknown current unit")

        # Convert length to meters.
        if 'nm' in length_str:
            L = float(length_str.split()[0]) * 1e-9
        else:
            raise ValueError("Unknown length unit")

        # Calculate current density J = I / L²
        J = I / (L ** 2)

        # Convert to a human-readable string (e.g. MA/m²)
        if J >= 1e6:
            J_str = f"{J / 1e6:.1f} MA/m²,"
        elif J >= 1e3:
            J_str = f"{J / 1e3:.1f} kA/m²,"
        else:
            J_str = f"{J:.1f} A/m²,"

        return J_str
    
    for i, (data, color) in enumerate(zip(thin_sets, thin_colours)):
        marker = markers[i % 4]
        #ax2_3.scatter(*unpack(data), marker=marker, color=color, s=190, label=f"Low-dose {sizes_thin[i]} {labels[i % 4]}")
        ax2_3.scatter(*unpack(data), marker=marker, color=color, s=190, label=f"Lo, {current_density(sizes_thin[i])} {labels[i % 4]}")
        
        # Plot lines between the dots?
        ##ax2_3.plot(*unpack(data), ':', color=color)

    for i, (data, color) in enumerate(zip(thick_sets, thick_colours)):
        marker = markers[i % 4]
        #ax2_3.scatter(*unpack(data), marker=marker, color=color, s=190, label=f"Medium-dose {sizes_thick[i]} {labels[i % 4]}")
        ax2_3.scatter(*unpack(data), marker=marker, color=color, s=190, label=f"Me, {current_density(sizes_thick[i])} {labels[i % 4]}")
        
        # Plot lines between the dots?
        ##ax2_3.plot(*unpack(data), ':', color=color)
    
    # Fit?
    if logarithmise_absolute_resistance:
        # Known initial resistances [Ω]
        known_R_inits = [
            33190.88588599869, 35996.30515883294, 34788.81359984259, 30632.500928136786,
            913.956910733973, 926.8692850514876, 921.1435199549404, 913.6422867354229,
            34258.96858796486, 30889.453600541423, 31455.78375195259, 31155.69961318053,
            3165.061639366152, 3109.172075494075, 3219.0579010482093, 3112.9178083456177
        ]

        # Known tau values [days]
        known_taus = [
            2.134386574074074, 2.1406597222222223, 2.1384606481481483, 2.1430902777777776,
            2.158738425925926, 2.160173611111111, 2.161516203703704, 2.1630555555555557,
            19.273703703703703, 19.277314814814815, 19.311886574074073, 19.278784722222223,
            19.294108796296296, 19.296747685185185, 19.29511574074074, 19.298194444444444
        ]
    
        # Perform and plot fits for log-X data
        from functools import partial

        def fit_and_plot_fixed(dataset, color, ax, R_init, tau):
            '''if len(dataset) < 3:
                print("Skipped a set; too short. Set: "+str(dataset))
                return  # Skip if too few points.'''

            x_data, y_data = np.array(list(unpack(dataset)))

            # Define a 1D fit function with R_init and tau fixed
            def fixed_model(t, A):
                return deltaR_model(t, A, tau, R_init)

            try:
                # Fit only A
                popt, _ = curve_fit(fixed_model, x_data, y_data, p0=[0.05], maxfev=10000)
                A_fit = popt[0]
                print(f"Fit: A={A_fit:.3e}, (R_init={R_init:.2f} Ω, tau={tau:.3f} days)")

                # Generate smooth line for plotting
                t_fit = np.linspace(min(x_data), max(x_data), 200)
                y_fit = fixed_model(t_fit, A_fit)
                if plot_fit:
                    ax.plot(t_fit, y_fit, color=color, lw=2.3)

            except Exception as e:
                print(f"Fit failed for dataset: {e}")

        '''# Apply fits for all datasets
        for data, color in zip(thin_sets, thin_colours):
            fit_and_plot(data, color, ax2_3)
        for data, color in zip(thick_sets, thick_colours):
            fit_and_plot(data, color, ax2_3)'''
        if logarithmise_absolute_resistance:
            for data, color, R_init, tau in zip(thin_sets, thin_colours, known_R_inits[:8], known_taus[:8]):
                fit_and_plot_fixed(data, color, ax2_3, R_init, tau)

            for data, color, R_init, tau in zip(thick_sets, thick_colours, known_R_inits[8:], known_taus[8:]):
                fit_and_plot_fixed(data, color, ax2_3, R_init, tau)
    
    # Adjust axes and such, if logarithmic or not.
    if logarithmise_absolute_resistance:
        ax2_3.set_xscale('log')
        ax2_3.set_xlim(-120.8, 160.8)
        ax2_3.set_ylim(-590, 22460)
        ax2_3.set_xlabel("Time since deposition [days]", fontsize=33)
        ax2_3.set_ylabel("Resistance increase [Ω]", fontsize=33)
        ax2_3.grid(True, which="both", ls="--")
    else:
        ##ax2_3.set_yscale('log')
        if override_set_xlog:
            ax2_3.set_xscale('log')
            ax2_3.set_xlim(-120.8, 60)
        else:
            ax2_3.set_xlim(-0.8, 32.8)
        ax2_3.set_ylim(-1, 42)
        ax2_3.set_xlabel("Time since deposition [days]", fontsize=33)
        ax2_3.set_ylabel("Resistance increase [%]", fontsize=33)
        ax2_3.grid(True, which="both", ls="--")
    
    ax2_3.grid(True, which="both", ls="--")
    ax2_3.tick_params(axis='both', labelsize=26)
    ##ax2_3.legend(fontsize=26, ncol=2, columnspacing=0.6, loc='lower left')
    ax2_3.legend(fontsize=26, ncol=1, columnspacing=0.6, loc='upper right')
    
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=164, bbox_inches='tight')
        print("Saved to ", savepath)
    plt.show()
    
def plot_free_energy_vs_total_current_of_rf_squid():
    # Assume zero externally applied magnetic field.
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    ##Phi_0 = 2.067833848e-15 [Wb]
    ##L = 100e-12 # [H]
    I_C = 100e-6  # [A]
    
    def beta_LRF(L, I_C):
        return 2*np.pi * (L*I_C)/(2.067833848e-15)
    
    def energy_in_rf_squid( i, beta_LRF):
        return (1/2) * beta_LRF * (i**2) - np.cos(beta_LRF * i)
    
    # Find energy, specifically E_tot / E_J
    currents = np.linspace(-1.5, 1.5, 400)
    free_energy_normalised_100pH = energy_in_rf_squid( currents, beta_LRF(100e-12, I_C) )
    free_energy_normalised_10pH = energy_in_rf_squid( currents, beta_LRF(50e-12, I_C) )
    free_energy_normalised_1pH = energy_in_rf_squid( currents, beta_LRF(10e-12, I_C) )
    free_energy_normalised_0p1pH = energy_in_rf_squid( currents, beta_LRF(0.1e-12, I_C) )
    
    
    fig, ax = plt.subplots(figsize=(12, 7))
    plt.plot(currents, free_energy_normalised_100pH, label="L: 100 pH, I_C: 100 µA, β: "+f"{beta_LRF(100e-12, I_C):.2f}")
    plt.plot(currents, free_energy_normalised_10pH, label="L: 50 pH, I_C: 100 µA, β: "+f"{beta_LRF(50e-12, I_C):.2f}")
    plt.plot(currents, free_energy_normalised_1pH, label="L: 10 pH, I_C: 100 µA, β: "+f"{beta_LRF(10e-12, I_C):.2f}")
    plt.plot(currents, free_energy_normalised_0p1pH, label="L: 0.1 pH, I_C: 100 µA, β: "+f"{beta_LRF(0.1e-12, I_C):.2f}")
    
    plt.ylabel("E_tot / E_ J [-]", fontsize=33)
    plt.xlabel("I_s / I_C [-]", fontsize=33)
    ax.tick_params(axis='both', labelsize=23)
    
    plt.grid()
    plt.legend(fontsize=20)
    plt.show()
    
def plot_drop_removed(
    savepath = ''
    ):
    ''' Illustrate how much drop [s] was removed from the beginning of the active manipulation data traces.
    '''
    
    # Data: Drop removed vs. Voltage
    datasets = [
        {"voltages": [950, 900, 850, 800, 750], "drop_removed": [0, 0, 0, 0, 0]},
        {"voltages": [950, 900, 850, 800, 750], "drop_removed": [3.856, 0, 0, 0, 0]},
        {"voltages": [1000, 950, 925, 900, 850, 800], "drop_removed": [10.307, 7.097, 65.764, 30.316, 218.010, 146.505]},
        {"voltages": [1000, 950, 925, 900, 850, 800], "drop_removed": [7.063, 65.427, 26.532, 52.200, 133.327, 143.178]},
        {"voltages": [1050, 1000, 950, 900], "drop_removed": [0, 0, 0, 0]},
    ]
    colours = ['#C41CEE', '#1C70EE', '#1CEE70', '#C4EE1C', '#EE1C1C']
    #labels = ["Set (a): 200x200 nm, soft", "Set (b): 300x300 nm, soft", "Set (c): 318x318 nm, hard", "Set (d): 354x354 nm, hard", "Set (e): 350x350 nm, medium"]
    labels = ["Low-dose 1", "Low-dose 2", "High-dose 1", "High-dose 2", "Medium-dose 1"]

    # Data: drop removed vs. max resistance change
    additional_dataset = [
        {"max_res_change": [96.22063964205007, 44.99158976805173, 25.279942849287096, 12.360495955953477, 5.968368957966308], "drop_removed": [0, 0, 0, 0, 0]},
        {"max_res_change": [76.9063451288274, 39.94898308378969, 16.76463788989173, 7.11048591268908, 3.934905378623821], "drop_removed": [3.856360673904419, 0, 0, 0, 0]},
        {"max_res_change": [7.577167605507995, 2.899798807851095, 2.333199200214886, 2.00698761012863, 0.5836589957764815, 0.4579783734475429], "drop_removed": [10.307, 7.097, 65.764, 30.316, 218.010, 146.505]},
        {"max_res_change": [7.604564690048288, 2.6164501359842163, 2.2059728006996515, 1.37511, 0.6087130768012594, 0.41831299658590115], "drop_removed": [7.063, 65.427, 26.532, 52.200, 133.327, 143.178]},
        {"max_res_change": [20.241740738032178, 14.231358388694026, 7.15015084526569, 3.849712062876298], "drop_removed": [0, 0, 0, 0]},
    ]
    
    # Create 2-row subplot
    ##fig1, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12.8, 9.81), sharey=False)
    fig1, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12.8, 8.70), sharey=False)

    # === First subplot: Drop removed vs. Voltage ===
    for i, data in enumerate(datasets):
        ax1.scatter(data["voltages"], data["drop_removed"], s=80, color=colours[i], label=labels[i])

    ax1.set_xlabel("Voltage [mV]", fontsize=33)
    ax1.set_ylabel("Drop removed [s]", fontsize=33, labelpad=36.3)
    ax1.set_xlim(-35, 1085)
    ax1.set_ylim(-10, 250)
    ax1.tick_params(axis='both', labelsize=30)
    ax1.legend(fontsize=26)
    ax1.grid()

    # === Second subplot: Drop removed vs. Max resistance change ===
    for i, data in enumerate(additional_dataset):
        ax2.scatter(data["max_res_change"], data["drop_removed"], s=80, color=colours[i], label=labels[i])

    ax2.set_xlabel("Max resistance change in plot [%]", fontsize=33, labelpad=18)
    ax2.set_ylabel("Drop removed [s]", fontsize=33, labelpad=36.3)
    ax2.set_xlim(-3.125, 103.125)
    ax2.set_ylim(-10, 250)
    ax2.tick_params(axis='both', labelsize=30)
    ax2.legend(fontsize=26)
    ax2.grid()

    # Final layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.32)  # Space between subplots

    # Save shits?
    if savepath != '':
        fig1.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show shits.
    plt.show()


def increase_in_percent_per_second_exponential(
    lower_voltage,
    higher_voltage,
    initial_resistance,
    alpha_0,
    gamma,
    ):
    ''' Calculate how many extra "percents per second" you get from
        cranking the manipulation voltage from lower_voltage
        to higher_voltage.
    '''
    
    def exponential_voltage_dependency( V, alpha_0, gamma ):
        return alpha_0 * ((np.e)**(gamma * V))
    
    # Get (linear) resistance manipulation rates.
    old_resistance_per_second = exponential_voltage_dependency( lower_voltage, alpha_0, gamma )
    new_resistance_per_second = exponential_voltage_dependency( higher_voltage, alpha_0, gamma )
    
    # In 1 second, I get 100 % more resistance, if old_resistance_per_second is initial_resistance.
    old_resistance_per_second_percent = (old_resistance_per_second / initial_resistance) * 100
    new_resistance_per_second_percent = (new_resistance_per_second / initial_resistance) * 100
    
    # Return the increase: "I get these many more percents per second"
    return new_resistance_per_second_percent / old_resistance_per_second_percent

def plot_dielectric_breakdown_data(
    folder_path,
    number_of_junction_sizes_probed,
    electrode_width_list,
    plot_broken_data = False,
    resistance_limit_for_shorted_junction_ohm = 260,
    max_voltage_for_defining_R_in_mV = 50,
    savepath = ''
    ):
    ''' Reads all .txt files in folder_path, extracts Voltage and Current data,
        and plots them on the same figure.
        
        Parameters
        ----------
        folder_path : str
            Path to folder containing .txt data files.
        plot_broken_data : boolean
            If true, plot the entire data line, including datapoints
            after which the junction broke.
        savepath : str, optional
            Path to save the plot. If just an empty string '',
            then the plot will not be saved.
    '''
    
    # Input sanitation.
    if not isinstance(electrode_width_list, (list, np.ndarray)):
        raise TypeError(
            f"Error! 'electrode_width_list' must be a list or numpy array, got {type(electrode_width_list).__name__}"
        )
    
    # Set up a colour table given how many junction sizes there were.
    # Define start and end hex colours
    start_hex   = "#EE1C1C"
    end_hex     = "#C4EE1C"
    
    # Function to convert hex to RGB (0-1 range)
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    
    # Function to convert RGB (0-1) to hex
    def rgb_to_hex(rgb):
        return "#{:02X}{:02X}{:02X}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
    
    # Convert start and end to HLS
    start_rgb = hex_to_rgb(start_hex)
    end_rgb   = hex_to_rgb(end_hex)

    start_hls = colorsys.rgb_to_hls(*start_rgb)
    end_hls   = colorsys.rgb_to_hls(*end_rgb)
    
    # Generate 10 evenly spaced HSL-adjusted colours
    colours = []
    for i in range(number_of_junction_sizes_probed):
        t = i / (number_of_junction_sizes_probed-1)  # Interpolation factor
        h = start_hls[0] + (end_hls[0] - start_hls[0]) * t
        l = start_hls[1] + (end_hls[1] - start_hls[1]) * t
        s = start_hls[2] + (end_hls[2] - start_hls[2]) * t
        rgb = colorsys.hls_to_rgb(h, l, s)
        colours.append(rgb_to_hex(rgb))
    
    ## Colours is now a list where junction size colours can be chosen from,
    ## in the big plot.
    
    # Create data table, that will hold the averages of the breakdown data.
    ##list_of_violins = [[]] * number_of_junction_sizes_probed  ## Apparently this is classic Python bullshit :P
    list_of_violins = [[] for _ in range(number_of_junction_sizes_probed)]
    
    def trim_after_max_voltage(voltage_mV, current_uA):
        voltage_mV = np.array(voltage_mV)
        current_uA = np.array(current_uA)
        
        # Find index of the maximum voltage_mV
        max_index = np.argmax(voltage_mV)

        # Slice up to and including max_index
        return voltage_mV[:max_index+1], current_uA[:max_index+1]
    
    # Create figure.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.7 * 2, 11))
    
    # Go through all .txt files in the folder.
    colour_counter = 0
    number_of_traces_plotted = 0
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # Expecting exactly two rows: voltage_mV and current_uA.
            voltage_mV_line = lines[0].strip().split(",")[1:]  # Skip header text.
            current_uA_line = lines[1].strip().split(",")[1:]  # Skip header text.

            # Convert to floats.
            voltage_mV = np.array(list(map(float, voltage_mV_line)))
            current_uA = np.array(list(map(float, current_uA_line)))
            
            # Plot data after the junction is broken?
            if not plot_broken_data:
                ## No.
                voltage_mV, current_uA = trim_after_max_voltage(voltage_mV, current_uA)
            
            # Shorted junction?
            max_voltage_mV_idx = np.argmax(voltage_mV)
            end_resistance = (voltage_mV[max_voltage_mV_idx] / 1000) / (current_uA[max_voltage_mV_idx] / 1e6)
            if not(end_resistance < resistance_limit_for_shorted_junction_ohm):
            
                ## Insert stuffs into plot!
                
                # What label should be used?
                if (electrode_width_list == []):
                    ##label_tag = filename
                    raise ValueError("Halted! The user has to define the junction areas. A simple side-length will suffice, like \"600 nm\".")
                elif (number_of_traces_plotted >= len(electrode_width_list)):
                    # Then, stop plotting.
                    label_tag = None
                else:
                    # In this case, use the user-provided legend list.
                    label_tag = electrode_width_list[number_of_traces_plotted]
                
                # Plot shits.
                ax1.plot(current_uA, voltage_mV, label=label_tag, color = colours[colour_counter], alpha=0.6)
                
                ## At this point, we can take statistics.
                
                # 1. Find the dU/dI rate to get R.
                idx = np.argmax(voltage_mV > max_voltage_for_defining_R_in_mV)
                if voltage_mV[idx] > max_voltage_for_defining_R_in_mV:
                    linear_domain_mV = voltage_mV[:idx]
                    linear_domain_uA = current_uA[:idx]
                else:
                    linear_domain_mV = voltage_mV
                    linear_domain_uA = current_uA
                slope, intercept, r_value, p_value, std_err = linregress(linear_domain_uA/1e6, linear_domain_mV/1e3)
                # ... the slope is R.
                trace_resistance = slope # [Ω]
                
                # 2. (R·A) goes on the X axis.
                ## TODO: handle cases when the user has not provided areas.
                ## Convert sidelength to area.
                side_length_m = int(electrode_width_list[colour_counter].replace(' nm', '')) * 1e-9
                area_m2 = side_length_m ** 2
                resistivity_length = trace_resistance * area_m2
                
                # 3. Collect datapoint.
                ## Note here that the colour counter can be used to keep track of which set this datapoint belongs to.
                (list_of_violins[colour_counter]).append( (resistivity_length , (voltage_mV[max_voltage_mV_idx] / 1000)) )
                
                # Increase counters.
                number_of_traces_plotted += 1
                
            # Whether or not we plotted the curve, keep track of counters.
            colour_counter += 1
            colour_counter %= number_of_junction_sizes_probed
    
    ## We can now treat list_of_violins.
    
    # Prepare data for the violin plot.
    mean_res_lengths = []  # Mean resistivity-length for each group.
    x_stds = []            # Standard deviation of resistivity-length for each group.
    y_data = []            # Voltage data for violin.
    y_means = []           # Mean voltages.
    y_stds = []            # Standard deviation for voltages.
    
    for group in list_of_violins:
        if len(group) == 0:
            continue  # Skip empty sets!

        # Group is a list of tuples: (resistivity_length, voltage)
        resistivities = [item[0] for item in group]
        voltages      = [item[1] for item in group]
        
        print("No. resistivities found: "+str(len(resistivities)))
        mean_res = np.mean(resistivities)
        std_res  = np.std(resistivities)

        mean_res_lengths.append(mean_res)
        x_stds.append(std_res)

        y_data.append(voltages)
        y_means.append(np.mean(voltages))
        y_stds.append(np.std(voltages))
    
    # Plot violin plots at mean resistivity positions
    ##parts = ax2.violinplot(y_data, positions=mean_res_lengths, showmeans=False, showextrema=False) # This is very messy. Don't.
    ## # Colour the things to match ax1.
    ## for pc, col in zip(parts['bodies'], colours):
    ##     pc.set_facecolor(col)
    ##     pc.set_alpha(0.4)
    
    # Overlay mean and error bars.
    ## for item in range(len(mean_res_lengths)):
    ##     ## Note the 1e3, to divide the y-axis into millivolts.
    ##     ax2.errorbar(mean_res_lengths[item] * 1e9, y_means[item] * 1e3, yerr=y_stds[item] * 1e3, color=colours[item], fmt='o', capsize=5, markersize=8)
    for item in range(len(mean_res_lengths)):
        ax2.errorbar(
            mean_res_lengths[item] * 1e9, # X value
            y_means[item] * 1e3,          # Y value
            xerr  = x_stds[item] * 1e9,   # Horizontal error bar
            yerr  = y_stds[item] * 1e3,   # Vertical error bar
            color = colours[item],
            fmt='o',
            capsize=5,
            markersize=8,
            label=electrode_width_list[item]
        )
    
    ax2.grid()
    ax2.tick_params(axis='both', labelsize=26)
    ##ax2.set_xlabel("Mean resistivity-length [nΩ·m²]", fontsize=33)
    ax2.set_xlabel(r'$\overline{\mathrm{R} \!\cdot\! \mathrm{A}}$ [n$\Omega$·m$^2$]', fontsize=33)
    ax2.set_ylabel("Breakdown voltage [mV]", fontsize=33)
    ax2.set_ylim(-65, 1400) ## TODO fix proper limits.
    ax2.legend(fontsize=26, loc='lower right')
    ax2.set_xlim(-0.06, 1.26)
    
    # Formatting for the regular breakdown voltage plot.
    ax1.grid()
    ax1.tick_params(axis='both', labelsize=26)
    ax1.set_xlabel("Current [µA]", fontsize=33)
    ax1.set_ylabel("Voltage [mV]", fontsize=33)
    #ax1.set_title("Dielectric breakdown data")
    ax1.legend(fontsize=26)
    ax1.set_ylim(-65, 1400) ## TODO fix proper limits.
    
    plt.tight_layout()
    if not (savepath == ''):
        fig.savefig(savepath, dpi=164, bbox_inches='tight')
        print("Figure saved to: " + str(savepath))
    plt.show()

def plot_ln2_data(
    file_path_ln2,
    file_path_G0T0,
    experimentally_found_G0 = (0.8824 + 0.8799 + 0.8751)/3, ## This number was found through Maurizio Toselli's 2024 experiments.
    experimentally_found_T0 = (783.1 + 792.9 + 762.4)/3,    ## This number was found through Maurizio Toselli's 2024 experiments.
    normalise_resistances = 0,
    savepath = '',
    plot = True,
    plot_no_junction_resistance_under_ohm = 260,
    error_factor_threshold = 0.005
    ):
    ''' Plot the resistance manipulation and relaxation from electrical
        annealing performed onto a junction bathing in liquid nitrogen.
        Any datapoints where the error bar is absurd, that is, too large
        relative to the resistance, will be filtered out.
        
        plot_no_junction_resistance_under_ohm: threshold for where you
        know the junction to be shorted. Experimentally, I have found
        that 260 Ω is a good indicator.
    '''
    # Storage
    resistances = []   # in kΩ
    errors = []        # std_err of slope, in kΩ
    times_min = []     # time in minutes
    temperatures = []  # in K

    # Read and split into blocks by blank lines (works with CRLF too)
    with open(file_path_ln2, 'r', encoding='utf-8') as f:
        content = f.read()
    
    blocks = [b.strip() for b in re.split(r'\r?\n\s*\r?\n', content) if b.strip()]
    
    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        I_vals = None
        V_vals = None
        t_val = None
        T_val = None
        
        # Parse!
        for ln in lines:
            parts = ln.split(';')
            key = parts[0].strip()
            vals = parts[1:]

            # Robust identification by the header text
            if 'I [' in key and 'nA' in key:
                try:
                    I_vals = np.array([float(x) for x in vals], dtype=float)
                except ValueError:
                    I_vals = None
            elif 'V [' in key and 'uV' in key:
                try:
                    V_vals = np.array([float(x) for x in vals], dtype=float)
                except ValueError:
                    V_vals = None
            elif 'Time' in key:
                try:
                    t_val = float(vals[0])
                except (IndexError, ValueError):
                    t_val = None
            elif 'Temperature' in key:
                try:
                    T_val = float(vals[0])
                except (IndexError, ValueError):
                    T_val = None

        # Only attempt fit if we have both arrays and they match in length
        if I_vals is None or V_vals is None:
            # skip block if data missing
            continue
        if I_vals.size != V_vals.size:
            # skip if mismatched lengths
            continue
        
        # Linear regression V [µV] vs I [nA] → slope in [µV/nA] == kΩ
        try:
            slope, intercept, r_value, p_value, std_err = linregress(I_vals, V_vals)
        except Exception:
            slope = np.nan
            std_err = np.nan

        R_kOhm = slope         # Slope is already in kΩ
        err_kOhm = std_err     # std_err same units → kΩ
        
        # Filter: remove datapoints with absurd error values
        if np.isnan(R_kOhm) or np.isnan(err_kOhm):
            continue
        if (abs(err_kOhm) > abs(R_kOhm) * error_factor_threshold) or (abs(R_kOhm) < plot_no_junction_resistance_under_ohm/1000):
            # Skip this point
            continue
        
        # Append!
        resistances.append(R_kOhm)
        errors.append(err_kOhm)

        # Time in minutes (if missing, store NaN)
        times_min.append((t_val / 60.0) if (t_val is not None) else np.nan)
        temperatures.append(T_val if (T_val is not None) else np.nan)

    # Convert to numpy arrays
    resistances = np.array(resistances, dtype=float)
    errors = np.array(errors, dtype=float)
    times_min = np.array(times_min, dtype=float)
    temperatures = np.array(temperatures, dtype=float)
    
    # Store the initial resistance (before any normalization)
    res_initial = resistances[0] if resistances.size > 0 else np.nan
    
    # There will be another trace that is normalised
    # from cryogenic to room-temperature.
    ## The idea from this comes from Maurizio Toselli:
    ## The junction is a MIM diode whose resistance vs. temperature
    ## can be mapped as it cools. That way, it can be mapped to
    ## Simmon's model, and through the average of three fits (or more)
    ## one knows the temperature translation for that junction.
    ## Specifically, G_0 and T_0 for the equation.
    """def simmons_model_OLD( T, T_0, G_0 ):
        ''' Convert the observed resistance for some temperature,
            into a room-temperature equivalent resistance.
            
            G(T) = G₀ · (1 + ( T / T₀ )^2)
            ... where G(T) is the normalised conductance. That is, some number
            between 1.0 and 0.0.
        '''
        return G_0 * ( 1 + (T/T_0)**2 )"""
    ## Update: Use the same def as MT himself used:
    ## equivalent_resistances2 = equivalent_R_roomT(R = np.array(resistances2), T = np.array(temperatures2), T_0 = 779.45)
    def simmons_model(T, T_0, R, room_temp = 297):
        ''' MT used 294 as room-temp, CK uses 297 b/c that's the data for R(0).
        '''
        G = 1/R
        return 1 / (G * ( 1 + (room_temp/T_0)**2 )/( 1 + (T/T_0)**2 ))
    
    ## # Get the normalised conductance, and convert to normalised resistances.
    ## normalised_G_temp = simmons_model_OLD(temperatures, experimentally_found_T0, experimentally_found_G0)
    ## normalised_R_temp = 1/normalised_G_temp
    ## resistances_T_adjusted_OLD = resistances / normalised_R_temp
    resistances_T_adjusted = simmons_model( temperatures, experimentally_found_T0, resistances )
    
    # Normalise if requested (divide resistances and errors by res_initial)
    if normalise_resistances == 1 and not np.isnan(res_initial) and res_initial != 0:
        resistances = ((resistances / res_initial) - 1) * 100
        resistances_T_adjusted = ((resistances_T_adjusted / resistances_T_adjusted[0]) - 1) * 100
        errors = (errors / res_initial) * 100
        ylabel_res = "Resistance increase [%]"
    elif normalise_resistances == 0:
        ylabel_res = "Resistance [kΩ]"
    else:
        raise ValueError("Unknown argument given for normalise_resistances: "+str(normalise_resistances))
    
    ####################################################################
    # Process data relating to the three cooldowns needed to establish #
    #  the normalised conductance and the characteristic temperature   #
    ####################################################################
    
    def get_output_file_name(input_file_name, plot_name):
        base_name, extension = input_file_name.rsplit('.', 1)
        output_file_name = f"{base_name}_{plot_name}"
        return output_file_name
    
    # Define the linear model for conductance in terms of temperature squared
    def linear_model(T_squared, m, c):
        return m * T_squared + c

    # List of input file names to process
    input_file_names = [
        "07-11-2024_13-04_LiquidNExperimentNoManipulationTest.csv",
        "07-11-2024_15-32_LiquidNExperimentNoManipulationTest.csv",
        "08-11-2024_17-55_LiquidNExperimentNoManipulation.csv"
    ]

    # Prepare the plot?
    if plot:
        # Weird and somewhat pre-emptive place to make the plot, but oh well.
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(25, 10), sharex=False)
    plot_handles = []
    colours = ["#EE1C1C", "#C4EE1C", "#1C70EE"]

    # Lists to store fit parameters for legend display
    fit_results = []  # Will store tuples of (G_0, T_0)

    for i, input_file_name in enumerate(input_file_names):
        output_file_name = get_output_file_name(input_file_name, "GvsT_all")

        # Arrays to hold data from each file
        times = []
        measurements = []
        resistances_G0T0 = []
        temperatures_G0T0 = []

        # Open and read the CSV file
        with open(os.path.join(file_path_G0T0, "LogFiles", input_file_name), mode='r') as file:
            reader = csv.reader(file, delimiter=';')

            for row in reader:
                if row and row[0] == "R [kOhm]":
                    resistances_G0T0.append(float(row[1]))
                elif row and "measurement" in row[0]:
                    measurements.append(int(row[1]))
                elif row and row[0] == "Time [s]":
                    times.append(float(row[1]))
                elif row and row[0] == "Temperature [K]":
                    temperatures_G0T0.append(float(row[1]))

        # Convert lists to numpy arrays
        resistances_G0T0 = np.array(resistances_G0T0)
        measurements = np.array(measurements)
        times = np.array(times)
        temperatures_G0T0 = np.array(temperatures_G0T0)

        # Normalize resistance based on file name
        if input_file_name == "08-11-2024_17-55_LiquidNExperimentNoManipulation.csv":
            normalised_resistance = resistances_G0T0 / resistances_G0T0[-1]
        else:
            normalised_resistance = resistances_G0T0 / resistances_G0T0[0]

        # Filter data
        normalised_resistance_filtered = []
        temperatures_filtered = []
        
        for j in measurements:
            if (normalised_resistance[j] >= 1 - 0.5 / 100) and (normalised_resistance[j] <= 1 + 12.5 / 100):
                normalised_resistance_filtered.append(normalised_resistance[j])
                temperatures_filtered.append(temperatures_G0T0[j])

        # Convert filtered lists to numpy arrays
        temperatures_filtered = np.array(temperatures_filtered)
        normalised_resistance_filtered = np.array(normalised_resistance_filtered)

        # Fit data using Simmon's model
        initial_guess = [0.9 / (790 ** 2), 0.9]
        popt, pcov = curve_fit(linear_model, temperatures_filtered**2, 1 / normalised_resistance_filtered, p0=initial_guess)
        m_fit, c_fit = popt

        # Calculate the fit parameters
        T_0 = np.sqrt(c_fit / m_fit)
        G_0 = c_fit
        fit_results.append((G_0, T_0))  # Append to results for legend

        # Generate the fitted conductance curve
        temperatures_fit = np.linspace(min(temperatures_filtered), max(temperatures_filtered), 500)
        conductance_fit = linear_model(temperatures_fit**2, *popt)

        # Plot data points and fit line
        if plot:
            handle_data = ax3.scatter((temperatures_filtered**2)/1e4, 1 / normalised_resistance_filtered, color=colours[i % len(colours)], marker='o', s=13, label=f'Data ({input_file_name})')
            handle_fit, = ax3.plot((temperatures_fit**2)/1e4, conductance_fit, linewidth=2, color=colours[i % len(colours)], linestyle='--')
        
        plot_handles.append(handle_fit)
    
    # Plotting
    if plot:
        ##fig, ax1 = plt.subplots(figsize=(12.5, 10))
        ##fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(25, 10), sharex=False)  # Declared earlier!

        # Temperature on left y-axis (red)
        ax1.plot(times_min, temperatures, '-', linewidth=2, color="#1CEE70", label="Temperature")
        ax1.set_xlabel("Time [min]", fontsize=33)
        ax1.set_ylabel("Temperature [K]", fontsize=33)##, color="#EE1C1C")
        ax1.set_ylim(bottom=-14.77625)
        ##ax1.tick_params(axis='y', labelcolor="#EE1C1C")

        # Resistance on right y-axis (blue)
        ax2 = ax1.twinx()
        ## Note that explicitly defining z-order here is mandatory for some reason.
        ax2.errorbar(times_min, resistances, yerr=errors,
                     fmt='o', markersize=6, ecolor="#EE1C1C", color="#EE1C1C",
                     capsize=3, label="Measured resistance increase", zorder=1)
        ax2.scatter(times_min, resistances_T_adjusted, s=13, color="#1C70EE", label="Temperature-corrected res. increase", zorder=2)
        ax2.set_ylabel(ylabel_res, fontsize=33)##, color="#1C70EE")
        ##ax2.tick_params(axis='y', labelcolor="#1C70EE")
        
        # Bump up ticks. Add grid.
        ax1.tick_params(axis='both', labelsize=26)
        ax2.tick_params(axis='both', labelsize=26)
        
        # Combined legend.
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        # Grid and legend.
        ax2.grid()
        ## Important: setting ax1.legend here,
        ## will make the grid very visible through the legend box.
        ## But using ax2.legend solves this problem, for now.
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=26)##, frameon=True, facecolor='white', framealpha=0.8)
        
        ###############
        # Second plot #
        ###############
        
        # Legend setup for the second plot.
        ax3.legend(
            plot_handles,
            [f"$G_0 = {G_0:.4f}$, $T_0 = {T_0:.1f}$" for G_0, T_0 in fit_results],
            fontsize=26,
            loc='best',
            title="Fit to Simmon's model:\n"+r"$G(T) = G_0 \left(1 + \left(\frac{T}{T_0}\right)^2\right)$",
            title_fontsize=28,
        )
        
        ax3.tick_params(axis='both', labelsize=26)
        ax3.set_xlabel("Temperature² [1000 K²]", fontsize=33)
        ax3.set_ylabel("Normalized conductance", fontsize=33)
        
        ax3.grid()
        plt.tight_layout()
        if not (savepath == ''):
            fig.savefig(savepath, dpi=164, bbox_inches='tight')
            print("Figure saved to: " + str(savepath))
        plt.show()

    # Return (time[min], resistance[kΩ or normalised], error[same units], temperature[K], res_initial[kΩ])
    return times_min, resistances, errors, temperatures, res_initial, resistances_G0T0, temperatures_G0T0

def map_pearson_coefficient_in_active_manipulation(
    name_of_sample = 'Hatmatilka',
    savepath = ''
    ):
    ''' Given a list of fitted parameters of electrical active resistance
        manipulation, where the numbers have been fitted to
        R(t) = α · t + β · t²
        
        ... then calculate the correlation thingies between α and β.
    '''
    
    print("Debug mode engaged: will overwrite alpha-list and beta-list with numbers hard-coded in the program.") # TODO
    
    # Voltages
    millivolt_low_dose    = [750, 800, 850, 900, 950] # mV
    millivolt_medium_dose = [900, 950, 1000, 1050] # mV
    millivolt_high_dose   = [800, 850, 900, 925, 950, 1000] # mV
    
    '''
    ## Ohm-equivalents!
    # Low-dose devices
    alpha_low_dose1     = [2.93, 6.42, 12.1, 22.6, 50.3]
    err_alpha_low_dose1 = [0.15, 0.18, 0.6, 0.5, 0.4]
    beta_low_dose1      = [-2.6, -5.7, -8.23, -16.5, -33.0]
    err_beta_low_dose1  = [0.5, 0.8, 0.25, 2.1, 1.5]
    
    alpha_low_dose2     = [0.80, 1.56, 3.920, 8.48, 17.96]
    err_alpha_low_dose2 = [0.05, 0.04, 0.017, 0.21, 0.12]
    beta_low_dose2      = [-0.328, -1.00, -2.46, -4.66, -7.9]
    err_beta_low_dose2  = [0.023, 0.17, 0.08, 0.08, 0.5]

    # Medium-dose devices
    alpha_medium_dose1     = [1.99, 3.46, 6.762, 10.449]
    err_alpha_medium_dose1 = [0.05, 0.03, 0.023, 0.021]
    beta_medium_dose1      = [-2.00, -3.09, -4.71, -4.87]
    err_beta_medium_dose1  = [0.27, 0.12, 0.10, 0.09]

    # High-dose devices
    alpha_high_dose1     = [0.0809, 0.159, 0.543, 0.602, 0.78, 2.28]
    err_alpha_high_dose1 = [0.0026, 0.019, 0.017, 0.015, 0.23, 0.16]
    beta_high_dose1      = [0.238, 0.259, -0.076, 0.153, -0.162, -0.95]
    err_beta_high_dose1  = [0.011, 0.008, 0.008, 0.006, 0.014, 0.06]
    
    alpha_high_dose2     = [0.130, 0.073, 0.255, 0.446, 0.632, 1.70]
    err_alpha_high_dose2 = [0.015, 0.013, 0.011, 0.005, 0.010, 0.10]
    beta_high_dose2      = [-0.174, 0.185, -0.035, 0.0266, -0.173, -0.217]
    err_beta_high_dose2  = [0.007, 0.006, 0.004, 0.0013, 0.005, 0.005]'''
    
    ## Percent equivalents!
    # Low-dose 1
    alpha_low_dose1     = [0.0262, 0.0566, 0.105, 0.192, 0.404]
    err_alpha_low_dose1 = [0.0013, 0.0016, 0.005, 0.004, 0.003]
    beta_low_dose1      = [-2.33e-5, -5.00e-5, -7.12e-5, -1.40e-4, -2.65e-4]
    err_beta_low_dose1  = [0.54e-5, 0.68e-5, 0.22e-5, 1.8e-5, 1.2e-5]
    
    # Low-dose 2
    alpha_low_dose2     = [0.0152, 0.0293, 0.0692, 0.159, 0.299]
    err_alpha_low_dose2 = [0.0010, 0.0007, 0.0003, 0.004, 0.002]
    beta_low_dose2      = [-6.26e-6, -1.87e-5, -4.35e-5, -8.74e-5, -1.31e-4]
    err_beta_low_dose2  = [4.4e-7, 3.1e-6, 1.4e-6, 1.5e-6, 8e-6]
    
    # Medium-dose 1
    alpha_medium_dose1     = [0.0176, 0.0319, 0.0595, 0.0974]
    err_alpha_medium_dose1 = [0.0004, 0.0003, 0.0002, 0.0002]
    beta_medium_dose1      = [-1.77e-5, -2.85e-5, -4.14e-5, -4.54e-5]
    err_beta_medium_dose1  = [2.4e-6, 1.2e-6, 9e-7, 8e-7]
    
    # High-dose 1
    alpha_high_dose1     = [0.000997, 0.00204, 0.00705, 0.00776, 0.0100, 0.0287]
    err_alpha_high_dose1 = [0.000032, 0.00024, 0.00022, 0.00019, 0.0029, 0.0020]
    beta_high_dose1      = [2.93e-6, 3.33e-6, -9.9e-7, 1.97e-6, -2.08e-6, -1.20e-5]
    err_beta_high_dose1  = [1.4e-7, 1.0e-7, 1.0e-7, 8e-8, 1.8e-7, 9e-7]
    
    # High-dose 2
    alpha_high_dose2     = [0.00198, 0.00119, 0.00467, 0.00687, 0.00977, 0.0259]
    err_alpha_high_dose2 = [0.00023, 0.00021, 0.00020, 0.00007, 0.00016, 0.0016]
    beta_high_dose2      = [-2.65e-6, 2.99e-6, -6.4e-7, 4.1e-7, -2.68e-6, -3.30e-6]
    err_beta_high_dose2  = [1e-7, 9e-8, 8e-8, 2e-8, 7e-8, 7e-8]
    
    low_dose_avg1 = []
    low_dose_avg2 = []
    medium_dose_avg1 = []
    high_dose_avg1 = []
    high_dose_avg2 = []
    for ii in range(len(millivolt_low_dose)):
        low_dose_avg1.append(alpha_low_dose1[ii]/beta_low_dose1[ii])
        low_dose_avg2.append(alpha_low_dose2[ii]/beta_low_dose2[ii])
    for ii in range(len(millivolt_medium_dose)):
        medium_dose_avg1.append(alpha_medium_dose1[ii]/beta_medium_dose1[ii])
    for ii in range(len(millivolt_high_dose)):
        high_dose_avg1.append(alpha_high_dose1[ii]/beta_high_dose1[ii])
        high_dose_avg2.append(alpha_high_dose2[ii]/beta_high_dose2[ii])
    
    print("Low-dose 1: "+str(low_dose_avg1))
    print(np.mean(low_dose_avg1))
    print("Low-dose 2: "+str(low_dose_avg2))
    print(np.mean(low_dose_avg2))
    print("Medium-dose 1: "+str(medium_dose_avg1))
    print(np.mean(medium_dose_avg1))
    print("High-dose 1: "+str(high_dose_avg1))
    print(np.mean(high_dose_avg1))
    print("High-dose 2: "+str(high_dose_avg2))
    print(np.mean(high_dose_avg2))
    
    # Beta-conversion into "kilo" before plotting.
    for i in range(len(millivolt_high_dose)):
        try:
            beta_low_dose1[i] = beta_low_dose1[i]*1e3
            beta_low_dose2[i] = beta_low_dose2[i]*1e3
            err_beta_low_dose1[i] = err_beta_low_dose1[i]*1e3
            err_beta_low_dose2[i] = err_beta_low_dose2[i]*1e3
        except IndexError:
            pass # Skip this one.
        try:
            beta_medium_dose1[i] = beta_medium_dose1[i]*1e3
            err_beta_medium_dose1[i] = err_beta_medium_dose1[i]*1e3
        except IndexError:
            pass # Skip this one.
        try:
            beta_high_dose1[i] = beta_high_dose1[i]*1e3
            beta_high_dose2[i] = beta_high_dose2[i]*1e3
            err_beta_high_dose1[i] = err_beta_high_dose1[i]*1e3
            err_beta_high_dose2[i] = err_beta_high_dose2[i]*1e3
        except IndexError:
            pass # Skip this one.
    
    '''# Plot!
    fig, axs = plt.subplots(1, 3, figsize=(25, 9), sharey=True)

    # Subplot (a) Low-dose
    axs[0].plot(millivolt_low_dose, alpha_low_dose1, 'o-', label='Low-dose 1 α', color="#C4EE1C")
    axs[0].plot(millivolt_low_dose, beta_low_dose1, 's--', label='Low-dose 1 β', color="#C4EE1C")
    axs[0].plot(millivolt_low_dose, alpha_low_dose2, 'o-', label='Low-dose 2 α', color="#1C70EE")
    axs[0].plot(millivolt_low_dose, beta_low_dose2, 's--', label='Low-dose 2 β', color="#1C70EE")
    #axs[0].set_title('(a) Low-dose devices')
    axs[0].set_xlabel('Voltage [mV]', fontsize=33)
    ##axs[0].set_ylabel('α [Ω/s]\nβ [mΩ/s²]', fontsize=33)
    axs[0].set_ylabel('α [s⁻¹]\nβ [ks⁻²]', fontsize=33)
    axs[0].tick_params(axis='both', labelsize=24)
    axs[0].set_xlim(625,1075)
    axs[0].legend(fontsize=26, loc='upper left')

    # Subplot (b) Medium-dose
    axs[1].plot(millivolt_medium_dose, alpha_medium_dose1, 'o-', label='Medium-dose 1 α', color="#EE1C1C")
    axs[1].plot(millivolt_medium_dose, beta_medium_dose1, 's--', label='Medium-dose 1 β', color="#EE1C1C")
    #axs[1].set_title('(b) Medium-dose device')
    axs[1].set_xlabel('Voltage [mV]', fontsize=33)
    axs[1].tick_params(axis='both', labelsize=24)
    axs[1].set_xlim(625,1075)
    axs[1].legend(fontsize=26, loc='upper left')

    # Subplot (c) High-dose
    axs[2].plot(millivolt_high_dose, alpha_high_dose1, 'o-', label='High-dose 1 α', color="#1CEE70")
    axs[2].plot(millivolt_high_dose, beta_high_dose1, 's--', label='High-dose 1 β', color="#1CEE70")
    axs[2].plot(millivolt_high_dose, alpha_high_dose2, 'o-', label='High-dose 2 α', color="#C41CEE")
    axs[2].plot(millivolt_high_dose, beta_high_dose2, 's--', label='High-dose 2 β', color="#C41CEE")
    #axs[2].set_title('(c) High-dose devices', fontsize=33)
    axs[2].set_xlabel('Voltage [mV]', fontsize=33)
    axs[2].tick_params(axis='both', labelsize=24)
    axs[2].set_xlim(625,1075)
    axs[2].legend(fontsize=26, loc='upper left')

    # Layout adjustment
    plt.tight_layout()
    for ax in axs: ax.grid(True)
    
    # Save plot?
    if savepath != '':
        base, _ = os.path.splitext(savepath)
        plt.savefig(base+"alpha_and_beta_vs_voltage.png", dpi=164, bbox_inches='tight')
    
    # Show shits, part 1.
    plt.show()'''
    # Plot!
    fig, axs = plt.subplots(1, 3, figsize=(25, 9), sharey=True)

    # Low-dose
    axs[0].errorbar(
        millivolt_low_dose, alpha_low_dose1,
        yerr=err_alpha_low_dose1,
        fmt='o-', capsize=5, label='Low-dose 1 α', color="#EE1C1C"
    )
    axs[0].errorbar(
        millivolt_low_dose, beta_low_dose1,
        yerr=err_beta_low_dose1,
        fmt='s--', capsize=5, label='Low-dose 1 β', color="#EE1C1C"
    )

    axs[0].errorbar(
        millivolt_low_dose, alpha_low_dose2,
        yerr=err_alpha_low_dose2,
        fmt='o-', capsize=5, label='Low-dose 2 α', color="#C4EE1C"
    )
    axs[0].errorbar(
        millivolt_low_dose, beta_low_dose2,
        yerr=err_beta_low_dose2,
        fmt='s--', capsize=5, label='Low-dose 2 β', color="#C4EE1C"
    )
    axs[0].set_xlabel('Voltage [mV]', fontsize=33)
    axs[0].set_ylabel('α [s⁻¹]\nβ [ks⁻²]', fontsize=33)
    axs[0].tick_params(axis='both', labelsize=24)
    axs[0].set_xlim(625, 1075)
    axs[0].legend(fontsize=26, loc='upper left')

    # Medium-dose
    axs[1].errorbar(
        millivolt_medium_dose, alpha_medium_dose1,
        yerr=err_alpha_medium_dose1,
        fmt='o-', capsize=5, label='Medium-dose 1 α', color="#1CEE70"
    )
    axs[1].errorbar(
        millivolt_medium_dose, beta_medium_dose1,
        yerr=err_beta_medium_dose1,
        fmt='s--', capsize=5, label='Medium-dose 1 β', color="#1CEE70"
    )
    axs[1].set_xlabel('Voltage [mV]', fontsize=33)
    axs[1].tick_params(axis='both', labelsize=24)
    axs[1].set_xlim(625, 1075)
    axs[1].legend(fontsize=26, loc='upper left')

    # High-dose
    axs[2].errorbar(
        millivolt_high_dose, alpha_high_dose1,
        yerr=err_alpha_high_dose1,
        fmt='o-', capsize=5, label='High-dose 1 α', color="#1C70EE"
    )
    axs[2].errorbar(
        millivolt_high_dose, beta_high_dose1,
        yerr=err_beta_high_dose1,
        fmt='s--', capsize=5, label='High-dose 1 β', color="#1C70EE"
    )

    axs[2].errorbar(
        millivolt_high_dose, alpha_high_dose2,
        yerr=err_alpha_high_dose2,
        fmt='o-', capsize=5, label='High-dose 2 α', color="#C41CEE"
    )
    axs[2].errorbar(
        millivolt_high_dose, beta_high_dose2,
        yerr=err_beta_high_dose2,
        fmt='s--', capsize=5, label='High-dose 2 β', color="#C41CEE"
    )
    axs[2].set_xlabel('Voltage [mV]', fontsize=33)
    axs[2].tick_params(axis='both', labelsize=24)
    axs[2].set_xlim(625, 1075)
    axs[2].legend(fontsize=26, loc='upper left')

    # Finish layout
    plt.tight_layout()
    for ax in axs:
        ax.grid(True)

    # Save?
    if savepath != '':
        base, _ = os.path.splitext(savepath)
        plt.savefig(base+"alpha_and_beta_vs_voltage.png", dpi=164, bbox_inches='tight')

    plt.show()
    
    # We continue with alpha-vs-beta.
    
    from scipy.stats import pearsonr

    # Function to calculate Pearson coefficient and print it
    def print_pearson(alpha, beta, label):
        corr, _ = pearsonr(alpha, beta)
        print(f'Pearson coefficient for {label}: {corr:.3f}')

    # Calculate Pearson coefficients for each trace
    print_pearson(alpha_low_dose1, beta_low_dose1, 'Low-dose 1')
    print_pearson(alpha_low_dose2, beta_low_dose2, 'Low-dose 2')
    print_pearson(alpha_medium_dose1, beta_medium_dose1, 'Medium-dose 1')
    print_pearson(alpha_high_dose1, beta_high_dose1, 'High-dose 1')
    print_pearson(alpha_high_dose2, beta_high_dose2, 'High-dose 2')
    
    # Create subplots
    ## TODO do better figure management.
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Set common xlim and ylim
    xlim = (-5, max(alpha_low_dose1 + alpha_low_dose2 + alpha_medium_dose1 + alpha_high_dose1 + alpha_high_dose2))
    ylim = (min(beta_low_dose1 + beta_low_dose2 + beta_medium_dose1 + beta_high_dose1 + beta_high_dose2), +2)
    #        max(beta_low_dose1 + beta_low_dose2 + beta_medium_dose1 + beta_high_dose1 + beta_high_dose2)*1.05)

    # Low-dose plot
    axs[0].plot(alpha_low_dose1, beta_low_dose1, label="Low-dose 1", color="#C4EE1C", marker='o', linestyle='-')
    axs[0].plot(alpha_low_dose2, beta_low_dose2, label="Low-dose 2", color="#1C70EE", marker='o', linestyle='-')
    #axs[0].set_title('Low-dose devices')
    axs[0].set_xlabel(r'$\alpha~[\Omega/$s$]$', fontsize=33)
    axs[0].set_ylabel(r'$\beta~[$m$\Omega/$s${}^2]$', fontsize=33)
    axs[0].legend(fontsize=26)
    axs[0].tick_params(axis='both', labelsize=24)
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    # Medium-dose plot
    axs[1].plot(alpha_medium_dose1, beta_medium_dose1, label="Medium-dose 1", color="#EE1C1C", marker='o', linestyle='-')
    #axs[1].set_title('Medium-dose device')
    axs[1].set_xlabel(r'$\alpha~[\Omega/$s$]$', fontsize=33)
    axs[1].set_ylabel(r'$\beta~[$m$\Omega/$s${}^2]$', fontsize=33)
    axs[1].legend(fontsize=26)
    axs[1].tick_params(axis='both', labelsize=24)
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)

    # High-dose plot
    axs[2].plot(alpha_high_dose1, beta_high_dose1, label="High-dose 1", color="#1CEE70", marker='o', linestyle='-')
    axs[2].plot(alpha_high_dose2, beta_high_dose2, label="High-dose 2", color="#C41CEE", marker='o', linestyle='-')
    #axs[2].set_title('High-dose devices')
    axs[2].set_xlabel(r'$\alpha~[\Omega/$s$]$', fontsize=33)
    axs[2].set_ylabel(r'$\beta~[$m$\Omega/$s${}^2]$', fontsize=33)
    axs[2].legend(fontsize=26)
    axs[2].tick_params(axis='both', labelsize=24)
    axs[2].set_xlim(xlim)
    axs[2].set_ylim(ylim)
    
    # Layout adjustment
    plt.tight_layout()
    
    # Save plot?
    if savepath != '':
        base, _ = os.path.splitext(savepath)
        plt.savefig(base+"alpha_vs_beta.png", dpi=164, bbox_inches='tight')
    
    # Show shits.
    plt.show()

def validate_second_or_third_order_polynomial():
    ''' Using the residuals acquired from fitting to the second-order and
        third order polynomial, analyse the RMSE acquired from both.
        Then from the mean RMSE, establish whether the third- or second-order
        polynomial is a better model.
    '''
    
    # 2nd-order RMSE arrays
    second_low1    = np.array([0.5435788903698674, 0.6906367827838787, 0.2185641226827453, 0.17975408607550908, 0.12140715728057479])
    second_low2    = np.array([0.4224931398561787, 0.30712404088350104, 0.13903095908245294, 0.14882897990160818, 0.07893635029247786])
    second_medium1 = np.array([0.12899911478059992, 0.12258723519038636, 0.09198758248640586, 0.07847907489438669])
    second_high1   = np.array([0.12778871133630634, 0.10140555798298326, 0.08435368707868546, 0.07796627066582179, 0.07015128242597864, 0.07219376498554592])
    second_high2   = np.array([0.0951672366929808, 0.08649234324301436, 0.08321069498057651, 0.08814376551090108, 0.0666364360271154, 0.06630354158804268])

    # 3rd-order RMSE arrays
    third_low1    = np.array([0.4547883673216632, 0.6863639714426755, 0.21406553111595414, 0.16173417758696748, 0.09736348642654973])
    third_low2    = np.array([0.2810290259145776, 0.2557708548823828, 0.11887676308001224, 0.11179757730163585, 0.07243916472565719])
    third_medium1 = np.array([0.12088184996419875, 0.07838672681005657, 0.08847923797622029, 0.07828562352238277])
    third_high1   = np.array([0.1277764895384079, 0.09665049404423984, 0.07987339812104315, 0.07458773074291568, 0.06409819967559804, 0.06896571282839141])
    third_high2   = np.array([0.09505739700259591, 0.08244318673688883, 0.08294473249795886, 0.08754839837621933, 0.06334988410018574, 0.06282052010268772])

    # Group all device-wise means together
    second_means = np.array([
        np.mean(second_low1),
        np.mean(second_low2),
        np.mean(second_medium1),
        np.mean(second_high1),
        np.mean(second_high2)
    ])

    third_means = np.array([
        np.mean(third_low1),
        np.mean(third_low2),
        np.mean(third_medium1),
        np.mean(third_high1),
        np.mean(third_high2)
    ])
    
    # Compare models
    diff = second_means - third_means
    relative_improvement = diff / second_means * 100 # Units of '% improvement'

    # Paired t-test
    t_stat, p_value = ttest_rel(second_means, third_means)

    print("Mean RMSE (2nd order):", np.mean(second_means))
    print("Mean RMSE (3rd order):", np.mean(third_means))
    print("Average improvement [% improvement]:", np.mean(relative_improvement))
    print("Per-device improvement [% improvement]:", relative_improvement)
    print("Paired t-test: t =", t_stat, ", p =", p_value)
