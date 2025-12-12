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
import matplotlib.pyplot as plt

"""
# Arguments used for the T1/Tp plot in the 2025 paper by C. Križan et al.

ppq(
[[5.9532e9, 6.0659e9, 6.1935e9, 6.3155e9, 6.4513e9, 6.5924e9, 6.7446e9, 6.8918e9],
 [5.9532e9, 6.0659e9, 6.1935e9, 6.3155e9, 6.4513e9, 6.5924e9, 6.7446e9, 6.8918e9],
 [5.9509e9, 6.0668e9, 6.1908e9, 6.3127e9, 6.4554e9, 6.5811e9, 6.7419e9, 6.8838e9],
 [5.9509e9, 6.0668e9, 6.1908e9, 6.3127e9, 6.4554e9, 6.5811e9, 6.7419e9, 6.8838e9]
],
[[4.8514e9, 4.9853e9, 4.1680e9, 4.9750e9, 5.5673e9, 5.4177e9, 5.3942e9, 5.5847e9],
 [4.5518e9, None,     4.0851e9, None,     None,     4.9811e9, 5.1232e9, 5.2863e9],
 [None, None,     4.4353e9, None,     5.1634e9, None,     4.9492e9, None    ],
 [None,     None,     4.2354e9, None,     4.8590e9, None,     4.8039e9, None    ]
],
[[-3108e3, -1764e3, -1024e3, -1435e3, -2887e3, -1671e3, -1434e3, -1627e3],
 [-1030e3, None,    -411e3,  None,    None,    -899e3,  -965e3,  -1159e3],
 [None, None,    -778e3,  None,    -1495e3, None,    -857e3,  None],
 [None,    None,    -620e3,  None,    -2054e3, None,    -574e3,  None]
],
[[-214.4e6, -210e6, -222.2e6, -210.3e6, -202.3e6, -204.8e6, -201.7e6, -203.2e6],
 [-214.4e6, None,   -222.2e6, None,     None,     -197.6e6, -207.2e6, -205.3e6],
 [None, None,   -216.8e6, None,     -207.4e6, None,     -208.9e6, None],
 [None,     None,   -222.0e6, None,     -207.4e6, None,     -208.9e6, None]
],
[[818e3, 707e3, 730e3, 692e3, 767e3, 812e3, 924e3, 898e3],
[818e3, 707e3, 730e3, 692e3, 767e3, 812e3, 924e3, 898e3],
[646e3, 647e3, 754e3, 735e3, 777e3, 735e3, 815e3,1011e3],
[646e3, 647e3, 754e3, 735e3, 777e3, 735e3, 815e3,1011e3]
],
[[55e-6, 38e-6, 95e-6, 48e-6, 29e-6, 33e-6, 35e-6, 32e-6],
 [46e-6, None,  54e-6, None,  None,  12e-6, 43e-6, 34e-6],
 [28e-6, None,  48e-6, None,  40e-6, None,  39e-6, None],
 [None,  None,  50e-6, None,  61e-6, None,  5e-6,  None]
])
"""

def plot_qubit_quality_vs_t1_tp(
    list_of_lists_of_resonator_frequencies_Hz,
    list_of_lists_of_qubit_frequencies_Hz,
    list_of_lists_of_dispersive_shift_2chi_Hz,
    list_of_lists_of_anharmonicities_Hz,
    list_of_lists_of_resonator_linewidth_kappa_Hz,
    list_of_lists_of_T1_s,
    transpose_axes = False,
    savepath = ''
    ):
    ''' Plot qubit quality factor versus T₁/T_{purcell}.
        
        Using N.T. Bronn 2015, https://ieeexplore.ieee.org/document/7156088,
        professor Per Delsing found how to calculate the Purcell decay
        time. And, the qubit-resonator coupling factor.
        
        list_of_lists_of_{rest of variable name}:
        assumed to be in the format of list( list( ) ) -- example, let's say
        you had four qubits studied in the first round, and five qubits
        studied in the second round. The correct argument would then be:
            list_of_lists_of_resonator_frequencies_Hz = [ [1,2,3,4] , [1,2,3,4,5] ]
            list_of_lists_of__qubit_frequencies_Hz    = [ [1,2,3,4] , [1,2,3,4,5] ]
        
        Note here that each sub-list must have a length correspondence.
            assert len( list_of_lists_of_resonator_frequencies_Hz[0] ) == len( list_of_lists_of__qubit_frequencies_Hz[o] )
        
        And, this assertion must be valid for all list_of_lists_of_{parameter}!
    '''
    
    # Define formulas.
    def calculate_g(
        omega_r,
        omega_q,
        chi,
        eta
        ):
        ''' omega_r: Resonator angular frequency.
            omega_q: Qubit angular frequency.
            chi:     Dispersive shift.
            eta:     Transmon anharmonicity.
        '''
        Delta = omega_q - omega_r
        return np.sqrt((Delta * chi) * (eta + Delta)/eta)
    
    def calculate_Tp(
        omega_r,
        omega_q,
        g,
        kappa
        ):
        ''' omega_r: Resonator angular frequency.
            omega_q: Qubit angular frequency.
            g:       Qubit-resontor coupling strength, see "calculate_g".
            kappa:   Resonator linewidth.
        '''
        Delta = omega_q - omega_r
        simplified_term = (Delta**2) / ((g**2)*kappa)
        
        # Insert the terms of the correction factor expression.
        term1 = 1
        term2 = -2*( Delta/omega_q )
        term3 = (5/4) * ( (Delta**2)/(omega_q**2) )
        term4 = -(1/4) * ( (Delta**3)/(omega_q**3) )
        
        # Important! This is an exact expression. Not an expansion thing.
        corr_factor = term1 + term2 + term3 + term4
        return simplified_term * corr_factor, (Delta/(2*np.pi))
    
    def calculate_qubit_quality_factor( omega_q, T1 ):
        return omega_q * T1
    
    # Create figure for plotting.
    fig, ax1 = plt.subplots(1, figsize=(12.6, 11.083), sharey=False)
    
    # For plotting purposes, keep track of the highest Y value in the plot.
    highest_ylim = 0.0
    
    # Loop through all the list_of_lists_of objects.
    # Each entry corresponds to one qubit.
    list_of_set_colours = ["#EE1C1C", "#1CEE70", "#1C70EE", "#C41CEE", "#C4EE1C"]
    list_of_qubit_scatter_symbols = ['s', '^', 'o', 'v', 'd', '*', 'x', 'p']
    for ii in range(len(list_of_lists_of_resonator_frequencies_Hz)):
        
        # Prepare axes.
        t1_tp_axis = []
        qubit_quality_axis = []
        
        # Get data.
        curr_omega_r_set = []
        curr_omega_q_set = []
        for kk in range(len(list_of_lists_of_resonator_frequencies_Hz[ii])):
            try:
                curr_omega_r_set.append( 2*np.pi * list_of_lists_of_resonator_frequencies_Hz[ii][kk] )
            except TypeError:
                curr_omega_r_set.append( None )
            try:
                curr_omega_q_set.append( 2*np.pi * list_of_lists_of_qubit_frequencies_Hz[ii][kk]     )
            except:
                curr_omega_q_set.append( None )
        curr_2chi_set   = list_of_lists_of_dispersive_shift_2chi_Hz[ii]
        curr_eta_set   = list_of_lists_of_anharmonicities_Hz[ii]
        curr_kappa_set = list_of_lists_of_resonator_linewidth_kappa_Hz[ii]
        curr_T1_set    = list_of_lists_of_T1_s[ii]
        
        # Statistics is nice. Let's prepare a T_p list.
        curr_Tp_list = []
        curr_Delta_Hz_list = []
        
        ## Let's make one scatter mass in the plot from index ii.
        
        # Calculate the X-axis T₁/T_p, and the Y-axis Q_qb.
        for jj in range(len(curr_omega_r_set)):
            
            # Are all parameters for this entry not None?
            if (curr_omega_r_set[jj] is not None) and (curr_omega_q_set[jj] is not None) and (curr_2chi_set[jj] is not None) and (curr_eta_set[jj] is not None) and (curr_kappa_set[jj] is not None):
            
                # Get the current coupling strength.
                curr_g = calculate_g(
                    omega_r = curr_omega_r_set[jj],
                    omega_q = curr_omega_q_set[jj],
                    chi = curr_2chi_set[jj]/2, # NOTE! g expects chi, not 2chi.
                    eta = curr_eta_set[jj]
                )
                
                # Calculate the Purcell decay time for this entry.
                curr_Tp, curr_Delta_Hz = calculate_Tp(
                    omega_r = curr_omega_r_set[jj],
                    omega_q = curr_omega_q_set[jj],
                    g = curr_g,
                    kappa = curr_kappa_set[jj]
                )
                
                # Calculate T₁/T_p for current entry.
                t1_tp_axis.append( curr_T1_set[jj] / curr_Tp )
                
                print("Set "+str(ii+1)+", Qubit "+str(jj+1)+", calculated Tp: "+str(curr_Tp)+" [s], calculated Delta: "+str(curr_Delta_Hz)+" [Hz]")
                curr_Tp_list.append(curr_Tp)
                curr_Delta_Hz_list.append(curr_Delta_Hz)
            
                ## Now, get qubit quality factor for the Y-axis.
                curr_quality_factor = calculate_qubit_quality_factor( 
                    omega_q = curr_omega_q_set[jj],
                    T1 = curr_T1_set[jj],
                )
                if curr_quality_factor > highest_ylim:
                    highest_ylim = curr_quality_factor
                qubit_quality_axis.append( curr_quality_factor )
                
                # Detect whether we are plotting a cross;
                # if we are, then increase its size.
                if (jj == 6):
                    bump_up_size = 140
                else:
                    bump_up_size = 0
                
                # At this point, we have one scatter dot to plot.
                if not transpose_axes:
                    try:
                        ax1.scatter(t1_tp_axis[-1], qubit_quality_axis[-1]/1e6, s=130+bump_up_size, label=None, marker=list_of_qubit_scatter_symbols[jj], color=list_of_set_colours[ii])
                    except ValueError:
                        print(list_of_set_colours[ii])
                else:
                    ax1.scatter(qubit_quality_axis[-1]/1e6, t1_tp_axis[-1], s=130+bump_up_size, label=None, marker=list_of_qubit_scatter_symbols[jj], color=list_of_set_colours[ii])
            
            else:
                # Then this entry is a None. Something is missing.
                t1_tp_axis.append( None )
                qubit_quality_axis.append( None )
        
        # At this point, we can print some stats about curr_Tp_list
        curr_Tp_array = np.array(curr_Tp_list)
        curr_Delta_Hz_array = np.array(curr_Delta_Hz_list)
        print("Set "+str(ii+1)+", Tp mean: "+str(np.mean(curr_Tp_array))+" [s], Tp std. deviation: "+str(np.std(curr_Tp_array, ddof=1))+" [s]")
        print("Set "+str(ii+1)+", Delta mean: "+str(np.mean(curr_Delta_Hz_array/1e9))+" [GHz], Delta std. deviation: "+str(np.std(curr_Delta_Hz_array/1e9, ddof=1))+" [GHz]")
    
    # Labels and formatting stuff.
    ax1.grid()
    ## Adjust to [10^6] in quality factors, to plot correctly.
    highest_ylim /= 1e6
    if not transpose_axes:
        ax1.set_xlabel(r"$T_1 / T_p$ [-]", fontsize=33)
        ax1.set_ylabel("Qubit quality factor [$10^6$]", fontsize=33)
        ax1.set_xlim(-0.05, 1.05)
        ## ax1.set_ylim(-highest_ylim * 0.05, highest_ylim * 1.05)
        ##ax1.set_ylim(0.0, highest_ylim * 1.05)
        ax1.set_ylim(0.0, 2.603896)
        
    else:
        # Note that the metric keeping track of the highest T₁/T_p
        # in the plot, is still labeled "highest_ylim", even though it's
        # now an x_lim...
        ax1.set_ylabel(r"$T_1 / T_p$ [-]", fontsize=33)
        ax1.set_xlabel("Qubit quality factor [$10^6$]", fontsize=33)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlim(-highest_ylim * 0.05, highest_ylim * 1.05)
    ax1.tick_params(axis='both', labelsize=26)
    
    # Tight layout.    
    plt.tight_layout()
    
    # Save plots?
    if savepath != '':
        plt.savefig(savepath, dpi=164, bbox_inches='tight')
    
    # Show stuff!
    plt.show()
    