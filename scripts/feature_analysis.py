
import numpy as np

#############################################################################
# USAGE:
## if the data is defined as below:
#
# y_jet0  = y[tX[:, i_PRI]==0]
# tx_jet0 = tX[tX[:, i_PRI]==0]

# y_jet1  = y[ tX[:, i_PRI] == 1]
# tx_jet1 = tX[tX[:, i_PRI] == 1]

# y_jet2  = y[ tX[:, i_PRI] > 1]
# tx_jet2 = tX[tX[:, i_PRI] > 1]
##----------------------------------
# Then it can be executed like this
# tx_0_filtered = np.delete(tx_jet0, [4,5,6,12,22,23,24,25,26,27,28], axis=1)
# tx_1_filtered = np.delete(tx_jet1, [4,5,6,12,22,26,27,28], axis=1)
# tx_2_filtered = np.delete(tx_jet2, [22], axis=1)

# tx_0_filtered = fill_nan(fill_missing(tx_0_filtered))
# tx_1_filtered = fill_nan(fill_missing(tx_1_filtered))
# tx_2_filtered = fill_nan(fill_missing(tx_2_filtered))

# tx_train_0 = featureExpand(tx_0_filtered, 0)
# tx_train_1 = featureExpand(tx_1_filtered, 1)
# tx_train_2 = featureExpand(tx_2_filtered, 2)
##############################################################################
# Then you can delete 



def get_featureMeaning_index(jet_num):
    index_label = {'mass': [], 'energy': [], 'momentum': [], \
        'eta': [], 'centrality': [], 'angle': [], 'ratio': []}
    if jet_num == 0:
        index_label['mass']       = [0,1,2]
        index_label['energy']     = [15,17]
        index_label['momentum']   = [3,5,6,9,12,18]
    elif jet_num == 1:
        index_label['mass']       = [0,1,2]
        index_label['energy']     = [15,17]
        index_label['momentum']   = [3,5,6,9,12,18,21]
    else:
        index_label['mass']       = [0,1,2,5]
        index_label['energy']     = [19,21]
        index_label['momentum']   = [3,8,9,13,16,22,25,28]
    return index_label


def feature2Energy(filtered_x, jet_num):
    """Assign function to every label to match the units of energy^2"""
    index_label = get_featureMeaning_index(jet_num)

    mass_term = filtered_x[:,index_label['mass']]
    tX_mass = np.c_[mass_term, np.power(mass_term, 2)]

    momentum_term = filtered_x[:, index_label['momentum']]
    tX_momentum = np.c_[momentum_term, np.power(momentum_term, 2)]

    energy_term = filtered_x[:, index_label['energy']]
    tX_energy = np.c_[energy_term, np.power(energy_term, 2)]

    tX_total = np.concatenate((np.ones((filtered_x.shape[0], 1)), tX_mass, tX_momentum, tX_energy), axis=1)
    
    return tX_total




def addFeat(filt_x, jet_num):
    """Consider the attributes of other features, such as angle, eta, centrality"""

    # Consider entry 7. DER_deltar_tau_lep & 10. DER_pt_ratio_lep_tau are related to hadronic tau and lepton
    index_entry = (7, 10) if jet_num > 1 else (4, 7)
    # Momentum effect
    momentum_index = 8 if jet_num > 1 else 5
    momentum_deltar  = np.c_[ filt_x[:,index_entry[0]] * filt_x[:,momentum_index], filt_x[:,index_entry[0]] * np.power(filt_x[:,momentum_index], 2) ]
    # From experiment proved irrevelant
    # momentum_ptratio = np.c_[ filt_x[:,index_entry[1]] * filt_x[:,momentum_index], filt_x[:,index_entry[1]] * np.power(filt_x[:,momentum_index], 2) ]

    # Mass effect
    # From experiment showed that need to consider inverse also
    # Transverse mass
    mass_trans_deltar     = np.c_[ filt_x[:,index_entry[0]] * filt_x[:,1], filt_x[:,index_entry[0]] * np.power(filt_x[:,1], 2) ]
    mass_trans_deltar_inv = np.c_[ 1/filt_x[:,index_entry[0]] * filt_x[:,1], 1/filt_x[:,index_entry[0]] * np.power(filt_x[:,1], 2) ]
    # Invariant mass
    mass_invar_deltar     = np.c_[ filt_x[:,index_entry[0]] * filt_x[:,2], filt_x[:,index_entry[0]] * np.power(filt_x[:,2], 2) ]
    mass_invar_deltar_inv = np.c_[ 1/filt_x[:,index_entry[0]] * filt_x[:,2], 1/filt_x[:,index_entry[0]] * np.power(filt_x[:,2], 2) ]
    
    tx_total = np.c_[ momentum_deltar, mass_trans_deltar, mass_trans_deltar_inv, mass_invar_deltar, mass_invar_deltar_inv ]


    # Consider entry 11: Centrality. 
    # 11 DER_met_phi_centrality is related to the missing transverse vector (entry 1 & 8)
    index_entry = 11 if jet_num > 1 else 8
    # mass
    mass_trans_met = np.c_[ filt_x[:,index_entry] * filt_x[:,1], filt_x[:,index_entry] * np.power(filt_x[:,1], 2) ]

    # momentum
    momentum_trans_met = np.c_[ filt_x[:,index_entry] * filt_x[:,momentum_index], filt_x[:,index_entry] * np.power(filt_x[:,momentum_index], 2) ]

    tx_total = np.c_[ tx_total ,mass_trans_met, momentum_trans_met ]


    # Consider entry 4,5,6,12. As they are a pair
    if jet_num > 1:
        # entry 4: pseudo-rapidity separation at most linearly change jetjet mass of entry 5
        jetjet_sep = np.c_[ filt_x[:,4] * filt_x[:,5], filt_x[:,4] * np.power(filt_x[:,5], 2) ]
        # entry 6: pseudo-rapidity product may linearly or inversely change mass^2
        jetjet_pro = np.c_[ filt_x[:,6] * filt_x[:,5], filt_x[:,6] * np.power(filt_x[:,5], 2) ]
        # entry 12: centrality of pseudo-rapidity
        jetjet_cen = np.c_[ filt_x[:,12] * filt_x[:,5], filt_x[:,12] * np.power(filt_x[:,5], 2) ]

        tx_total = np.concatenate((tx_total, jetjet_sep, jetjet_pro, jetjet_cen), axis=1)

    # entry 13-15, the hadronic tau
    index_entry = 13 if jet_num > 1 else 9
    momentum_tau = np.c_[ filt_x[:,index_entry+1] * filt_x[:,index_entry], filt_x[:,index_entry+1] * np.power(filt_x[:,index_entry], 2) ]

    # entry 16-18, the lepton
    index_entry = 16 if jet_num > 1 else 12
    momentum_lep = np.c_[ filt_x[:,index_entry+1] * filt_x[:,index_entry], filt_x[:,index_entry+1] * np.power(filt_x[:,index_entry], 2) ]

    # entry 19, 20, the missing transverse energy
    index_entry = 19 if jet_num > 1 else 15
    energy_rapidity = np.tan( filt_x[:,index_entry+1] )
    energy_trans = np.c_[ energy_rapidity * filt_x[:,index_entry], energy_rapidity * np.power(filt_x[:,index_entry], 2) ]

    tx_total = np.c_[ tx_total, momentum_tau, momentum_lep, energy_trans]

    # entry 22-24, the leading jet
    if jet_num > 0:
        index_entry = 22 if jet_num > 1 else 18
        momentum_leadjet = np.c_[ filt_x[:,index_entry+1] * filt_x[:,index_entry], filt_x[:,index_entry+1] * np.power(filt_x[:,index_entry], 2) ]

        tx_total = np.c_[ tx_total, momentum_leadjet]
    
    if jet_num > 1:
        index_entry = 25
        momentum_secondjet = np.c_[ filt_x[:,index_entry+1] * filt_x[:,index_entry], filt_x[:,index_entry+1] * np.power(filt_x[:,index_entry], 2) ]

        tx_total = np.c_[ tx_total, momentum_secondjet]

    
    return tx_total


def featureExpand(filt_x, jet_num):
    x_total = feature2Energy(filt_x, jet_num)
    x_total = np.concatenate((x_total, addFeat(filt_x, jet_num)), axis=1)
    
    return x_total


def fix_missing_values(x):
    x[x==-999] = np.nan
    return x


def fix_nan_values(x):
    x = np.nan_to_num(x)
    return x