import numpy as np

def feature_expand(x_jet, jet_num):
    """
    Execute physics processing to the dataset.

    :param x_jet: x subset with a unique jet number.
    :param jet_num: jet number identifier.
    :return: processed x subset.
    """
    print("Executing feature engineering pipeline...")
    x_total = feature_to_energy(x_jet, jet_num)
    x_total = np.concatenate((x_total, add_features(x_jet, jet_num)), axis=1)

    return x_total


def feature_to_energy(x_jet, jet_num):
    """
    Assign function to every label to match the units of energy^2

    :param x_jet: x subset with a unique jet number.
    :param jet_num: jet number identifier.
    :return: processed x subset.
    """
    index_label = get_feature_meaning_index(jet_num)

    mass_term = x_jet[:, index_label['mass']]
    tX_mass = np.c_[mass_term, np.power(mass_term, 2)]

    momentum_term = x_jet[:, index_label['momentum']]
    tX_momentum = np.c_[momentum_term, np.power(momentum_term, 2)]

    energy_term = x_jet[:, index_label['energy']]
    tX_energy = np.c_[energy_term, np.power(energy_term, 2)]

    tX_total = np.concatenate((tX_mass, tX_momentum, tX_energy), axis=1)

    return tX_total


def get_feature_meaning_index(jet_num):
    """
    Return indexes of measurements features depending on the jet number.

    :param jet_num: jet number identifier.
    :return: dictionary containing physics information.
    """
    index_label = {'mass': [], 'energy': [], 'momentum': [],
                   'eta': [], 'centrality': [], 'angle': [], 'ratio': []}
    if jet_num == 0:
        index_label['mass']       = [0,1,2]
        index_label['energy']     = [15,17]
        index_label['momentum']   = [3,5,6,9,12]
    elif jet_num == 1:
        index_label['mass']       = [0,1,2]
        index_label['energy']     = [15,17]
        index_label['momentum']   = [3,5,6,9,12,18,21]
    else:
        index_label['mass']       = [0,1,2,5]
        index_label['energy']     = [19,21]
        index_label['momentum']   = [3,8,9,13,16,22,25,28]
    return index_label


def add_features(x_jet, jet_num):
    """
    Consider the attributes of other features, such as angle, eta, centrality

    :param x_jet: x subset with a unique jet number.
    :param jet_num: jet number identifier.
    :return: processed x subset.
    """

    # Consider entry 7. DER_deltar_tau_lep & 10. DER_pt_ratio_lep_tau are related to hadronic tau and lepton
    index_entry = (7, 10) if jet_num > 1 else (4, 7)
    # Momentum effect
    momentum_index = 8 if jet_num > 1 else 5
    momentum_deltar  = np.c_[x_jet[:, index_entry[0]] * x_jet[:, momentum_index], x_jet[:, index_entry[0]] * np.power(x_jet[:, momentum_index], 2)]
    # From experiment proved irrevelant
    # momentum_ptratio = np.c_[ filt_x[:,index_entry[1]] * filt_x[:,momentum_index], filt_x[:,index_entry[1]] * np.power(filt_x[:,momentum_index], 2) ]

    # Mass effect
    # From experiment showed that need to consider inverse also
    # Transverse mass
    mass_trans_deltar     = np.c_[x_jet[:, index_entry[0]] * x_jet[:, 1], x_jet[:, index_entry[0]] * np.power(x_jet[:, 1], 2)]
    mass_trans_deltar_inv = np.c_[1 / x_jet[:, index_entry[0]] * x_jet[:, 1], 1 / x_jet[:, index_entry[0]] * np.power(x_jet[:, 1], 2)]
    # Invariant mass
    mass_invar_deltar     = np.c_[x_jet[:, index_entry[0]] * x_jet[:, 2], x_jet[:, index_entry[0]] * np.power(x_jet[:, 2], 2)]
    mass_invar_deltar_inv = np.c_[1 / x_jet[:, index_entry[0]] * x_jet[:, 2], 1 / x_jet[:, index_entry[0]] * np.power(x_jet[:, 2], 2)]

    tx_total = np.c_[ momentum_deltar, mass_trans_deltar, mass_trans_deltar_inv, mass_invar_deltar, mass_invar_deltar_inv ]


    # Consider entry 11: Centrality.
    # 11 DER_met_phi_centrality is related to the missing transverse vector (entry 1 & 8)
    index_entry = 11 if jet_num > 1 else 8
    # mass
    mass_trans_met = np.c_[x_jet[:, index_entry] * x_jet[:, 1], x_jet[:, index_entry] * np.power(x_jet[:, 1], 2)]

    # momentum
    momentum_trans_met3 = np.c_[x_jet[:, index_entry] * x_jet[:, 3], x_jet[:, index_entry] * np.power(x_jet[:, 3], 2)]
    momentum_trans_met8 = np.c_[x_jet[:, index_entry] * x_jet[:, momentum_index], x_jet[:, index_entry] * np.power(x_jet[:, momentum_index], 2)]

    tx_total = np.c_[ tx_total ,mass_trans_met, momentum_trans_met3, momentum_trans_met8 ]


    # Consider entry 4,5,6,12. As they are a pair
    if jet_num > 1:
        # entry 4: pseudo-rapidity separation at most linearly change jetjet mass of entry 5
        jetjet_sep = np.c_[x_jet[:, 4] * x_jet[:, 5], x_jet[:, 4] * np.power(x_jet[:, 5], 2)]
        # entry 6: pseudo-rapidity product may linearly or inversely change mass^2
        jetjet_pro = np.c_[x_jet[:, 6] * x_jet[:, 5], x_jet[:, 6] * np.power(x_jet[:, 5], 2)]
        # entry 12: centrality of pseudo-rapidity
        jetjet_cen = np.c_[x_jet[:, 12] * x_jet[:, 5], x_jet[:, 12] * np.power(x_jet[:, 5], 2)]

        tx_total = np.concatenate((tx_total, jetjet_sep, jetjet_pro, jetjet_cen), axis=1)

    # entry 13-15, the hadronic tau
    index_entry = 13 if jet_num > 1 else 9
    momentum_tau = np.c_[x_jet[:, index_entry + 1] * x_jet[:, index_entry], x_jet[:, index_entry + 1] * np.power(x_jet[:, index_entry], 2)]

    # entry 16-18, the lepton
    index_entry = 16 if jet_num > 1 else 12
    momentum_lep = np.c_[x_jet[:, index_entry + 1] * x_jet[:, index_entry], x_jet[:, index_entry + 1] * np.power(x_jet[:, index_entry], 2)]

    # entry 19, 20, the missing transverse energy
    index_entry = 19 if jet_num > 1 else 15
    energy_rapidity = np.tan(x_jet[:, index_entry + 1])
    energy_trans = np.c_[energy_rapidity * x_jet[:, index_entry], energy_rapidity * np.power(x_jet[:, index_entry], 2)]

    tx_total = np.c_[ tx_total, momentum_tau, momentum_lep, energy_trans]

    # entry 22-24, the leading jet
    if jet_num > 0:
        index_entry = 22 if jet_num > 1 else 18
        momentum_leadjet = np.c_[x_jet[:, index_entry + 1] * x_jet[:, index_entry], x_jet[:, index_entry + 1] * np.power(x_jet[:, index_entry], 2)]

        tx_total = np.c_[ tx_total, momentum_leadjet]

    if jet_num > 1:
        index_entry = 25
        momentum_secondjet = np.c_[x_jet[:, index_entry + 1] * x_jet[:, index_entry], x_jet[:, index_entry + 1] * np.power(x_jet[:, index_entry], 2)]

        tx_total = np.c_[ tx_total, momentum_secondjet]


    return tx_total