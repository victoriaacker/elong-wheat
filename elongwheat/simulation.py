# -*- coding: latin-1 -*-

from __future__ import division  # use "//" to do integer division

import warnings
import copy

import numpy as np
import pandas as pd

from elongwheat import model
from elongwheat import parameters

"""
    elongwheat.simulation
    ~~~~~~~~~~~~~~~~~~

    The module :mod:`elongwheat.simulation` is the front-end to run the Elong-Wheat model.

    :copyright: Copyright 2014-2015 INRA-ECOSYS, see AUTHORS.
    :license: see LICENSE for details.

"""

#: the inputs needed by ElongWheat
HIDDENZONE_INPUTS = ['leaf_is_growing', 'internode_is_growing', 'leaf_pseudo_age', 'internode_pseudo_age', 'leaf_pseudostem_length', 'internode_distance_to_emerge', 'leaf_L', 'internode_L',
                     'lamina_Lmax', 'width', 'thickness', 'width_prev', 'thickness_prev', 'sheath_Lmax', 'temperature',
                     'hiddenzone_age', 'leaf_Wmax', 'SSLW', 'LSSW', 'leaf_is_emerged', 'internode_Lmax', 'internode_Lmax_lig', 'LSIW',
                     'internode_is_visible', 'sucrose', 'amino_acids', 'fructan', 'proteins', 'leaf_enclosed_mstruct', 'leaf_enclosed_Nstruct', 'internode_enclosed_mstruct',
                     'internode_enclosed_Nstruct', 'mstruct', 'is_over', 'mean_conc_sucrose', 'sheath_is_emerged']
ELEMENT_INPUTS = ['length', 'Wmax', 'is_growing', 'age', 'is_over']
AXIS_INPUTS = ['xylem_water_potential', 'SAM_temperature', 'delta_teq', 'teq_since_primordium', 'status', 'nb_leaves', 'GA', 'SAM_height', 'cohort', 'sum_TT']

#: the outputs computed by ElongWheat
# TODO : add be default all the attributes of the class HiddenZoneInit and ElementInit, and define which attribute is set by growthwheat.parameters or elongwheat.parameters
HIDDENZONE_OUTPUTS = ['sheath_is_emerged', 'res', 'delta_teq', 'leaf_is_growing', 'internode_is_growing', 'leaf_pseudo_age', 'delta_leaf_pseudo_age', 'internode_pseudo_age', 'delta_internode_pseudo_age', 'leaf_pseudostem_length', 'lamina_Lmax',
                      'width_prev', 'thickness_prev', 'sheath_Lmax', 'temperature', 'SL_ratio',
                      'delta_internode_distance_to_emerge', 'leaf_L', 'delta_leaf_L', 'internode_L','delta_internode_L', 'leaf_Wmax',
                      'SSLW', 'LSSW', 'leaf_is_emerged', 'internode_Lmax', 'internode_Lmax_lig', 'LSIW', 'internode_is_visible', 'sucrose', 'amino_acids', 'fructan', 'proteins',
                      'leaf_enclosed_mstruct', 'init_leaf_L', 'width', 'thickness',
                      'leaf_enclosed_Nstruct', 'internode_enclosed_mstruct', 'internode_enclosed_Nstruct', 'mstruct', 'is_over', 'ratio_DZ',
                      'mean_conc_sucrose', 'leaf_is_remobilizing', 'internode_is_remobilizing']
ELEMENT_OUTPUTS = ['length', 'Wmax', 'is_growing', 'sucrose', 'amino_acids', 'fructan', 'proteins', 'nitrates', 'starch', 'cytokinins', 'width', 'thickness',
                   'mstruct', 'Nstruct', 'age', 'Nresidual', 'max_proteins', 'senesced_length_element', 'green_area', 'max_mstruct',
                   'senesced_mstruct', 'is_over']
AXIS_OUTPUTS = ['xylem_water_potential', 'SAM_temperature', 'delta_teq', 'res', 'delta_teq_roots', 'teq_since_primordium', 'status', 'nb_leaves', 'GA', 'SAM_height', 'cohort', 'sum_TT']

#: the inputs and outputs of ElongWheat.
HIDDENZONE_INPUTS_OUTPUTS = sorted(set(HIDDENZONE_INPUTS + HIDDENZONE_OUTPUTS))
ELEMENT_INPUTS_OUTPUTS = sorted(set(ELEMENT_INPUTS + ELEMENT_OUTPUTS))
AXIS_INPUTS_OUTPUTS = sorted(set(AXIS_INPUTS + AXIS_OUTPUTS))

#: topology colums for ligule height dataframe
LIGULE_TOPOLOGY_COLUMNS = ['axis_id', 'phytomer', 'ligule height']


class SimulationError(Exception):
    pass


class SimulationRunError(SimulationError):
    pass


class Simulation(object):
    """The Simulation class allows to initialize and run a simulation.
    """

    def __init__(self, delta_t=1, update_parameters=None):
        """The inputs of elong-Wheat."""
        #: `inputs` is a dictionary of dictionaries:
        #:     {'hiddenzone': {(plant_index, axis_label, metamer_index): {hiddenzone_input_name: hiddenzone_input_value, ...}, ...},
        #:      'elements': {(plant_index, axis_label, metamer_index, organ_label, element): {element_input_name: element_input_value, ...}, ...},
        #:      'axes': {(plant_index, axis_label): {axis_input_name: axis_input_value, ...}, ...},
        #:      'sheath_internode_lengths': {(plant_index, axis_label, metamer_index): {'sheath': [list of sheath length belonging to the phytomer],
        #:                                                                             'cumulated_internode': [list of internode lengths cumulated from phytomer 1 to n]}, ...}
        #:
        #: See :TODO?
        #: for more information about the inputs.
        self.inputs = {}

        #: The outputs of elong-Wheat.
        #:
        #: `outputs` is a dictionary of dictionaries:
        #:     {'hiddenzone': {(plant_index, axis_label, metamer_index): {hiddenzone_input_name: hiddenzone_input_value, ...}, ...},
        #:      'elements': {(plant_index, axis_label, metamer_index, organ_label, element): {element_output_name: element_output_value, ...}, ...},
        #:      'axes': {(plant_index, axis_label): {axis_output_name: axis_output_value, ...}, ...}}
        #:
        #: See :TODO?
        #: for more information about the inputs.
        self.outputs = {}

        #: the delta t of the simulation (in seconds)
        self.delta_t = delta_t

        #: Update parameters if specified
        if update_parameters:
            parameters.__dict__.update(update_parameters)

    def initialize(self, inputs):
        """
        Initialize :attr:`inputs` from `inputs`.

        :param dict inputs: must be a dictionary with the same structure as :attr:`inputs`.
        """
        self.inputs.clear()
        self.inputs.update(inputs)

    def run(self, Tair, Tsoil, optimal_growth_option):
        """
        Run the simulation.

        :param float Tair: Air temperature at t (degree Celsius)
        :param float Tsoil: Soil temperature at t (degree Celsius)
        :param bool optimal_growth_option: if True the model will assume optimal growth conditions
        """

        # Copy the inputs into the output dict
        self.outputs.update({inputs_type: copy.deepcopy(all_inputs) for inputs_type, all_inputs in self.inputs.items() if inputs_type in {'hiddenzone', 'elements', 'axes',
                                                                                                                                          'sheath_internode_lengths'}})

        # Hidden zones
        all_hiddenzone_inputs = self.inputs['hiddenzone']
        all_hiddenzone_outputs = self.outputs['hiddenzone']

        # Elements
        all_element_inputs = self.inputs['elements']
        all_element_outputs = self.outputs['elements']

        # Axes
        all_axes_inputs = self.inputs['axes']
        all_axes_outputs = self.outputs['axes']

        # sheath and internode lengths
        all_sheath_internode_lengths = self.inputs['sheath_internode_lengths']

        # Ligule heights
        all_ligule_height_df = pd.DataFrame(columns=LIGULE_TOPOLOGY_COLUMNS)

        # --  Beginning of computations

        # -----------------------------
        # ---------- SAM ---------------
        # -----------------------------

        for axis_id, axis_inputs in sorted(all_axes_inputs.items()):
            curr_axis_outputs = all_axes_outputs[axis_id]
            nb_leaves = curr_axis_outputs['nb_leaves']

            # height of the SAM
            below_internode_lengths = all_sheath_internode_lengths[axis_id][nb_leaves]['cumulated_internode']
            SAM_height = model.calculate_cumulated_internode_length(below_internode_lengths)
            curr_axis_outputs['SAM_height'] = SAM_height

            # SAM temperature
            growth_temperature = model.calculate_growing_temperature(Tair, Tsoil, SAM_height)
            curr_axis_outputs['SAM_temperature'] = growth_temperature

            # temperature-compensated time
            curr_axis_outputs['delta_teq'] = model.calculate_time_equivalent_Tref(growth_temperature, self.delta_t)
            curr_axis_outputs['delta_teq_roots'] = model.calculate_time_equivalent_Tref(Tsoil, self.delta_t)
            curr_axis_outputs['res'] = model.modified_Arrhenius_equation(growth_temperature)

            # cumulated thermal time
            curr_axis_outputs['sum_TT'] = model.calculate_cumulated_thermal_time(curr_axis_outputs['sum_TT'], growth_temperature, curr_axis_outputs['delta_teq'])

            # update SAM status, leaf number and
            init_leaf, curr_axis_outputs['nb_leaves'], curr_axis_outputs['status'], curr_axis_outputs['teq_since_primordium'] = model.calculate_SAM_primodia(axis_inputs['status'],
                                                                                                                                                             curr_axis_outputs['teq_since_primordium'],
                                                                                                                                                             curr_axis_outputs['delta_teq'], nb_leaves,
                                                                                                                                                             curr_axis_outputs['cohort'])

            # GA production
            curr_axis_outputs['GA'] = model.calculate_SAM_GA(curr_axis_outputs['status'], curr_axis_outputs['teq_since_primordium'])

            # hiddenzone initiation
            for i in range(0, init_leaf):
                # Initialise hiddenzone
                hiddenzone_id = axis_id + tuple([1 + i + curr_axis_outputs['nb_leaves'] - init_leaf])  # TODO: peut etre simplifié tant que 'calculate_SAM_status' renvoie 1 erreur si init_leaf>1
                new_hiddenzone = parameters.HiddenZoneInit().__dict__
                self.outputs['hiddenzone'][hiddenzone_id] = new_hiddenzone

            # Ligule height
            all_ligule_height_df = model.calculate_ligule_height(all_sheath_internode_lengths[axis_id], all_element_inputs, axis_id, all_ligule_height_df)

            self.outputs['axes'][axis_id] = curr_axis_outputs

        # -----------------------------
        # ---------- Elements ---------
        # -----------------------------

        for element_id, element_inputs in sorted(all_element_inputs.items()):
            # Update element's age, only used by ADEL to adapt element's geometry (sor far lamina curvature, could be used to adapt stem geometry too)
            # TODO : the calculation of element's age must be extracted from elongwheat as it is run even for mature leaves
            curr_age = all_element_inputs[element_id]['age']
            axis_id = element_id[:2]
            curr_axis_outputs = all_axes_outputs[axis_id]
            self.outputs['elements'][element_id]['age'] = model.calculate_cumulated_thermal_time(curr_age, curr_axis_outputs['SAM_temperature'], curr_axis_outputs['delta_teq'])

        # -----------------------------
        # ---------- Hiddenzones ------
        # -----------------------------
        for hiddenzone_id, hiddenzone_inputs in sorted(all_hiddenzone_inputs.items()):

            axe_label = hiddenzone_id[1]
            axis_id = hiddenzone_id[:2]
            phytomer_id = hiddenzone_id[2]

            #: Tillers: in this version tillers functioning is replicated from corresponding elements of MS
            if axe_label != 'MS':
                tiller_to_MS_phytomer_id = tuple([axis_id[0], 'MS', all_axes_outputs[axis_id]['cohort'] + phytomer_id - 1])

                if tiller_to_MS_phytomer_id in all_hiddenzone_outputs.keys():
                    self.outputs['hiddenzone'][hiddenzone_id] = all_hiddenzone_outputs[tiller_to_MS_phytomer_id]

                    if all_hiddenzone_outputs[tiller_to_MS_phytomer_id]['leaf_is_emerged']:
                        # Lamina
                        tiller_to_MS_lamina_id = tiller_to_MS_phytomer_id + tuple(['blade', 'LeafElement1'])
                        tiller_lamina_id = hiddenzone_id + tuple(['blade', 'LeafElement1'])
                        if tiller_to_MS_lamina_id in all_element_outputs.keys():
                            self.outputs['elements'][tiller_lamina_id] = all_element_outputs[tiller_to_MS_lamina_id]
                        # else:
                        #     warnings.warn('No leaf found on main stem for tiller {}.'.format(tiller_to_MS_lamina_id))

                        # Emerged Sheath
                        tiller_to_MS_emerged_sheath_id = tiller_to_MS_phytomer_id + tuple(['sheath', 'StemElement'])
                        tiller_emerged_sheath_id = hiddenzone_id + tuple(['sheath', 'StemElement'])
                        if tiller_to_MS_emerged_sheath_id in all_element_outputs.keys():
                            self.outputs['elements'][tiller_emerged_sheath_id] = all_element_outputs[tiller_to_MS_emerged_sheath_id]
                        # else:
                        #     warnings.warn('No emerged sheath found on main stem for tiller {}.'.format(tiller_to_MS_emerged_sheath_id))

                        # Enclosed Sheath
                        tiller_to_MS_enclosed_sheath_id = tiller_to_MS_phytomer_id + tuple(['sheath', 'HiddenElement'])
                        tiller_enclosed_sheath_id = hiddenzone_id + tuple(['sheath', 'HiddenElement'])
                        if tiller_to_MS_enclosed_sheath_id in all_element_outputs.keys():
                            self.outputs['elements'][tiller_enclosed_sheath_id] = all_element_outputs[tiller_to_MS_enclosed_sheath_id]
                        # else:
                        #     warnings.warn('No enclosed sheath found on main stem for tiller {}.'.format(tiller_to_MS_enclosed_sheath_id))

                        # Emerged internode
                        tiller_to_MS_emerged_internode_id = tiller_to_MS_phytomer_id + tuple(['internode', 'StemElement'])
                        tiller_emerged_internode_id = hiddenzone_id + tuple(['internode', 'StemElement'])
                        if tiller_to_MS_emerged_internode_id in all_element_outputs.keys():
                            self.outputs['elements'][tiller_emerged_internode_id] = all_element_outputs[tiller_to_MS_emerged_internode_id]
                        # else:
                        #     warnings.warn('No emerged internode found on main stem for tiller {}.'.format(tiller_to_MS_emerged_internode_id))

                        # Enclosed internode
                        tiller_to_MS_enclosed_internode_id = tiller_to_MS_phytomer_id + tuple(['internode', 'HiddenElement'])
                        tiller_enclosed_internode_id = hiddenzone_id + tuple(['internode', 'HiddenElement'])
                        if tiller_to_MS_enclosed_internode_id in all_element_outputs.keys():
                            self.outputs['elements'][tiller_enclosed_internode_id] = all_element_outputs[tiller_to_MS_enclosed_internode_id]
                        # else:
                        #     warnings.warn('No enclosed internode found on main stem for tiller {}.'.format(tiller_to_MS_enclosed_internode_id))
                # else:
                #     warnings.warn('No main stem found for tiller {}.'.format(tiller_to_MS_phytomer_id))

            #: Main Stem
            else:
                curr_hiddenzone_outputs = all_hiddenzone_outputs[hiddenzone_id]
                curr_axis_outputs = all_axes_outputs[axis_id]
                curr_hiddenzone_outputs['init_leaf_L'] = hiddenzone_inputs['leaf_L']    # leaf length before turgor-growth sub-model
                curr_hiddenzone_outputs['hiddenzone_age'] += curr_axis_outputs['delta_teq']
                curr_hiddenzone_outputs['delta_teq'] = curr_axis_outputs['delta_teq']
                curr_hiddenzone_outputs['res'] = curr_axis_outputs['res']
                curr_hiddenzone_outputs['temperature'] = growth_temperature

                hidden_sheath_id = hiddenzone_id + tuple(['sheath', 'HiddenElement'])
                visible_sheath_id = hiddenzone_id + tuple(['sheath', 'StemElement'])
                hidden_internode_id = hiddenzone_id + tuple(['internode', 'HiddenElement'])
                visible_internode_id = hiddenzone_id + tuple(['internode', 'StemElement'])
                next_hiddenzone_id = tuple(list(hiddenzone_id[:2]) + [hiddenzone_id[2] + 1])

                # Found previous hiddenzone
                prev_hiddenzone_id = tuple(list(axis_id) + [phytomer_id - 1])
                if prev_hiddenzone_id in all_hiddenzone_inputs:
                    prev_leaf_emerged = all_hiddenzone_inputs[prev_hiddenzone_id]['leaf_is_emerged']
                    prev_hiddenzone_inputs = all_hiddenzone_inputs[prev_hiddenzone_id]
                    # Width and thickness of the previous leaf
                    curr_hiddenzone_outputs['width_prev'] = all_hiddenzone_inputs[prev_hiddenzone_id]['width']
                    curr_hiddenzone_outputs['thickness_prev'] = all_hiddenzone_inputs[prev_hiddenzone_id]['thickness']
                else:
                    prev_leaf_emerged = True

                prev_leaf2_hiddenzone_id = tuple(list(axis_id) + [phytomer_id - 2])
                if prev_leaf2_hiddenzone_id in all_hiddenzone_inputs:
                    prev_leaf2_emerged = all_hiddenzone_inputs[prev_leaf2_hiddenzone_id]['leaf_is_emerged']
                else:
                    prev_leaf2_emerged = True

                # Cumulated length of internodes up to the hidden zone
                below_internode_lengths = all_sheath_internode_lengths[axis_id][phytomer_id]['cumulated_internode']
                bottom_hiddenzone_height = model.calculate_cumulated_internode_length(below_internode_lengths)

                # Distance between the bottom of the hiddenzone and the highest previous ligule
                leaf_pseudostem_length = model.calculate_leaf_pseudostem_length(all_ligule_height_df[all_ligule_height_df['axis_id'] == axis_id], bottom_hiddenzone_height, phytomer_id)
                curr_hiddenzone_outputs['leaf_pseudostem_length'] = leaf_pseudostem_length
                curr_hiddenzone_outputs['delta_leaf_pseudostem_length'] = leaf_pseudostem_length - hiddenzone_inputs['leaf_pseudostem_length']  # Variable used in growthwheat

                # Calculate the internode pseudostem length
                curr_internode_L = hiddenzone_inputs['internode_L']
                internode_distance_to_emerge = model.calculate_internode_distance_to_emerge(all_ligule_height_df[all_ligule_height_df['axis_id'] == axis_id], bottom_hiddenzone_height, phytomer_id,
                                                                                            curr_internode_L)
                curr_hiddenzone_outputs['internode_distance_to_emerge'] = internode_distance_to_emerge
                curr_hiddenzone_outputs['delta_internode_distance_to_emerge'] = internode_distance_to_emerge - hiddenzone_inputs['internode_distance_to_emerge']  # Variable used in growthwheat

                # In case leaf is already mature but internode is growing, we update sheath visible and hidden lengths.
                if not curr_hiddenzone_outputs['leaf_is_growing'] and curr_hiddenzone_outputs['leaf_is_emerged']:
                    if hidden_sheath_id in self.inputs['elements'].keys():
                        sheath_hidden_length = self.inputs['elements'][hidden_sheath_id]['length']
                    else:
                        sheath_hidden_length = 0.
                        new_sheath = parameters.ElementInit().__dict__
                        self.outputs['elements'][hidden_sheath_id] = new_sheath
                    if visible_sheath_id not in self.outputs['elements'].keys():
                        new_sheath = parameters.ElementInit().__dict__
                        self.outputs['elements'][visible_sheath_id] = new_sheath
                    total_sheath_L = sheath_hidden_length + self.outputs['elements'][visible_sheath_id]['length']
                    updated_sheath_hidden_length = min(total_sheath_L, leaf_pseudostem_length)
                    updated_sheath_visible_length = max(0, total_sheath_L - updated_sheath_hidden_length)
                    self.outputs['elements'][hidden_sheath_id]['length'] = updated_sheath_hidden_length
                    self.outputs['elements'][visible_sheath_id]['length'] = updated_sheath_visible_length

                #: Leaf elongation
                if curr_hiddenzone_outputs['leaf_is_growing']:

                    if leaf_pseudostem_length < 0:
                        warnings.warn('Pseudostem length of {} decreased while leaf growing.'.format(hiddenzone_id))

                    if prev_leaf2_emerged and not curr_hiddenzone_outputs['leaf_is_emerged']:
                        time_prev_leaf2_emergence = prev_hiddenzone_inputs['leaf_pseudo_age']
                        # Age de Ln-2 dans element ? ou teIN prolongé ? ou empêcher HZ  de disparaître dans cette simul ?

                        curr_hiddenzone_outputs['mean_conc_sucrose'] = model.calculate_mean_conc_sucrose(hiddenzone_inputs['mean_conc_sucrose'],
                                                                                                         time_prev_leaf2_emergence,
                                                                                                         curr_axis_outputs['delta_teq'],
                                                                                                         hiddenzone_inputs['sucrose'],
                                                                                                         hiddenzone_inputs['mstruct'])

                    if not prev_leaf_emerged:  #: Before the emergence of the previous leaf. Exponential-like elongation.
                        # delta leaf length
                        delta_leaf_L = model.calculate_deltaL_preE(axis_inputs['xylem_water_potential'], hiddenzone_inputs['sucrose'], hiddenzone_inputs['leaf_L'], hiddenzone_inputs['amino_acids'], hiddenzone_inputs['mstruct'],
                                                                   curr_axis_outputs['delta_teq'], phytomer_id, optimal_growth_option)
                        leaf_L = hiddenzone_inputs['leaf_L'] + delta_leaf_L

                        # Maximum leaf length before previous leaf emergence (En-1)
                        # leaf_L = min(leaf_L, 0.006949)  # Gauthier et al. (2021)

                        curr_hiddenzone_outputs['ratio_DZ'] = 1
                        curr_hiddenzone_outputs['length'] = min(leaf_L, hiddenzone_inputs['leaf_pseudostem_length'])

                        # Width and thickness of the previous leaf
                        curr_hiddenzone_outputs['width_prev'] = all_hiddenzone_inputs[prev_hiddenzone_id]['width']
                        curr_hiddenzone_outputs['thickness_prev'] = all_hiddenzone_inputs[prev_hiddenzone_id]['thickness']

                    else:   #: After the emergence of the previous leaf. Elongation in turgor-growth sub-model, according to Lockhart's equations

                        # -----------------------------------------------------------------------------------------------------------
                        leaf_pseudo_age = model.calculate_leaf_pseudo_age(hiddenzone_inputs['leaf_pseudo_age'], curr_axis_outputs['delta_teq'])
                        curr_hiddenzone_outputs['leaf_pseudo_age'] = leaf_pseudo_age
                        curr_hiddenzone_outputs['delta_leaf_pseudo_age'] = leaf_pseudo_age - hiddenzone_inputs['leaf_pseudo_age']

                        delta_leaf_L = 0    #: in turgor-growth sub-model
                        leaf_L = hiddenzone_inputs['leaf_L'] + delta_leaf_L

                        # Ratio (mass) of Division Zone in the hiddenzone
                        curr_hiddenzone_outputs['ratio_DZ'] = model.calculate_ratio_DZ_postE(leaf_L, curr_hiddenzone_outputs['leaf_pseudo_age'], leaf_pseudostem_length)

                        lamina_id = hiddenzone_id + tuple(['blade', 'LeafElement1'])

                        # CASE OF GROWING LEAF WITH INDISSOCIED EMERGED TISSUES - 06.09.2024
                        #: Lamina has not emerged
                        if not curr_hiddenzone_outputs['leaf_is_emerged']:
                            #: Test of leaf emergence against distance to leaf emergence. Assumes that a leaf cannot emerge before the previous one
                            curr_hiddenzone_outputs['leaf_is_emerged'] = model.calculate_leaf_emergence(hiddenzone_inputs['leaf_L'], leaf_pseudostem_length)
                            if curr_hiddenzone_outputs['leaf_is_emerged']:  # Initialise lamina outputs
                                new_lamina = parameters.ElementInit().__dict__
                                self.outputs['elements'][lamina_id] = new_lamina

                                # Length of the HZ at the emergence - UPDATE VICTORIA 07.01.25
                                curr_hiddenzone_outputs['length_hz_En'] = curr_hiddenzone_outputs['leaf_L']
                                # Width of the HZ at the emergence - UPDATE VICTORIA 07.01.25

                                curr_lamina_outputs = all_element_outputs[lamina_id]
                                # Length of emerged lamina
                                lamina_L = model.calculate_lamina_L(leaf_L, leaf_pseudostem_length, hiddenzone_id)
                                curr_lamina_outputs['length'] = lamina_L

                                # Initialise variables for the next hidden zone as its previous leaf has now emerged
                                if next_hiddenzone_id in all_hiddenzone_inputs:
                                    next_hiddenzone_outputs = all_hiddenzone_outputs[next_hiddenzone_id]
                                    next_hiddenzone_outputs['lamina_Lmax'] = model.calculate_lamina_Lmax(hiddenzone_id[2])  #: Maximal lamina length
                                    # curr_hiddenzone_outputs['leaf_Wmax'] = model.calculate_leaf_Wmax(hiddenzone_id[2])  #: Maximal leaf width
                                    next_hiddenzone_outputs['leaf_pseudo_age'] = 0  #: Pseudo age of the leaf since beginning of automate growth (s)
                                    # Width and thickness of the previous leaf
                                    next_hiddenzone_outputs['width_prev'] = all_hiddenzone_inputs[prev_hiddenzone_id]['width']
                                    next_hiddenzone_outputs['thickness_prev'] = all_hiddenzone_inputs[prev_hiddenzone_id]['thickness']
                                    self.outputs['hiddenzone'][next_hiddenzone_id] = next_hiddenzone_outputs
                                else:
                                    warnings.warn('No next hidden zone found for hiddenzone {}.'.format(hiddenzone_id))

                                # Define lamina_Wmax and structural weight of the current sheath and lamina
                                # curr_lamina_outputs['Wmax'] = model.calculate_leaf_Wmax(hiddenzone_id[2])
                                # curr_lamina_outputs['Wmax'] = curr_hiddenzone_outputs['leaf_Wmax']
                                curr_hiddenzone_outputs['SSLW'] = model.calculate_SSLW(hiddenzone_id[2], curr_hiddenzone_outputs['mean_conc_sucrose'], optimal_growth_option)
                                curr_hiddenzone_outputs['LSSW'] = model.calculate_LSSW(hiddenzone_id[2], curr_hiddenzone_outputs['mean_conc_sucrose'], optimal_growth_option)

                                # Update of lamina outputs
                                self.outputs['elements'][lamina_id] = curr_lamina_outputs

                        # : Lamina has emerged and is growing
                        elif curr_hiddenzone_outputs['leaf_is_emerged'] and all_element_inputs[lamina_id]['is_growing']:
                            curr_lamina_outputs = all_element_outputs[lamina_id]

                            #: Test end of elongation
                            if curr_hiddenzone_outputs['leaf_pseudo_age'] >= parameters.te:
                                curr_lamina_outputs['is_growing'] = False
                                curr_hiddenzone_outputs['leaf_is_growing'] = False
                                curr_hiddenzone_outputs['leaf_is_remobilizing'] = True

                                curr_hiddenzone_outputs['sheath_is_emerged'] = True

                                # Initialise hidden sheath outputs
                                new_sheath = parameters.ElementInit().__dict__
                                self.outputs['elements'][hidden_sheath_id] = new_sheath
                                curr_hidden_sheath_outputs = all_element_outputs[hidden_sheath_id]
                                self.outputs['elements'][hidden_sheath_id]['length'] = leaf_pseudostem_length  # Length of hidden sheath
                                curr_hidden_sheath_outputs['width'] = all_hiddenzone_inputs[hiddenzone_id]['width']
                                curr_hidden_sheath_outputs['thickness'] = all_hiddenzone_inputs[hiddenzone_id]['thickness']
                                curr_hidden_sheath_outputs['is_growing'] = False
                                self.outputs['elements'][hidden_sheath_id] = curr_hidden_sheath_outputs  # Update of sheath outputs

                                # Sheath : Lamina Ratio
                                SL_ratio = model.calculate_SL_ratio(hiddenzone_id[2])
                                # Initialise visible sheath outputs
                                new_sheath = parameters.ElementInit().__dict__
                                self.outputs['elements'][visible_sheath_id] = new_sheath
                                curr_visible_sheath_outputs = all_element_outputs[visible_sheath_id]
                                # Lamina and sheath lengths
                                curr_lamina_outputs['length'] = curr_hiddenzone_outputs['leaf_L'] / (1 + SL_ratio)
                                # curr_visible_sheath_outputs['length'] = curr_hiddenzone_outputs['leaf_L'] - curr_lamina_outputs['length'] - self.outputs['elements'][hidden_sheath_id]['length']
                                curr_visible_sheath_outputs['length'] = max(1E-03, curr_hiddenzone_outputs['leaf_L'] - curr_lamina_outputs['length'] - self.outputs['elements'][hidden_sheath_id]['length'])

                                curr_visible_sheath_outputs['is_growing'] = False
                                curr_visible_sheath_outputs['width'] = all_hiddenzone_inputs[hiddenzone_id]['width']
                                curr_visible_sheath_outputs['thickness'] = all_hiddenzone_inputs[hiddenzone_id]['thickness']
                                self.outputs['elements'][visible_sheath_id] = curr_visible_sheath_outputs  # Update of sheath outputs

                                # Lamina and sheath Lmax
                                curr_hiddenzone_outputs['lamina_Lmax'] = curr_lamina_outputs['length']
                                curr_hiddenzone_outputs['sheath_Lmax'] = curr_visible_sheath_outputs['length'] + curr_hidden_sheath_outputs['length']
                                curr_hiddenzone_outputs['SL_ratio'] = SL_ratio

                                # Initialise variables for the next internode
                                if next_hiddenzone_id in all_hiddenzone_inputs:
                                    next_hiddenzone_outputs = all_hiddenzone_outputs[next_hiddenzone_id]
                                    if curr_axis_outputs['GA']:
                                        next_hiddenzone_outputs['internode_Lmax'] = model.calculate_internode_Lmax(next_hiddenzone_outputs['internode_L'])  #: Estimate of final internode length
                                        next_hiddenzone_outputs['internode_Lmax_lig'] = next_hiddenzone_outputs['internode_Lmax']  #: Estimate of final internode length at previous leaf ligulation
                                    next_hiddenzone_outputs['LSIW'] = model.calculate_LSIW(next_hiddenzone_outputs['LSSW'], next_hiddenzone_id[2],optimal_growth_option=True)  #: Lineic Structural Internode Weight
                                    next_hiddenzone_outputs['internode_pseudo_age'] = 0  #: Pseudo age of the internode since beginning of automate growth (s)
                                    self.outputs['hiddenzone'][next_hiddenzone_id] = next_hiddenzone_outputs
                                else:
                                    warnings.warn('No next hidden zone found for hiddenzone {}.'.format(hiddenzone_id))

                            # Update of lamina outputs
                            self.outputs['elements'][lamina_id] = curr_lamina_outputs


                # : Leaf is mature but internode may be growing
                else:
                    leaf_L = hiddenzone_inputs['leaf_L']
                    delta_leaf_L = 0

                # Update of leaf outputs
                curr_hiddenzone_outputs['leaf_L'] = leaf_L
                curr_hiddenzone_outputs['delta_leaf_L'] = delta_leaf_L

                # ------------------------------------------------------------------------------------------------------
                #: Internode elongation
                #: Initialisation of internode elongation
                if (not curr_hiddenzone_outputs['internode_is_growing']) and (curr_hiddenzone_outputs['internode_L'] == 0):
                    #: As for leaf primordia, we neglect CN growth due to IN length initialisation
                    curr_hiddenzone_outputs['internode_is_growing'], curr_hiddenzone_outputs['internode_L'] = model.calculate_init_internode_elongation(curr_hiddenzone_outputs['hiddenzone_age'])
                    if curr_hiddenzone_outputs['internode_is_growing']:
                        new_internode = parameters.ElementInit().__dict__
                        self.outputs['elements'][hidden_internode_id] = new_internode
                        self.outputs['elements'][hidden_internode_id]['length'] = curr_hiddenzone_outputs['internode_L']

                if curr_hiddenzone_outputs['internode_is_growing']:

                    #: Found previous lamina to know if the previous leaf is ligulated.
                    prev_lamina_id = tuple(list(hiddenzone_id[:2]) + [hiddenzone_id[2] - 1]) + tuple(['blade', 'LeafElement1'])
                    if prev_lamina_id in all_element_inputs:
                        prev_leaf_ligulated = not all_element_inputs[prev_lamina_id]['is_growing']
                    else:
                        prev_leaf_ligulated = False

                    #: Before ligulation of the leaf on the previous phytomer. Exponential-like elong. cf. Kirby 1988, Malvoisin 1984 II
                    if not prev_leaf_ligulated:
                        delta_internode_L = model.calculate_delta_internode_L_preL(phytomer_id, curr_hiddenzone_outputs['sucrose'], curr_hiddenzone_outputs['internode_L'],
                                                                                   curr_hiddenzone_outputs['amino_acids'], curr_hiddenzone_outputs['mstruct'],
                                                                                   curr_axis_outputs['delta_teq'], optimal_growth_option=True)
                        internode_L = curr_hiddenzone_outputs['internode_L'] + delta_internode_L  # TODO: Ckeck internode_L is not too large (in the case of long delta_t)

                        curr_hiddenzone_outputs['internode_L'] = internode_L
                        curr_hiddenzone_outputs['delta_internode_L'] = delta_internode_L
                        # Hidden internode
                        if hidden_internode_id not in self.outputs['elements'].keys():
                            new_internode = parameters.ElementInit().__dict__
                            self.outputs['elements'][hidden_internode_id] = new_internode
                        self.outputs['elements'][hidden_internode_id]['length'] = internode_L

                    #: After ligulation of the leaf on the previous phytomer.
                    else:
                        prev_internode_pseudo_age = curr_hiddenzone_outputs['internode_pseudo_age']
                        internode_pseudo_age = model.calculate_internode_pseudo_age(prev_internode_pseudo_age, curr_axis_outputs['delta_teq'])

                        curr_hiddenzone_outputs['internode_pseudo_age'] = internode_pseudo_age
                        curr_hiddenzone_outputs['delta_internode_pseudo_age'] = internode_pseudo_age - curr_hiddenzone_outputs['internode_pseudo_age']

                        #: Elongation only if Gibberelin production by SAM
                        if curr_axis_outputs['GA']:
                            #: Case of internodes that will not fully elongate, GA synthesis started after their previous leaf ligulation (i.e. no Lmax defined)
                            if (np.isnan(curr_hiddenzone_outputs['internode_Lmax'])) or (np.isnan(curr_hiddenzone_outputs['internode_Lmax_lig'])):
                                curr_hiddenzone_outputs['internode_Lmax'] = curr_hiddenzone_outputs['internode_Lmax_lig'] = model.calculate_short_internode_Lmax(curr_hiddenzone_outputs['internode_L'],
                                                                                                                                                                 curr_hiddenzone_outputs[
                                                                                                                                                                     'internode_pseudo_age'])

                            delta_internode_L = model.calculate_delta_internode_L_postL(prev_internode_pseudo_age, curr_hiddenzone_outputs['internode_pseudo_age'],
                                                                                        hiddenzone_inputs['internode_L'], curr_hiddenzone_outputs['internode_Lmax_lig'],
                                                                                        hiddenzone_inputs['sucrose'], hiddenzone_inputs['amino_acids'],
                                                                                        hiddenzone_inputs['mstruct'], optimal_growth_option=True)
                            internode_L = hiddenzone_inputs['internode_L'] + delta_internode_L

                            # Update internode_Lmax
                            if hiddenzone_inputs['internode_Lmax']:
                                curr_hiddenzone_outputs['internode_Lmax'] = model.calculate_update_internode_Lmax(curr_hiddenzone_outputs['internode_Lmax_lig'], internode_L, internode_pseudo_age)

                            #: Internode is not visible
                            if not curr_hiddenzone_outputs['internode_is_visible']:
                                #: Test of internode emergence.
                                curr_hiddenzone_outputs['internode_is_visible'] = model.calculate_internode_visibility(curr_hiddenzone_outputs['internode_L'], internode_distance_to_emerge)
                                if curr_hiddenzone_outputs['internode_is_visible']:  #: Initialise internode outputs
                                    new_internode_outputs = parameters.ElementInit().__dict__
                                    self.outputs['elements'][visible_internode_id] = new_internode_outputs
                                    self.outputs['elements'][visible_internode_id]['length'] = min(curr_hiddenzone_outputs['internode_Lmax'],
                                                                                                   model.calculate_emerged_internode_L(internode_L, internode_distance_to_emerge))
                                    self.outputs['elements'][hidden_internode_id]['length'] = internode_distance_to_emerge
                                else:
                                    self.outputs['elements'][hidden_internode_id]['length'] = internode_L

                            #: Internode is visible
                            else:
                                self.outputs['elements'][visible_internode_id]['length'] = min(curr_hiddenzone_outputs['internode_Lmax'],
                                                                                               model.calculate_emerged_internode_L(internode_L, internode_distance_to_emerge))
                                self.outputs['elements'][hidden_internode_id]['length'] = internode_distance_to_emerge

                            #: Test end of elongation
                            if model.calculate_end_internode_elongation(internode_L, curr_hiddenzone_outputs['internode_Lmax'], curr_hiddenzone_outputs['internode_pseudo_age']):
                                curr_hiddenzone_outputs['internode_is_growing'] = False
                                curr_hiddenzone_outputs['internode_is_remobilizing'] = True
                                # Visible internode
                                if curr_hiddenzone_outputs['internode_is_visible']:
                                    self.outputs['elements'][visible_internode_id]['is_growing'] = False
                                # Hidden internode
                                self.outputs['elements'][hidden_internode_id]['is_growing'] = False

                        #: Internode not elongating because no GA
                        else:
                            internode_L = curr_hiddenzone_outputs['internode_L']
                            delta_internode_L = 0
                            # Test end of elongation
                            if model.calculate_end_internode_elongation(internode_L, curr_hiddenzone_outputs['internode_Lmax'], curr_hiddenzone_outputs['internode_pseudo_age']):
                                curr_hiddenzone_outputs['internode_is_growing'] = False
                                curr_hiddenzone_outputs['internode_is_remobilizing'] = True
                                curr_hiddenzone_outputs['internode_Lmax'] = curr_hiddenzone_outputs['internode_L']
                                # Visible internode
                                if curr_hiddenzone_outputs['internode_is_visible']:
                                    self.outputs['elements'][visible_internode_id]['is_growing'] = False
                                # Hidden internode
                                if hidden_internode_id not in self.outputs['elements'].keys():
                                    new_internode = parameters.ElementInit().__dict__
                                    self.outputs['elements'][hidden_internode_id] = new_internode
                                self.outputs['elements'][hidden_internode_id]['is_growing'] = False
                                self.outputs['elements'][hidden_internode_id]['length'] = min(internode_L, internode_distance_to_emerge)

                #: Internode not elongating (not yet or already mature)
                else:
                    internode_L = curr_hiddenzone_outputs['internode_L']
                    delta_internode_L = 0

                # Update internodes outputs
                curr_hiddenzone_outputs['internode_L'] = internode_L
                curr_hiddenzone_outputs['delta_internode_L'] = delta_internode_L

                # Only growing hiddenzones are sent
                # after end of elongation (leaf and/or internode), it should :
                #       - pass by growth wheat for remobilisation
                #       - pass another time by elong wheat for update of curr_element_outputs['final_hidden_length']
                # the hiddenzone will then be deleted since both growing flags are False and both delta_L are zeros.
                if hiddenzone_inputs['internode_is_growing'] or curr_hiddenzone_outputs['leaf_is_growing'] or \
                        curr_hiddenzone_outputs.get('leaf_is_remobilizing', False) or curr_hiddenzone_outputs.get('internode_is_remobilizing', False):
                    self.outputs['hiddenzone'][hiddenzone_id] = curr_hiddenzone_outputs
                else:  # End of internode elongation
                    del self.outputs['hiddenzone'][hiddenzone_id]
                    print('HiddenZone growth {} is over'.format(hiddenzone_id))
