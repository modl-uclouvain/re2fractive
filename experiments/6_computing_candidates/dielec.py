import json
import monty.json
import modif_spec as mod_spec
import numpy as np
import pandas as pd
import traceback as tb
import sys

from abipy.abilab import Structure

from atomate2.vasp.jobs.core import DielectricMaker
from atomate2.vasp.powerups import (
    update_user_incar_settings,
    update_user_potcar_functional,
    update_user_kpoints_settings
)

from fireworks import LaunchPad
from importlib import reload

from jobflow import run_locally
from jobflow.managers.fireworks import flow_to_workflow

from pathlib import Path


reload(mod_spec)

#==============================================================================

#   Paths to the candidates datasets
candidate_structure_path    = Path(__file__).parent.parent / "5_predicting_latest_structures" / "candidate_structures.json"
candidate_mpid_path         = Path(__file__).parent.parent / "5_predicting_latest_structures" / "candidates.csv"

#   Loading the candidates datasets
if not candidate_mpid_path.exists():
    raise Exception(f"The file containing the MP ids of the candidates was not found at {candidate_mpid_path}.")
else:
    df_candidates = pd.read_csv(candidate_mpid_path, index_col=0)
    mpids = df_candidates.index.to_list()
    formulae = df_candidates['formula'].values

if not candidate_structure_path.exists():
    raise Exception(f"The file containing the structures of the candidates was not found at {candidate_structure_path}.")
else:
    with open(candidate_structure_path) as f:
        structures = json.load(f, cls=monty.json.MontyDecoder)

#   Iterating over the candidates to add them in the Launchpad 
for mpid, struc in zip(mpids,structures):
    print(mpid)
    print(struc.composition.reduced_formula)

    # Initializes the MP structure
    structure  = struc.get_primitive_structure()
    
    # Create a Dfpt dielectric WF to compute the dielectric tensor
    dielec_flow = DielectricMaker().make(structure=structure)
    
    # Power ups
        # Increase the kpoints density to 3000 per reciprocal atom for balance accuracy/complexity
    dielec_flow = update_user_kpoints_settings( dielec_flow, {"grid_density": 3000})
    
        # Specify:  the parallelization to optimize performance
        #           the electronic convergence criterion            (1E-5 eV otherwise by default),
        #           do not enforce a type of psp --> read in POTCAR (PBEsol otherwise by default)
    dielec_flow = update_user_incar_settings(   dielec_flow, {"KPAR": 8, "EDIFF": 1E-6, "GGA": None})
    
        # Choose the type of PSP, here PBE_54
    dielec_flow = update_user_potcar_functional(dielec_flow, "PBE_54")
    
    # Let's add a metadata to recover it easily from the MongoDb afterwards with {"spec._tasks.job.metadata.Label": "HSE_Etot(x)"}
    dielec_flow.update_metadata({"Batch": "re2fractive_v1", "mp-id": f"{mpid}"})
    
    # convert the flow to a fireworks WorkFlow object
    wf = flow_to_workflow(dielec_flow)
    
    # add the _preserve_fworker keywork to the spec of each FW so that parents and children run on the same FWorker (machine here)
    wf = mod_spec.preserve_fworker(wf)
    
    # submit the workflow to the FireWorks launchpad (requires a valid connection)
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)

#   Final step: launch the calculations on a cluster using qlaunch
