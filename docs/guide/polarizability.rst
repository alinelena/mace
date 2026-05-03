.. _polarizability:

=====================================================================
Dipole Moments and Polarizabilities with MACE
=====================================================================

Using MACE-MDP: Pretrained Model for Dipole Moments and Polarizabilities
=========================================================================

If you need ready-to-use predictions of molecular dipole moments and fully
anisotropic polarizability tensors for organic systems, the pretrained
`MACE-MDP <https://github.com/Nilsgoe/MACE-MDP>`_ model is the recommended
starting point. It is trained on the SPICE-α dataset (>1.8 million
electric-field response calculations across molecular and condensed-phase
environments) and achieves first-principles accuracy at a fraction of the cost.
See the :ref:`foundation_models` page for the full model listing.

.. warning::

   Fine-tuning of MACE-MDP is not available at this time.

The model is available via the ``mace_mdp`` convenience function, which
automatically downloads and caches the model on first use:

.. code-block:: python

   from mace.calculators import mace_mdp
   from ase import build

   calc = mace_mdp(device="cuda", default_dtype="float64")
   atoms = build.molecule("H2O")
   atoms.calc = calc

Getting Dipole Moments
----------------------

Use ASE's standard ``get_dipole_moment()`` after attaching the calculator:

.. code-block:: python

   mu = atoms.get_dipole_moment()  # returns array of shape (3,) in e·Å
   print("Dipole moment (e·Å):", mu)

Getting Polarizability
----------------------

Use ``get_property`` with the ``"polarizability"`` key to obtain the full 3×3
tensor:

.. code-block:: python

   import numpy as np

   alpha = np.asarray(calc.get_property("polarizability", atoms)).reshape(3, 3)
   print("Polarizability tensor (Å³):\n", alpha)

**Tip:**
For the **spherical (irreducible) polarizability** components, useful for
Raman spectra, use the ``"polarizability_sh"`` property:

.. code-block:: python

   spherical_alpha = calc.get_property("polarizability_sh", atoms)

Tutorial Jupyter notebooks for IR spectra, Raman spectra, and
dipole/polarizability extraction are provided in the
`examples/ <https://github.com/Nilsgoe/MACE-MDP/tree/main/examples>`_
directory of the MACE-MDP repository.

If you use MACE-MDP in your work, please cite [1]_.

.. [1] Nils Gönnheimer, Karsten Reuter, Venkat Kapil, and Johannes T. Margraf,
   *"MACE-MDP: A General Dipole and Polarizability Model for Organic Molecules
   and Materials"*, ChemRxiv (2025).
   `DOI:10.26434/chemrxiv.15000716 <https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000716>`_

----

Training Your Own Model
=======================

Training Example
----------------

A typical training command for an AtomicDielectric MACE model looks like:

.. code-block:: bash

  python /../mace/mace/cli/run_train.py \
       --name="mace_mu_alpha" \
       --train_file="train.xyz" \
       --valid_file="val.xyz" \
       --test_dir="test.xyz" \
       --model="AtomicDielectricMACE" \
       --E0s="average" \
       --num_interactions=2 \
       --num_channels=128 \
       --max_L=2 \
       --correlation=3 \
       --MLP_irreps="16x0e+16x1o+16x2e" \
       --dipole_key="REF_dipole" \
       --polarizability_key="REF_polarizability" \
       --loss="dipole_polar" \
       --weight_decay=5e-10 \
       --polarizability_weight=2000 \
       --dipole_weight=1000 \
       --clip_grad=1.0 \
       --batch_size=128 \
       --valid_batch_size=128 \
       --max_num_epochs=40 \
       --scheduler_patience=15 \
       --patience=15 \
       --eval_interval=1 \
       --ema \
       --error_table="DipolePolarRMSE" \
       --default_dtype="float64" \
       --device=cuda \
       --seed=123 \
       --restart_latest \
       --save_cpu

Setting --MLP_irreps="16x0e+16x1o+16x2e" and --max_L=2 are crutial for predicting polarizability correctly.
Compared to a MACE - MLIP these models usually need less epochs to converge.

Extracting Polarizability Using ASE
------------------------------------

Once training is complete and you have your model file (e.g., `mace_mu_alpha.model`), you can extract polarizability tensors from a trajectory.

Example extraction script:

.. code-block:: python
    
   import numpy as np
   from ase.io import Trajectory
   from mace.calculators.mace import MACECalculator

   # Setup calculator
   polar_calc = MACECalculator(
       model_paths="mace_mu_alpha.model",
       model_type="DipolePolarizabilityMACE",
       device="cuda",
       default_dtype="float64"
   )

   traj = Trajectory("test.traj", "r")
   n_frames=len(traj)
   alpha = np.empty((n_frames, 3, 3), dtype=float)

   for i, atoms in enumerate(traj):
       atoms.calc = polar_calc
       alpha[i] = np.asarray(atoms.calc.get_property("polarizability", atoms)).reshape(3, 3)
       print(f"Frame {i} Polarizability: ", alpha[i])
   traj.close()


**Tip:**  
To get the **spherical polarizability** (e.g. for Raman spectra), use the property `"polarizability_sh"` with `get_property`:

.. code-block:: python

   spherical_alpha = atoms.calc.get_property("polarizability_sh", atoms)