# Code origins and scientific scope

Source: read-only ICASSP2027_JCT_RSA_Code_Data_Release, submission_code/scripts.
The ten scripts below were read in full before adaptation. No related_experiments code or results are substituted for this submission baseline.

| Source filename | Implementation to preserve |
| --- | --- |
| ridge_clip_specificity_v2.py | Four targets, per-window ridge and single-window held-out comparison |
| heldout_temporal_fixed_model.py | Concept-disjoint frozen temporal mapping |
| partial_rsa_variance_partition.py | Pixel-PCA-controlled partial Spearman |
| intermediate_layer_partial_rsa.py | Intermediate layer controls |
| layerwise_hierarchy_upgrade.py | Layerwise fixed mapping and per-window partial RSA |
| ridge_fixed_model_sweep.py | Fixed versus per-window mapping |
| ridge_group_temporal_permutation_fullcohort.py | Per-window ridge, sign flips and legacy FDR code |
| heldout_uncertainty.py | Held-out extraction and bootstrap; despite introductory comments, actually reloads EEG |
| fixed_model_subject_robustness.py | Participant-level frozen-mapping robustness |
| fixed_model_window_robustness.py | Training-window sensitivity |

All ten loaders require BrainVision headers and their EEG/marker dependencies, RSVP events TSV, and concept image directories. Scientific dependencies include NumPy, pandas, MNE, PyTorch, torchvision, OpenAI CLIP, SciPy, scikit-learn, Pillow and Matplotlib. No archived script imports a local project module; previous sys.path injections are unnecessary.

The scripts contain genuine ridge, embedding extraction/projection, RDM, RSA, temporal and control implementations. They are historical method code, not a numerical certification. Original event labels use prefix truncation after epoch selection; packaging does not establish event alignment. The historical BH routine is not certified as correct. Runtime interpretation strings are legacy statements, not new scientific conclusions. These limitations must remain disclosed in the public README.

Top-level computation is enclosed in explicit run functions. Path assignments use the INI configuration; unnecessary old sys.path injections were removed. Missing-image and failed-participant branches now fail rather than silently fill or reduce the cohort. These are I/O/error-handling adaptations, not numerical repairs. The scientific Random target is an explicit null-control, never a fallback dataset.

Plotting source exists for numerical panels, but exact reproduction of final manuscript assets is not established. No final figures or experimental outputs are included.
