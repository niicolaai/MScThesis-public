# exoRL

Reinforcement-learning-based control for a lower-limb exoskeleton (WE2), trained in Isaac Lab.
Code accompanying the MSc thesis *"Reinforcement Learning-Based Control Design for Lower-Limb Exoskeletons Traversing Varying Terrains"* (University of Twente).

**Note:** The exoskeleton robot model (URDF/USD, meshes), hardware control files and the raw actuator
parameter file are proprietary and are not included in this repository. The environment therefore
cannot be run out of the box: `source/exoRL/exoRL/tasks/direct/exorl/exoWE2_V2.py` expects a robot USD file.

Human reference motion (subject AB08, level-ground walking) is taken from the open gait dataset by Camargo et al. (2021),
*"A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions"*,
Journal of Biomechanics, 119, 110320. https://doi.org/10.1016/j.jbiomech.2021.110320
The raw data is not redistributed here; to run the scripts in `humanMotion/`, download the dataset and place the
subject's `ik/` files under `humanMotion/AB08/ik/`.

## License

All rights reserved. This code is published for viewing only; see [LICENSE](LICENSE).
