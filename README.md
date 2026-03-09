# Interesting Physics

This repository contains interesting physical phenomena that I have simulated, focusing mainly on General Relativity (GR) and the nature of light from a quantum-mechanical perspective (optics).

## Structure

The repository is separated into core computational modules, runnable examples, and output directories.

### Core Modules (`module/`)
This directory contains the Python scripts and mathematical models used to run the simulations. 

* **`GR/` (General Relativity):** Handles the mathematics of curved spacetime. Includes scripts for modeling light ray orbits (`lightray_orbit.py`) and analyzing effective potentials (`schwarzschild_effective_potential.py`).
* **`optics/`:** Contains scripts for wave optics and diffraction phenomena. Includes specific models for Arago spots (`arago_spot.py`), general diffraction (`general_diffraction.py`), knife-edge diffraction (`knife_edge.py`), and slit diffraction (`slit.py`). It also contains the core `JWST_simulation.py` engine.

### Examples and Simulations (`examples/`)
This folder contains all the interactive Jupyter Notebooks and standalone Python scripts that put the core modules to use. **Note: All Python scripts within this directory must be executed from inside the `examples/` folder.**

* **Notebooks:** `light_orbit.ipynb` and `optics.ipynb` for photon orbits and quantum nature of light.
* **Astrophysics & JWST:** Scripts for simulating the Cosmic Microwave Background (`cmb_gaussian.py`), the JWST mirror (`jwst_mirror.py`), and various starfields (`starfield_bright.py`, `starfield_colorful.py`, `starfield_gravlense.py`).
* **Optics:** Scripts for generating static visualizations (`optics_image.py`) and animations (`optics_animation.py`).

### Outputs
* **`images/`:** Stores generated visual outputs, categorized into `GR`, `JWST`, and `optics` subfolders.
* **`videos/`:** Stores generated `.mp4` and `.gif` animation files, categorized into `GR` and `optics` subfolders.


## Reference: JWST Dimensions

The simulations are modeled after the actual physical parameters of the James Webb Space Telescope:

* Primary Mirror Diameter: ~6.5 m (21.3 ft)
* Clear Aperture: 25 m² 
* Mirror Segments: 18 hexagonal segments
* Focal Length: 131.4 m
* Sunshield Dimensions: 21.197 m x 14.162 m (69.5 ft x 46.5 ft)

*(Data sourced from the [NASA JWST Fact Sheet](https://science.nasa.gov/mission/webb/fact-sheet/))*

## Getting Started

To run the simulations, navigate into the `examples` directory first. For example, from the project root in your terminal:

```bash
cd examples
python starfield_bright.py
```

Or, launch Jupyter Notebook from the project root and open `examples/light_orbit.ipynb` or `examples/optics.ipynb`.

## Dependencies

To run the code in this repository, you will need the following Python packages installed:

* `numpy`
* `matplotlib`
* `scipy`
* `astropy` (only for `Schwarzschild_effective_potential.py`)
* `camb` (for Cosmic Microwave Background simulations)
* `cupy` (Optional, but highly recommended if you have a compatible GPU for hardware acceleration)

You can generally install the CPU-bound dependencies via pip:

```bash
pip install numpy matplotlib scipy astropy camb
```

*(Refer to the [CuPy documentation](https://docs.cupy.dev/en/stable/install.html) for specific installation instructions corresponding to your CUDA version if you wish to use GPU acceleration).*