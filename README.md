# Leaf It to Renewal: Improved Predictive Maintenance Policies via Renewal Theory and Decision Trees

**Short Description:** 

This project implements various predictive maintenance policies for managing preventive replacements and preventive ordering of single-component systems. The input to the policies are component-specific remaining useful life (RUL) distributions. Here, these distributions are obtained by a provided RUL simulator.

---

## Installation

1. Prerequisites

    The most straightforward way to get the project up and running on your machine is to use the 'uv' manager. The code was developed on python 3.10; the package requirements can be found in the .toml and .lock files. 
    To install uv on your device, please visit [docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repository
   ```bash
   git clone https://github.com/Dakout/leaf_it_to_renewal.git
   cd branching_out

3. Set up the environment
    ```bash
    uv sync --locked

---

## Usage 

A tutorial notebook is provided, which shows how to:
1. Create time-to-failure and prognostics data with the underlying simulator
2. Evaluate a policy's performance on this data for a single and multiple cost settings

I changed notation multiple times throughout the project, so the code notation does not fit with the paper in all cases. The most prominent case is $\hat{C}_\inf$ which denotes the final evaluated performance of each heuristic in the paper, is denoted as R in the code (following the notation from Kamariotis et al. (2024)).

Furthermore, I only included the heuristic code with the RUL-simulator integration. I skipped the small neural network case study on the C-MAPSS dataset, as I had to change the whole underlying code to make the interface work with the NN's output, which is now very messy. 

I tried to document my code extensively and give the used functions expressive names. However, if things are still not clear or if you have any troubles in running the code, then please contact me via daniel.koutas@tum.de.

---

## Project Information
- **Authors:** Daniel Koutas, Daniel Straub
- **Organization:** ERA Group, TUM 
- **Publication:** Koutas, D., & Straub, D. (2025). Leaf It to Renewal: Improved Predictive Maintenance Policies via Renewal Theory and Decision Trees. arXiv preprint [arXiv:2509.20145](https://arxiv.org/abs/2509.20145).

---

## Citation
If you find it useful for your research, please consider citing it:

**BibTeX Entry:**
```bibtex
@article{koutas2025leaf,
  title={Leaf It to Renewal: Improved Predictive Maintenance Policies via Renewal Theory and Decision Trees},
  author={Koutas, Daniel and Straub, Daniel},
  journal={arXiv preprint arXiv:2509.20145},
  year={2025}
}
```
(Feel free to also give the project a star on GitHub if you like it.)

