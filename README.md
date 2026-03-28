# Immune Checkpoint Blockade ODE Model

## Overview
This repository contains an implementation of the immune checkpoint blockade (ICB) ordinary differential equations (ODE) model. This model simulates the dynamics of tumor growth in response to immune checkpoint therapies.

## Extensions
### Monte Carlo Simulation
The model includes a Monte Carlo extension which allows for the simulation of variability in model parameters to assess uncertainty in model predictions.

### Dosing Schedules
Additionally, various dosing schedules can be implemented to explore the effects of different treatment timelines on tumor response.

## Installation Instructions
1. Clone the repository:
   ```
   git clone https://github.com/thepathyougo299-sudo/Check_Point_Model_382_LHD.git
   cd Check_Point_Model_382_LHD
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
To run the simulation, use the following command:
```bash
python run_simulation.py --param <parameter_file>
```
Replace `<parameter_file>` with the path to your parameters file.

## Outputs
The model generates several output files that include:
- Tumor volume over time
- Immune cell populations
- Model parameter summary

For detailed output information, please refer to the documentation within the codebase and the examples provided.