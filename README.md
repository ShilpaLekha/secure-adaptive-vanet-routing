# Secure Adaptive Multi-Strategy Routing in Heterogeneous VANETs under Dynamic Adversarial Conditions

This repository contains the implementation and dataset for the M.Tech research project:

**"Secure Adaptive Multi-Strategy Routing in Heterogeneous VANETs under Dynamic Adversarial Conditions"**

## Overview

The proposed framework enhances routing security in Vehicular Ad-hoc Networks (VANETs) through:

- Heterogeneous node classification
- Mutual authentication before route establishment
- Multi-strategy secure routing
- Dynamic attacker modeling
- Real-time rerouting under adversarial conditions

The framework is evaluated using the roadNet-CA real-world road network dataset.

## Dataset

This project uses the **roadNet-CA** road network dataset from the Stanford Network Analysis Project (SNAP).

Dataset Source:

https://snap.stanford.edu/data/roadNet-CA.html

The dataset file used in the experiments (`roadNet-CA.txt`) is included in this repository.

## Routing Strategies

The framework evaluates four routing strategies:

1. Direct Avoidance Routing
2. K-Shortest Path Routing
3. Weighted Random Routing
4. Node-Disjoint Routing

## Performance Metrics

The following metrics are evaluated:

- Packet Delivery Ratio (PDR)
- Energy Consumption
- End-to-End Delay
- Throughput
- Attack Evasion Rate

## Requirements

- Python 3.x
- NetworkX
- Matplotlib
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Simulation

```bash
python road_simulator.py
```

## Repository Contents

```text
road_simulator.py      # Main simulation code
roadNet-CA.txt         # Road network dataset
requirements.txt       # Python dependencies
README.md              # Project documentation
```

## Student Author

Shilpa M

M.Tech Computer Science and Engineering