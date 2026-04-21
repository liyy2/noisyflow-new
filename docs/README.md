# NoisyFlow Docs

## Method schematics

<p align="center">
  <a href="../assets/Noisyflow-Mar24th-schematics-updated.pdf">
    <img src="../assets/noisyflow-main-schematic-600dpi.png" width="900" alt="NoisyFlow pipeline schematic for private federated generation, transport, synthesis, and downstream evaluation." />
  </a>
</p>
<p align="center">
  <em>Figure 1. NoisyFlow pipeline. Clients train DP-enabled flow-matching generators locally. Stage 2 learns ICNN- or flow-matching-based transports that align source domains with the target domain. The server synthesizes target-like data for downstream classifier training and evaluation.</em>
</p>

<p align="center">
  <a href="../assets/schematics.pdf">
    <img src="../assets/noisyflow-schematic-600dpi.png" width="760" alt="Additional NoisyFlow schematic for the federated synthetic data generation workflow." />
  </a>
</p>
<p align="center">
  <em>Figure 2. Expanded training and evaluation workflow, including client-side generator training, transport fitting, server-side synthesis, and downstream utility/privacy evaluation.</em>
</p>

Start here:
- Overview: docs/overview.md
- Configuration reference: docs/configuration.md
- Data builders: docs/data.md
- Experiments and CLI: docs/experiments.md
- Attacks: docs/attacks.md
- Code map: docs/architecture.md
