# NoisyFlow Docs

## Method schematics

<p align="center">
  <a href="../assets/Noisyflow-Mar24th-schematics-updated.pdf">
    <img src="../assets/noisyflow-main-schematic-600dpi.png" width="900" alt="NoisyFlow pipeline schematic for private federated generation, transport, synthesis, and downstream evaluation." />
  </a>
</p>
<p align="center">
  <em>Figure 1. NoisyFlow protocol for federated domain adaptation. Each source client trains a label-conditional flow-matching generator on private data. Stage 2 learns a transport map into the target-reference distribution using ICNN/CellOT or flow-matching transport. The server samples from the client generators, applies the learned transports, and trains target-domain classifiers with transported synthetic labels and limited target-reference labels.</em>
</p>

<p align="center">
  <a href="../assets/schematics.pdf">
    <img src="../assets/noisyflow-schematic-600dpi.png" width="760" alt="Additional NoisyFlow schematic for the federated synthetic data generation workflow." />
  </a>
</p>
<p align="center">
  <em>Figure 2. Expanded experimental workflow. The benchmark protocol reports Ref-only, Synth-only, and Ref+Synth classifiers, evaluates target-test accuracy and distributional alignment, and summarizes privacy-utility tradeoffs under DP-SGD.</em>
</p>

Start here:
- Overview: docs/overview.md
- Configuration reference: docs/configuration.md
- Data builders: docs/data.md
- Experiments and CLI: docs/experiments.md
- Attacks: docs/attacks.md
- Code map: docs/architecture.md
