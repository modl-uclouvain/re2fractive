# re<sup>2</sup>fractive

The aim of this project is to design and discover materials with high refractive indices by exploiting new and existing databases, machine learning predictions and high-throughtput DFT calculations, all within a dynamic active learning framework.

This repository accompanies the preprint:

> V. Trinquet, M. L. Evans, C. Hargreaves, P-P. De Breuck, G-M. Rignanese, "Optical materials discovery and design with federated databases and machine learning" (2024).

![[](img/flow.svg)](img/flow.svg)

Some functionality is still missing from the first public release:

- [ ] Direct integration with atomate2/jobflow-remote workflows for automatic
  job submission after candidate selection.
- [ ] Automatic selection according to custom acquisition functions.
