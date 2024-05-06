# re<sup>2</sup>fractive

The aim of this project is to design and discover materials with high refractive indices by exploiting new and existing databases, machine learning predictions and high-throughtput DFT calculations, all within a dynamic active learning framework.

This repository accompanies the preprint:

> V. Trinquet, M. L. Evans, C. Hargreaves, P-P. De Breuck, G-M. Rignanese, "Optical materials discovery and design with federated databases and machine learning" (2024).

The active learning campaign described their can be repeated with:

```python
from re2fractive.campaign import Campaign, LearningStrategy
from re2fractive.datasets import NaccaratoDataset, MP2023Dataset, Alexandria2024Dataset

learning_strategy = LearningStrategy(
    max_n_features=100,
    feature_select_strategy="always",
    hyperopt_strategy="always",
)

campaign = Campaign.new_campaign_from_dataset(
    NaccaratoDataset,
    datasets=[MP2023Dataset, Alexandria2024Dataset],
    learning_strategy=learning_strategy
)

campaign.run(epochs=8)
```



![[](img/flow.svg)](img/flow.svg)

Some functionality is still missing from the first public release:

- [ ] Direct integration with atomate2/jobflow-remote workflows for automatic
  job submission after candidate selection.
- [ ] Automatic selection according to custom acquisition functions.
