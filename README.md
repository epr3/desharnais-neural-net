### Description

This project is used for the calculations and layer compositions described in the [article](article.pdf).

It uses the Desharnais dataset to estimate effort required for software development projects using both a MLP and LSTM neural net.

It generates a png and a csv for each type of neural net used containing the predicted effort, the actual effort and the magnitude of relative error. The png contains the layer composition.

More details in the article.

### Requirements
1. Python 3.11
2. [Poetry](https://python-poetry.org/)
3. [Graphviz](https://graphviz.gitlab.io/)

### Installation

1. Install dependencies.

```poetry install```

2. Run scripts to generate the csv and plots

```poetry run python mlp.py```

```poetry run python lstm.py```

