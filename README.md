# A Functional Programming Approach to Composable Bayesian Workflow

Submitted as talk proposal to [Bayes Comp 2023](https://bayescomp2023.com/)

_Abstract_:
Bayesian modeling in practice is an iterative process, in which a practitioner implicitly or explicitly follows the [Bayesian workflow](https://arxiv.org/abs/2011.01808) (Gelman et al 2020) to build models and inferences that are closest to the “reality” within the computational constraints. A composable model building capability is often desired as it makes developing bigger and more complex Bayesian models easier: for example, changing the priors of a collection of random variables. Moreover, a composable approach could enable more flexibility in constructing inferences that optimize for local model structure, thus have the opportunity to improve inference quality compared to using a general inference methods statistical packages offer (e.g., NUTS with different schemes of adaptation). In this talk, I will explain how adopting a functional programming perspective benefits the development of composable Bayesian modeling and programmable inference, with example using [TensorFlow Probability on JAX](https://www.tensorflow.org/probability/examples/TensorFlow_Probability_on_JAX) (for the modeling part) and [Blackjax](https://blackjax-devs.github.io/blackjax/) (for the inference part).

## Set up
```shell
conda create -n bayescomp23 python=3.10
conda activate bayescomp23
pip install -r requirements.txt
```