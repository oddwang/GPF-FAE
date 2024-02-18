# Procedural Fairness in Machine Learning

This is the code for the paper "Procedural Fairness in Machine Learning", in which we propose a metric to evaluate procedural fairness of ML models, and propose two methods to imporve model's procedural fairness.

### Personal Use Only. No Commercial Use.

Part of the code that improves the model's procedural fairness is based on the "You shouldn't trust me: Learning models which conceal unfairness from multiple explanation methods": (https://github.com/bottydim/adversarial_explanations).
## Running experiments

Evaluating the procedural fairness of the model:

```
python GPF_FAE_metric.py
```
Two methods to improve the procedural fairness of ML models:

```
python retraining_method.py
python modifying_method.py
```

## Dependencies

We require the following dependencies:
- aif360==0.5.0
- dill==0.3.7
- Keras==2.3.1
- lime==0.2.0.1
- matplotlib==3.5.3
- numpy==1.21.6
- pandas==1.1.5
- scikit_learn==1.0.2
- scipy==1.7.3
- seaborn==0.13.2
- shap==0.41.0
- tensorflow==1.14.0
- torch==1.12.1
