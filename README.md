# collabo

<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

A package for human-algorithm collaborative Bayesian optimization.

# Installation 

```
$ pip install collabo
```

# Usage

Collabo can be used without a specific function for scenarios involving real world experiments. Here, objective values are manually input. Optionally a function can be provided and objective values are evaluated automatically. 

```python
import collabo as cb 

# initialize a collaborator object
colleague = cb.Collaborator('./data.json',bounds=[(0,5),(0,5),(0,5),(0,5)])
colleague.design_experiments(10) # design experiments
collab.propose_solutions(3) # propose solutions
collab.view_choices() # view solutions 
collab.make_choice(2) # choose solution to evaluate 
```

Collabo also enables expert-informed design of experiments by optimally distributing initial solutions around fixed pre-defined solutions. 

```python
collab.design_experiments(10,fixed_solutions=[[1,1,1,1],[2,2,2,2],[3,3,3,3]])
```

Data can also be loaded from a previously generated JSON file, or existing experiments can be used providing they are in the following correct format.

```json
{"experiments": [{"solution": [0.0, 0.0, 3.3, 3.3], "objective": 6.6}, {"solution": [1.6, 3.3, 1.6, 1.6], "objective": 8.3}, {"solution": [3.3, 1.6, 5.0, 0.0], "objective": 10.0}, {"solution": [5.0, 5.0, 0.0, 5.0], "objective": 15.0}, {"solution": [5.0, 0.0, 4.98, 4.99], "objective": 14.98}]}
```

### todo

- Improve data loading and current state. 
- Store and visualize choices proposed and selected.