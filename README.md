# CER  
**Reinforcement Learning with Conditional Expectation Reward**

### Data Preparation
Prepare the data by running:

```python
python -m recipe.cer.src.data_preparation --data_source TIGER-Lab/WebInstruct-verified --n_repeat 1
python -m recipe.cer.src.data_preparation --data_source DigitalLearningGmbH/MATH-lighteval --n_repeat 1
python -m recipe.cer.src.data_preparation --data_source math-ai/math500 --n_repeat 16
python -m recipe.cer.src.data_preparation --data_source math-ai/amc23 --n_repeat 16
python -m recipe.cer.src.data_preparation --data_source math-ai/aime24 --n_repeat 16
python -m recipe.cer.src.data_preparation --data_source math-ai/aime25 --n_repeat 16
python -m recipe.cer.src.data_preparation --data_source TIGER-Lab/MMLU-Pro --n_repeat 1
python -m recipe.cer.src.data_preparation --data_source m-a-p/SuperGPQA --n_repeat 1
```

### Training and evaluation
Train and evaluate the model by running:

```bash
bash recipe/cer/run.sh
```

### Acknowledgement
This project is based on [verl](https://github.com/volcengine/verl)
