# Multi-Agent Novelty-GuidEd Sample Reuse (MANGER)
Code for AAAI 2025 paper "Novelty-Guided Data Reuse for Efficient and Diversified Multi-Agent Reinforcement Learning". Our approach aims to enhance sample efficiency in multi-agent training by incorporating observation novelty, which has been validated in the SMAC and GRF environments. Specifically, we use an RND network to measure the novelty of each agent's state and assign additional update steps through network splitting. If you wish to execute our method, you can do so by entering the following command:

```python src/main.py --config=qmix_sep --env-config=sc2```

If you want to run the GRF environment, please execute:

```python src/main.py --config=qmix_sep --env-config=gfootball```
