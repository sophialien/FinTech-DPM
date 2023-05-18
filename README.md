# FinTech-DPM
This repo is the official implementations of **Contrastive Learning and Reward Smoothing for Deep Portfolio Management**. International Joint Conference on Artificial Intelligence(IJCAI) 2023.

**Abstract**\
In this study, we used reinforcement learning (RL) models to invest assets in order to earn returns. The models were trained to interact with a simulated environment based on historical market data and learn trading strategies. However, using deep neu- ral networks based on the returns of each period can be challenging due to the unpredictability of finan- cial markets. As a result, the policies learned from training data may not be effective when tested in real-world situations. To address this issue, we in- corporated contrastive learning and reward smooth- ing into our training process. Contrastive learning allows the RL models to recognize patterns in as- set states that may indicate future price movements. Reward smoothing, on the other hand, serves as a regularization technique to prevent the models from seeking immediate but uncertain profits. We tested our method against various traditional finan- cial techniques and other deep RL methods, and found it to be effective in both the U.S. stock market and the cryptocurrency market.

<p align="center">
  <img src="https://github.com/sophialien/FinTech-DPM/blob/main/DPM/ContrastiveLearning.png" width="500" />
</p>
