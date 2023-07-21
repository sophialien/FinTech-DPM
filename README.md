# Contrastive Learning and Reward Smoothing for Deep Portfolio Management

[**Video**](https://recorder-v3.slideslive.com/?share=84574&s=305e6cd5-47b2-415f-a731-373dc7697818)


This repo is the official implementations of [**Contrastive Learning and Reward Smoothing for Deep Portfolio Management**](https://people.cs.nctu.edu.tw/~yushuen/data/DeepPortfolioManagement23.pdf). International Joint Conference on Artificial Intelligence(IJCAI) 2023. 

**Authors:** Yun-Hsuan Lien, Yuan-Kui Li, Yu-Shuen Wang 

## Abstract
In this study, we used reinforcement learning (RL) models to invest assets in order to earn returns. The models were trained to interact with a simulated environment based on historical market data and learn trading strategies. However, using deep neu- ral networks based on the returns of each period can be challenging due to the unpredictability of financial markets. As a result, the policies learned from training data may not be effective when tested in real-world situations. To address this issue, we incorporated contrastive learning and reward smoothing into our training process. Contrastive learning allows the RL models to recognize patterns in asset states that may indicate future price movements. Reward smoothing, on the other hand, serves as a regularization technique to prevent the models from seeking immediate but uncertain profits. We tested our method against various traditional finacial techniques and other deep RL methods, and found it to be effective in both the U.S. stock market and the cryptocurrency market.

<p align="center">
  <img src="https://github.com/sophialien/FinTech-DPM/blob/main/DPM/ContrastiveLearning.png" width="500" />
</p>

## Citation
If you find this work useful in your research, please consider citing:
```
@inproceedings{lien2023PPMSRT,
 author={Yun-Hsuan Lien, Yuan-Kui Li, Yu-Shuen Wang},
 booktitle={International Joint Conference on Artificial Intelligence(IJCAI)},
 year={2023}
 }
```

## Contact Information
If you have any questions, please contact Sophia Yun-Hsuan Lien: sophia.yh.lien@gmail.com
