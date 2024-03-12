# 困惑度

困惑度(perplexity)是序列生成模型的评估标准之一。困惑度越小，句子概率越大。困惑度公式为单词概率倒数的几何平均值。对s=w1w2...wN，N为句子长度，PP(s)公式如下：
$$
\begin{align}
PP(s)&=P(w_1w_2...w_N)^{-\frac{1}{N}}\\
     &=\sqrt[N]{\frac{1}{P(w_1w_2...w_N)}}\\
     &=\sqrt[N]{\prod_{i=1}^N{\frac{1}{p(w_i|w_1w_2...w_{i-1})}}}
\end{align}
$$
|V|为词表大小，T为样本数，由于交叉熵代价函数公式：
$$
J=-\frac{1}{T}\sum_{t=1}^{T}\sum_{j=1}^{|V|}y_{t,j}log(\hat{y}_{t,j})
$$
显然可以发现
$$
PP=e^J
$$
当然，这个对所有句子困惑度取了平均值

