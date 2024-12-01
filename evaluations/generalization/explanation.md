## Normalization vs Generalization

- 🟢 With Log-Transform\
  Coefficients are first log-transformed (logarithm to the base 10 of the absolute values plus epsilon), then normalized
- 🔴 Without Log-Transform\
  Coefficients are only normalized

The dataset normalization approach appears to have a significant impact on the generalization ability of the models. In
particular, the log-transform of the coefficients seems to help tremendously.

Without Log-Transform, the models overfit to the trained generator and do not generalize on all unseen generators. While
they generalize on related generators (MelGAN family), they do not generalize on unrelated generators equally well. The
HiFi-GAN, Multi-Band MelGAN and (large) BigVGAN appear to be the generators that are most difficult to detect. While the
recognition rates for the model trained with log-transformed coefficients are still notably worse for these generators,
the discrepancies are not as striking as for the models trained without log-transformed coefficients.

### Evaluation Examples
#### Wide-24 with `sym7` wavelet, trained only on Full-Band MelGAN
<table>
<tr><th>🟢 With Log-Transform</th><th>🔴 Without Log-Transform</th></tr>
<tr>
<td>

| Reference    | Generator            | Acc. max   | Acc. μ ±σ         | EER min   | EER μ ±σ         |
|--------------|----------------------|------------|-------------------|-----------|------------------|
| jsut         | Multi-Band MelGAN    | 59.06%     | 53.36% ±5.45%     | 0.394     | 0.476 ±0.073     |
| jsut         | Parallel WaveGAN     | 66.03%     | 52.05% ±10.63%    | 0.346     | 0.533 ±0.149     |
| ljspeech     | Avocodo              | 80.08%     | 78.37% ±1.74%     | 0.196     | 0.209 ±0.014     |
| ljspeech     | BigVGAN              | 78.61%     | 78.33% ±0.33%     | 0.203     | 0.209 ±0.004     |
| **ljspeech** | **Full-Band MelGAN** | **81.45%** | **80.62% ±0.59%** | **0.183** | **0.190 ±0.005** |
| ljspeech     | HiFi-GAN             | 66.30%     | 65.62% ±0.47%     | 0.302     | 0.315 ±0.009     |
| ljspeech     | Large BigVGAN        | 73.60%     | 73.10% ±0.55%     | 0.243     | 0.253 ±0.007     |
| ljspeech     | MelGAN               | 74.90%     | 74.38% ±0.61%     | 0.234     | 0.239 ±0.004     |
| ljspeech     | Multi-Band MelGAN    | 72.32%     | 71.39% ±0.56%     | 0.267     | 0.273 ±0.009     |
| ljspeech     | Parallel WaveGAN     | 82.81%     | 81.41% ±1.00%     | 0.170     | 0.181 ±0.008     |
| ljspeech     | WaveGlow             | 84.48%     | 83.07% ±0.90%     | 0.156     | 0.170 ±0.008     |
| **ALL**      | -                    | **73.41%** | **71.97% ±1.42%** | **0.256** | **0.277 ±0.021** |

</td>
<td>

| Reference    | Generator            | Acc. max   | Acc. μ ±σ         | EER min   | EER μ ±σ         | 
|--------------|----------------------|------------|-------------------|-----------|------------------|
| jsut         | Multi-Band MelGAN    | 50.27%     | 47.88% ±1.61%     | 0.502     | 0.544 ±0.043     |
| jsut         | Parallel WaveGAN     | 49.85%     | 43.44% ±5.06%     | 0.464     | 0.595 ±0.093     |
| ljspeech     | Avocodo              | 74.33%     | 70.61% ±2.19%     | 0.271     | 0.334 ±0.036     |
| ljspeech     | BigVGAN              | 52.07%     | 51.14% ±0.55%     | 0.559     | 0.609 ±0.032     |
| **ljspeech** | **Full-Band MelGAN** | **96.93%** | **95.64% ±0.94%** | **0.030** | **0.043 ±0.010** |
| ljspeech     | HiFi-GAN             | 52.42%     | 51.55% ±0.60%     | 0.441     | 0.457 ±0.011     |
| ljspeech     | Large BigVGAN        | 59.64%     | 59.06% ±0.63%     | 0.448     | 0.461 ±0.011     |
| ljspeech     | MelGAN               | 82.65%     | 80.12% ±3.07%     | 0.147     | 0.179 ±0.031     |
| ljspeech     | Multi-Band MelGAN    | 88.49%     | 86.70% ±1.42%     | 0.102     | 0.114 ±0.012     |
| ljspeech     | Parallel WaveGAN     | 53.88%     | 52.34% ±0.93%     | 0.512     | 0.586 ±0.051     |
| ljspeech     | WaveGlow             | 79.66%     | 78.17% ±0.91%     | 0.190     | 0.201 ±0.008     |
| **ALL**      | -                    | **65.99%** | **65.15% ±0.55%** | **0.351** | **0.375 ±0.014** |  

</td>
</tr>
</table>

#### Wide-24 with `coif9` wavelet, trained only on Full-Band MelGAN
<table>
<tr><th>🟢 With Log-Transform</th><th>🔴 Without Log-Transform</th></tr>
<tr>
<td>

| Reference    | Generator            | Acc. max   | Acc. μ ±σ         | EER min   | EER μ ±σ         |
|--------------|----------------------|------------|-------------------|-----------|------------------|
| jsut         | Multi-Band MelGAN    | 50.89%     | 49.90% ±1.20%     | 0.444     | 0.517 ±0.053     |
| jsut         | Parallel WaveGAN     | 51.68%     | 47.72% ±4.81%     | 0.445     | 0.591 ±0.106     |
| ljspeech     | Avocodo              | 76.73%     | 75.55% ±0.81%     | 0.225     | 0.234 ±0.008     |
| ljspeech     | BigVGAN              | 77.85%     | 76.58% ±0.95%     | 0.209     | 0.224 ±0.012     |
| **ljspeech** | **Full-Band MelGAN** | **81.55%** | **80.91% ±0.51%** | **0.181** | **0.189 ±0.006** |
| ljspeech     | HiFi-GAN             | 69.57%     | 68.70% ±0.50%     | 0.275     | 0.281 ±0.004     |
| ljspeech     | Large BigVGAN        | 73.46%     | 71.68% ±1.37%     | 0.248     | 0.265 ±0.015     |
| ljspeech     | MelGAN               | 78.17%     | 77.57% ±0.39%     | 0.206     | 0.211 ±0.005     |
| ljspeech     | Multi-Band MelGAN    | 71.73%     | 71.01% ±0.70%     | 0.266     | 0.273 ±0.005     |
| ljspeech     | Parallel WaveGAN     | 80.48%     | 79.16% ±0.87%     | 0.192     | 0.200 ±0.006     |
| ljspeech     | WaveGlow             | 84.63%     | 83.23% ±0.99%     | 0.154     | 0.170 ±0.010     |
| **ALL**      | -                    | **71.53%** | **71.09% ±0.47%** | **0.273** | **0.287 ±0.011** |

</td>
<td>

| Reference    | Generator            | Acc. max   | Acc. μ ±σ         | EER min   | EER μ ±σ         |
|--------------|----------------------|------------|-------------------|-----------|------------------|
| jsut         | Multi-Band MelGAN    | 51.91%     | 50.07% ±1.49%     | 0.478     | 0.508 ±0.018     |
| jsut         | Parallel WaveGAN     | 59.03%     | 49.29% ±7.65%     | 0.396     | 0.529 ±0.089     |
| ljspeech     | Avocodo              | 86.08%     | 81.79% ±3.57%     | 0.141     | 0.192 ±0.045     |
| ljspeech     | BigVGAN              | 51.60%     | 50.85% ±0.45%     | 0.528     | 0.581 ±0.045     |
| **ljspeech** | **Full-Band MelGAN** | **94.85%** | **92.33% ±1.50%** | **0.052** | **0.077 ±0.015** |
| ljspeech     | HiFi-GAN             | 50.79%     | 49.31% ±1.16%     | 0.492     | 0.516 ±0.020     |
| ljspeech     | Large BigVGAN        | 58.12%     | 57.74% ±0.24%     | 0.442     | 0.472 ±0.025     |
| ljspeech     | MelGAN               | 75.93%     | 71.04% ±3.33%     | 0.235     | 0.273 ±0.027     |
| ljspeech     | Multi-Band MelGAN    | 83.24%     | 78.98% ±3.02%     | 0.141     | 0.182 ±0.029     |
| ljspeech     | Parallel WaveGAN     | 58.41%     | 55.46% ±2.11%     | 0.414     | 0.497 ±0.073     |
| ljspeech     | WaveGlow             | 80.33%     | 77.85% ±1.83%     | 0.188     | 0.215 ±0.019     |
| **ALL**      | -                    | **65.59%** | **64.97% ±0.38%** | **0.345** | **0.367 ±0.017** |

</td>
</tr>
</table>
