# LSTM-based Feature-Imitating Network (FIN) for sEMG Hand Movement Recognition
This codebase implements a lightweight LSTM-based Feature-Imitating Network (FIN) for surface electromyography (sEMG) signal processing, enabling robust and accurate hand movement recognition with limited labeled data by imitating standard temporal features. Full details of the work were published [here](https://arxiv.org/abs/2405.19356).

## Study Overview
### Abstract
Surface Electromyography (sEMG) is a non-invasive signal that is used in the recognition of hand movement patterns, the diagnosis of diseases, and the robust control of prostheses. Despite the remarkable success of recent end-to-end Deep Learning approaches, they are still limited by the need for large amounts of labeled data. To alleviate the requirement for big data, we propose utilizing a feature-imitating network (FIN) for closed-form temporal feature learning over a 300ms signal window on Ninapro DB2, and applying it to the task of 17 hand movement recognition. We implement a lightweight LSTM-FIN network to imitate four standard temporal features (entropy, root mean square, variance, simple square integral). We observed that the LSTM-FIN network can achieve up to 99% R2 accuracy in feature reconstruction and 80% accuracy in hand movement recognition. Our results also showed that the model can be robustly applied for both within- and cross-subject movement recognition, as well as simulated low-latency environments. Overall, our work demonstrates the potential of the FIN modeling paradigm in data-scarce scenarios for sEMG signal processing.

### Key Contributions
1. We propose an LSTM-based FIN for closed-form temporal feature learning and demonstrate its ability to learn closed-form feature representations, including Entropy (ENT), Root Mean Square (RMS), Variance (VAR), and Simple Square Integral (SSI).
2. We demonstrate the applicability of our LSTM-FIN on a downstream hand movement classification task, outperforming the baseline CNN classifier using ground-truth features.
3. We evaluate the transfer learning capabilities of the LSTM-FIN to unseen subjects.
4. We explore generating future feature values from current time windows to evaluate overall model classification performance in a simulated low-latency scenario.


## Citation
If you find this code useful for your research, please cite our accepted [ICASSP 2025 paper](https://arxiv.org/abs/2405.19356):
```
@article{wu2024lstm,
  title={An LSTM Feature Imitation Network for Hand Movement Recognition from sEMG Signals},
  author={Wu, Chuheng and Atashzar, S Farokh and Ghassemi, Mohammad M and Alhanai, Tuka},
  journal={arXiv preprint arXiv:2405.19356},
  year={2025}
}
```

## Directory Structure
### LSTM model
- _code/lstm_model_fine.py_.

This code implements a Feature Imitating Network (FIN) using an LSTM (Long Short-Term Memory) model for reconstructing specific features of temporal data (i.e. sEMG signal).

### LSTM Feature Imitating Network (FIN) : 
#### Training
- _code/var.py_
- _code/rms.py_
- _code/ssi.py_
- _code/ent.py_.

The scripts train a Feature Imitating Network (FIN) using LSTM for learning the Variance (VAR), Root Mean Square (RMS), Simple Square Integratl (SSI), and Entropy (ENT) features of sEMG signals (see Fig. 1). The objective to train a model to reconstruct each feature of sEMG signal data and evaluates the model's reconstruction performance. The model is an LSTM neural network. The inputs are sEMG signal data and corresponding labels (movement classes). The output is a trained model that can reconstruct RMS features and be evaluated on metrics like accuracy and loss.

#### Validation
- _code/var_val.py_
- _code/rms_val.py_
- _code/ssi_val.py_
- _code/ent_val.py_

These scripts validate the performance of a trained LSTM-based Feature Imitating Network (FIN), specifically for reconstructing each of the four features (VAR, RMS, SSI, ENT) of the sEMG signals.

### CNN training:
  - _code/CNN.py_

This script trains a downstream Convolutional Neural Network (CNN) for classification tasks using closed-form features extracted from surface electromyography (sEMG) signals. The CNN models consists of three convolutional layers: the first layer transforms the input tensor (4 features, 12 channels) into 64 filters using a 5×5 kernel, reducing spatial dimensions, followed by batch normalization, PReLU activation, and dropout for regularization. The second layer further reduces the dimensions with 32 filters, while the third layer outputs a 1x1 spatial representation with 17 filters (one per movement class). This final output is passed through batch normalization and PReLU before being flattened into a classification vector. 
 
### Combining LSTM-FIN + CNN for hand movement classification:
  - _code/fine_tune_percentage_training.py_

This script implements fine-tuning of a pretrained LSTM model and a non-trained (i.e. randomly initialized) CNN model for a classification task involving surface electromyography (sEMG) signals. The process integrates both LSTM-based FIN ouputs (i.e. reconstructed features) and CNN-based downstream classification, using layer-wise gradient descent to optimize the models together. 

### Simulated low-latency:

To generate future-in-time hand movement classification predictions, use the `--prediction` argument to define the number of future time windows to predict. This results in shifting the input and output labels to simulate predicting features for future windows.

### images/: 

Contains figures included in the study paper.


### The recommended sequence for running the code is:
1. LSTM FIN training + validation.
2. CNN training.
3. Combining LSTM-FIN + CNN.

## Code Environment:
- Code was run on SLURM NYU HPC Greene nodes and NYUAD Jubail nodes.
- RTX8000 GPU was used, requiring 64GB in GPU memory.

### Requirements

The code was run using the followning libraries and library version.
```sh
python>=3.8
torch>=1.8.0
```

## Data
1. __Download__ the [Ninapro DB2](https://ninapro.hevs.ch/instructions/DB2.html) dataset developed by [Atzori, M. et al. Electromyography data for non-invasive naturally-controlled robotic hand prostheses. Sci. Data 1:140053 doi: 10.1038/sdata.2014.53 (2014)](https://www.nature.com/articles/sdata201453).
2. __Unzip__ all packages and extract the same exercises (Exercise B) into your project folder and update the `data_dir` variable in the code.

## Acknowledgements
Authors acknowledge the support provided by the [Center for AI and Robotics (CAIR)](https://nyuad.nyu.edu/en/research/faculty-labs-and-projects/center-for-artificial-intelligence-and-robotics.html) at New York University Abu Dhabi (funded by Tamkeen under NYUAD Research Institute Award CG010), and U.S. National Science Foundation (NSF) Award #2320050. Authors acknowledge the technical support provided by the High-Performance Computing team at both NYUAD and NYU.

## Contact
For questions on this code base, please contanct Chuheng Wu, New York University - cw3755@nyu.edu
