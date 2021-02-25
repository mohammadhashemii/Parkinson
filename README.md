# A Neural-based Approach to Aid Early Parkinson’s Disease Diagnosis

by 
Dr. Armin Salimi-Badr,
Mohammad Hashemi

This paper has been submitted for *[ikt-2020 conference](http://iktconference.ir/2020)* and was publishd in *[IEEE](https://ieeexplore.ieee.org/abstract/document/9345635)*.

![](https://github.com/M-Hsh/Parkinson/blob/main/images/sample.png)

*A random sample(patient subject) which used for training the proposed model.*

## Abstract

> In this paper, a neural approach based on using Long-Short Term Memory (LSTM) neural networks is proposed to diagnose patients suffering from PD. 
In this study, it is shown that the temporal patterns of the gait cycle are different for healthy persons and patients. 
Therefore, by using a recurrent structure like LSTM, able to analyze the dynamic nature of the gait cycle, the proposed method extracts the temporal patterns 
to diagnose patients from healthy persons. Utilized data to extract the temporal shapes of the gait cycle are based on changing 
vertical Ground Reaction Force (vGRF), measured by 16 sensors placed in the soles of shoes worn by each subject. To reduce the number of data dimensions, 
the sequences of corresponding sensors placed in different feet are combined by subtraction. 
This method analyzes the temporal pattern of time- series collected from different sensors, without extracting special features representing statistics of 
different parts of the gait cycle. Finally, by recording and presenting data from 10 seconds of subject walking, the proposed approach can diagnose the patient 
from healthy persons with an average accuracy of 97.66%, and the total F1 score equal to 97.78%.

All the tools for data pre-processing used to generate the appropriate data fetched into the proposed model are in
the `my_utils.py` python file.
The calculations and figure generation are all run inside
[Jupyter notebooks](http://jupyter.org/).
The data used in this study was obtained from [Physionet](https://physionet.org/content/gaitpdb/1.0.0/) and data preprocessing which is used for this project
is described in the following.

## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/M-Hsh/Parkinson.git

or [download a zip archive](https://github.com/M-Hsh/Parkinson.zip).

## Requirements

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.yml`.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command in the repository folder (where `requirements.yml`
is located) to create a separate environment and install all required
dependencies in it:

    conda env create

## Citation
If you find our code useful in your research, please, consider citing our paper:

> @INPROCEEDINGS{9345635,  author={A. {Salimi-Badr} and M. {Hashemi}},

> booktitle={2020 11th International Conference on Information and Knowledge Technology (IKT)},


> title={A Neural-Based Approach to Aid Early Parkinson's Disease Diagnosis},

> year={2020},

> doi={10.1109/IKT51791.2020.9345635}}
