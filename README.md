# Dual Quality Assessment

This GitHub repository contains the official code for the papers,

> [Adversarial robustness assessment: Why in evaluation both L0 and L∞ attacks are necessary](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0265723)\
> Shashank Kotyan and Danilo Vasconcellos Vargas, \
> PLOS One (2022).

> [One pixel attack for fooling deep neural networks](https://ieeexplore.ieee.org/abstract/document/8601309)\
> Jiawei Su, Danilo Vasconcellos Vargas, Kouichi Sakurai\
> IEEE Transactions on Evolutionary Computation (2019).
 
## Citation

If this work helps your research and/or project in anyway, please cite:

```bibtex
@article{kotyan2022adversarial,
  title={Adversarial robustness assessment: Why in evaluation both L 0 and L∞ attacks are necessary},
  author={Kotyan, Shashank and Vargas, Danilo Vasconcellos},
  journal={PloS one},
  volume={17},
  number={4},
  pages={e0265723},
  year={2022},
  publisher={Public Library of Science San Francisco, CA USA}
}

@article{su2019one,
  title     = {One pixel attack for fooling deep neural networks},
  author    = {Su, Jiawei and Vargas, Danilo Vasconcellos and Sakurai, Kouichi},
  journal   = {IEEE Transactions on Evolutionary Computation},
  volume    = {23},
  number    = {5},
  pages     = {828--841},
  year      = {2019},
  publisher = {IEEE}
}
```

## Testing Environment

The code is tested on Ubuntu 18.04.3 with Python 3.7.4.

## Getting Started

### Requirements

To run the code in the tutorial locally, it is recommended, 
- a dedicated GPU suitable for running, and
- install Anaconda. 

The following python packages are required to run the code. 
- `cma==2.7.0`
- `matplotlib==3.1.1`
- `numpy==1.17.2`
- `pandas==0.25.1`
- `scipy==1.4.1`
- `seaborn==0.9.0`
- `tensorflow==2.1.0`

---

### Steps

1. Clone the repository.

```bash
git clone https://github.com/shashankkotyan/DualQualityAssessment.git
cd ./DualQualityAssessment
```

2. Create a virtual environment 

```bash
conda create --name dqa python=3.7.4
conda activate dqa
```

3. Install the python packages in `requirements.txt` if you don't have them already.

```bash
pip install -r ./requirements.txt
```

4. Run an adversarial attack with the following command.

    a) Run the Pixel Attack with the following command

    ```bash
    python -u code/run_attack.py pixel [ARGS] > run.txt
    ```

    b) Run the Threshold Attack with the following command

    ```bash
    python -u code/run_attack.py threshold [ARGS] > run.txt
    ```

<!--
To be Included

5. Calculate the statstics for the attacks.


# ```bash
# python -u code/run_stats.py > run_stats.txt     
# ```
-->

## Arguments for run_attack.py

TBD

## Notes

TBD

## Milestones

- [ ] Tutorials
- [ ] Addition of Comments in the Code
- [ ] Cross Platform Compatibility
- [ ] Description of Method in Readme File

## License

Dual Quality Assessment is licensed under the MIT license. 
Contributors agree to license their contributions under the MIT license.

## Contributors and Acknowledgements

TBD

## Reaching out

You can reach me at shashankkotyan@gmail.com or [\@shashankkotyan](https://twitter.com/shashankkotyan).
If you tweet about Dual Quality Assessment, please use one of the following tags `#pixel_attack`, `#threshold_attack`, `#dual_quality_assessment`,  and/or mention me ([\@shashankkotyan](https://twitter.com/shashankkotyan)) in the tweet.
For bug reports, questions, and suggestions, use [Github issues](https://github.com/shashankkotyan/DualQualityAssessment/issues).
