# MARLTCD

[Repository Status](#repository-status) 

[1 TCD Implementation](#1-tcd-implementation) 

[2 Experimental Configurations](#2-experimental-configurations) 

[3 Supporting Data](#3-supporting-data)

## Repository Status {#repository-status}

This repository provides the core resources for reproducing the key findings of our paper:

- **Core TCD Implementation**: The primary algorithm and model code.
- **Experimental Configurations**: Hyperparameter settings and network architecture details.
- **Supporting Data**: Additional experimental data referenced in the manuscript.

> **Note:** We are actively enhancing this repository to improve its usability. This includes adding comprehensive documentation, refactoring auxiliary scripts, and ensuring full reproducibility. Updates will be integrated progressively.

Please feel free to **open an issue** for any immediate questions.

## 1 TCD Implementation {#1-tcd-implementation}

path: MARLTCD\src\modules\TCD_AAO.py & TCD_AOE.py

## 2 Experimental Configurations {#2-experimental-configurations}

path: MARLTCD\src\config\algs

## 3 Supporting Data {#3-supporting-data}

<p align="left">Detailed data for <strong>Figure 6</strong> in the manuscript are presented below:</p>

|     Method      |    MMM2    |  5m_vs_6m   | 3s5z_vs_3s6z |
| :-------------: | :--------: | :---------: | :----------: |
|    QPLEX+TCD    | 93.54±2.32 | 84.02±7.22  | 50.35±10.03  |
| QPLEX+DM w/o CL | 91.67±3.18 | 81.00±5.34  | 48.61±11.48  |
|   QPLEX+FTDM    | 87.50±8.76 | 74.17±11.21 | 36.11±12.38  |
|    QPLEX+FDM    | 84.22±8.42 | 73.96±10.71 |  27.43±9.52  |
|      QPLEX      | 80.21±8.74 | 70.21±10.03 | 24.31±11.34  |

<p align="left">Detailed data for <strong>Figure 7(a)</strong> in the manuscript are presented below:</p>

|  λ   |  τ   |  3s_vs_5z   |
| :--: | :--: | :---------: |
| 0.1  | 0.01 | 98.06±5.40  |
| 0.1  | 0.05 | 95.49±6.11  |
| 0.05 | 0.01 | 95.77±7.22  |
| 0.05 | 0.05 | 96.79±7.29  |
|  0   |  0   | 87.31±8.11  |
| 0.1  | 0.1  | 79.14±13.69 |
| 0.5  | 0.1  | 41.81±35.33 |

<p align="left">Detailed data for <strong>Figure 8</strong> in the manuscript are presented below:</p>

| Method | 3_vs_11_with_keeper |
| :----: | :-----------------: |
|  TCD   |     46.23±11.01     |
|  RND   |      31.6±3.61      |
|  ICM   |     26.93±4.22      |
| QPLEX  |     26.74±4.97      |
| MAVEN  |     15.56±12.15     |

## Thanks

Portions of the code in this repository are built upon and significantly inspired by the following pioneering works. We express our sincere gratitude to the authors (all references are formally cited in the paper):

- The StarCraft Multi-Agent Challenge (SMAC) benchmark and baseline implementations:  
  M. Samvelyan et al., "The StarCraft Multi-Agent Challenge," AAMAS 2019. 

- The Google Research Football environment:  
  K. Kurach et al., "Google Research Football: A Novel Reinforcement Learning Environment," AAAI 2020.
