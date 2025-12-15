# MARLTCD



## Repository Status

This repository provides the core resources for reproducing the key findings of our paper:

- **Core TCD Implementation**: The primary algorithm and model code.
- **Experimental Configurations**: Hyperparameter settings and network architecture details.
- **Supporting Data**: Additional experimental data referenced in the manuscript.

> **Note:** We are actively enhancing this repository to improve its usability. This includes adding comprehensive documentation, refactoring auxiliary scripts, and ensuring full reproducibility. Updates will be integrated progressively.

Please feel free to **open an issue** for any immediate questions.

## 1 TCD Implementation

path: MARLTCD\src\modules\TCD_AAO.py & TCD_AOE.py

## 2 Experimental Configurations

path: MARLTCD\src\config\algs

## 3 Supporting Data

<p align="left">Detailed data for <strong>Figure 6</strong> in the manuscript are presented below:</p>

<!DOCTYPE html>
<html>
<head>
    <style>
        table {
            border-collapse: collapse;
            width: auto;
            margin: 20px auto;
            font-family: Arial, sans-serif;
        }
        th, td {
            border: 1px solid black;
            padding: 8px 12px;
            text-align: center;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .scenario-header {
            background-color: #f2f2f2;
        }
        .best-score {
            font-weight: bold;
        }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th rowspan="2">Method</th>
                <th colspan="3">Scenario</th>
            </tr>
            <tr>
                <th>MMM2</th>
                <th>5m_vs_6m</th>
                <th>3s5z_vs_3s6z</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>QPLEX+TCD</td>
                <td>93.54±2.32</td>
                <td>84.02±7.22</td>
                <td>50.35±10.03</td>
            </tr>
            <tr>
                <td>QPLEX+DM w/o CL</td>
                <td>91.67±3.18</td>
                <td>81.00±5.34</td>
                <td>48.61±11.48</td>
            </tr>
            <tr>
                <td>QPLEX+FTDM</td>
                <td>87.50±8.76</td>
                <td>74.17±11.21</td>
                <td>36.11±12.38</td>
            </tr>
            <tr>
                <td>QPLEX+FDM</td>
                <td>84.22±8.42</td>
                <td>73.96±10.71</td>
                <td>27.43±9.52</td>
            </tr>
            <tr>
                <td>QPLEX</td>
                <td>80.21±8.74</td>
                <td>70.21±10.03</td>
                <td>24.31±11.34</td>
            </tr>
        </tbody>
    </table>
</body>
</html>

<p align="left">Detailed data for <strong>Figure 7</strong> in the manuscript are presented below:</p>



<p align="left">Detailed data for <strong>Figure 8</strong> in the manuscript are presented below:</p>



## Thanks

Portions of the code in this repository are built upon and significantly inspired by the following pioneering works. We express our sincere gratitude to the authors (all references are formally cited in the paper):

- The StarCraft Multi-Agent Challenge (SMAC) benchmark and baseline implementations:  
  M. Samvelyan et al., "The StarCraft Multi-Agent Challenge," AAMAS 2019. 

- The Google Research Football environment:  
  K. Kurach et al., "Google Research Football: A Novel Reinforcement Learning Environment," AAAI 2020.
