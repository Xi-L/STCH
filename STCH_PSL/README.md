The code is mainly designed to be simple and readable, it contains:

- <code>run_stch_psl.py</code> is a ~200-line main file to run the STCH for Pareto set learning (PSL) algorithm;
- <code>model.py</code> is a simple FC Pareto Set model;
- <code>problem.py</code> contains all test problems used in this paper;
- The folder <code>data</code> contains the problem information for the RE problems, which is obtained from the [reproblems repository](https://github.com/ryojitanabe/reproblems).

- If you find the RE problems are useful for your research, please also cite the RE paper:

```
@article{tanabe2020easy,
  title={An easy-to-use real-world multi-objective optimization problem suite},
  author={Tanabe, Ryoji and Ishibuchi, Hisao},
  journal={Applied Soft Computing},
  volume={89},
  pages={106078},
  year={2020},
  publisher={Elsevier}
}
```
