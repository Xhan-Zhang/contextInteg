# contextInteg_dm
## Dependencies
The code is tested in Pytorch 1.5.1, Python 3.5, and on MacOS 10.14 and Ubuntu 16.04.
Scikit-learn (http://scikit-learn.org/stable/) is necessary for many analyses.


## Pretrained models
We provide 20 pretrained models and their auxillary data files for analyses. https://drive.google.com/file/d/1DWb8EgATlDEPF5jO9l8-KOT6fGeA374f/view?usp=sharing


## Start to train
Train a network by typing
python main.py --rule_trains contextInteg_decision_making -r2 0 -w2 0 --index 1 -lr 0.0001


## Get started with some simple analyses
After training (you can interrupt at any time), you can do some analysis using the files in the folder \src\analysis.


## Acknowledgement
This code is impossible without the following papers:

(1) G. R. Yang et al. Task representations in neural networks trained to perform many cognitive tasks. Nat. Neurosci., 22, 297 (2019).

(2) Zedong Bi and Changsong Zhou. Understanding the computation of time using neural network models. PNAS, 117, 19 (2020)

(3) Eli Pollock and Mehrdad Jazayeri. Engineering recurrent neural networks from task-relevant manifolds and dynamics. PLoS Comput. Biol., 16 (8) (2020).
