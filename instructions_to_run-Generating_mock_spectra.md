Here are the instructions to run various codes with recommended workflow.

*21cmFAST_coeval_LOS.py*

python 21cmFAST_coeval_LOS.py

This code runs a 21cmFAST simulation version 3.3.1 and generates Nlos LOS data with IGM fields (overdensity, neutral hydrogen fraction, gas temperature, peculiar velocity, photoionization rate, J21). The LOS are uniformly distributed in the simulation box along all three axes. The output is saved in datasets/21cmFAST_los/los/.

Note - set the relevant parameters before running the code. The parameters are set in the code itself (lines 30-50)