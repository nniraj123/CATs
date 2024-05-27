# Cross-Attractor Transforms (CATs)

CATs are a pair of transformation between the phase spaces of reference truth and the 
imperfect model. These inexpensive mappings have the potential to improve
forecasts from imperfect models whose states lie on a totally different 
attractor than the reference truth.

This repository trains and evaluates CATs for the Lorenz'96 chaotic model. 

## Description

This repository contains both ipython notebooks and python scripts. The 
notebooks are helpful in understanding the theory and implementation. The 
python scripts were mostly used for training different flavours of CATs for 
different lead times.

`CATs_L96.ipynb` is the main notebook that implements CATs for Lorenz'96 as 
discussed in the paper, whereas `` 
    

## Data
Lorenz'96 model equations are numerically integrated to obtain the dataset 
required for training CATs. These are embedded within the notebooks and the 
python scripts.

   
