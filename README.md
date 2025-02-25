# Convergent and Selective Representations in the Insula

This repository contains analysis scripts for the study, Kwon et al. (2025) "Convergent and selective representations of pain, appetitive processes, aversive processes, and cognitive control in the insula"

## Overview

This study analyzes functional convergence across four key task domains (somatic pain, non-somatic appetitive processes, aversive processes, and cognitive control) in a large-scale Bayesian mega-analysis of fMRI data (N=540, systematically sampled from 36 studies). The code in this repository reproduces the analyses from the paper.

## Dependencies

### Required Software

- MATLAB R2019b or newer
- SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)
- MATLAB Statistics and Machine Learning Toolbox
- CanlabCore toolbox (https://github.com/canlab/CanlabCore)

## Repository Structure

### Main Analysis Scripts

- `insula_1_BF_conjunct.mlx`: Performs Bayes Factor analyses to identify domain-general and domain-selective voxels in the insula
- `insula_2_neurosynth.mlx`: Conducts meta-analytic functional decoding using Neurosynth
- `insula_3_coactivation.mlx`: Analyzes coactivation patterns between insular zones and other brain regions
- `insula_4_cytoarchitecture.mlx`: Maps functional zones onto cytoarchitectonic regions using the Julich-Brain Atlas
- `insula_5_neurotransmitter.mlx`: Characterizes neurotransmitter system profiles of the identified insular zones

### Support Vector Machine Analysis

- `SVM/mtSVM_insula_domain.m`: Trains and evaluates multi-class SVM classifiers for domain-level classification
- `SVM/mtSVM_insula_subdomain.m`: Trains and evaluates multi-class SVM classifiers for subdomain-level classification
- `SVM/mtSVMs_insula_apptVSaver_domain.m`: Performs pairwise classification between appetitive and aversive processes domains