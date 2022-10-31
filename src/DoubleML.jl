module DoubleML

# Packages
using LinearAlgebra, MLJ, MLJModelInterface, Random, DataFrames
using StatsModels, Distributions, Gadfly, MLJDecisionTreeInterface


export naive_DML, crossfit_DML, MC_sim
include("functions.jl")

# Data generation
export gen_plm
include("genData.jl")


end
