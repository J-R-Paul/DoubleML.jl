module DoubleML

# Packages
using LinearAlgebra, MLJ, Random, DataFrames
using StatsModels, Distributions, Gadfly


export naive_DML, crossfit_DML, MC_sim
include("functions.jl")

# Data generation
export gen_plm
include("genData.jl")


end
