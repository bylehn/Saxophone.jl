module Saxophone

using LinearAlgebra
using SparseArrays
using Graphs
using GeometryBasics
using DelaunayTriangulation
using Statistics

# We'll organize by submodules
include("types.jl")        # Core data types
include("utils.jl")        # Utility functions
include("energies.jl")     # Energy calculations
include("simulation.jl")   # Simulation functions 
include("visualization.jl") # Visualization tools

# Exports
export System, initialize!, create_delaunay_graph!
export compute_angle_between_triplet, calculate_angle_triplets
export calculate_initial_angles, poisson_ratio

end # module