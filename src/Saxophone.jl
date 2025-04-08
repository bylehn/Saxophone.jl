module Saxophone
const VERSION_INDICATOR = "v0.1.2" # Change this when testing reloading
# Import packages used by the module
using LinearAlgebra
using SparseArrays
using Graphs
using DelaunayTriangulation
using Statistics
using Random
using Plots

# Include files with implementations
include("types.jl")
include("utils.jl")
include("visualization.jl")

# Export public interface
export System, initialize!
export visualize_network

end # module Saxophone