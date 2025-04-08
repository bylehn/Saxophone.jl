### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 3e6bc5c6-117d-4e42-8d2a-f973ea3fab65
begin
    using Pkg
    Pkg.activate("..")
    
    # External packages
    using LinearAlgebra
    using SparseArrays
    using Graphs
    using DelaunayTriangulation
    using Statistics
    using Random
    using Plots
end

# ╔═╡ 2e260a2d-6ebc-436d-a080-af402a16cfef
begin
	using Markdown
	using InteractiveUtils
end

# ╔═╡ c5c2c26a-4d35-4690-8d9e-9615894d1b61
begin
    # Include source files directly
    include("../src/types.jl")    # This defines your System type
    include("../src/utils.jl")    # This includes your utility functions
    include("../src/visualization.jl") # This includes visualization functions
end

# ╔═╡ 0ba46c00-da69-4b93-ae7a-83c76be57a09
# Create a test system
system = System(15, 1.0, 44, 2.0, 0.35)

# ╔═╡ 0c921266-57d0-4ae4-b24c-842555ce8a5c
# Initialize the system
initialize!(system)

# ╔═╡ 885b819e-46a5-4902-964b-2fd36e8d141d
# Visualize the network
visualize_network(system)

# ╔═╡ d2295335-23f5-4007-98ce-4c61a00b96bb
# Inspect some system properties
begin
    println("Number of nodes: ", system.N)
    println("Number of edges: ", size(system.E, 1))
    println("Number of angle triplets: ", size(system.angle_triplets, 1))
end

# ╔═╡ Cell order:
# ╠═2e260a2d-6ebc-436d-a080-af402a16cfef
# ╠═3e6bc5c6-117d-4e42-8d2a-f973ea3fab65
# ╠═c5c2c26a-4d35-4690-8d9e-9615894d1b61
# ╠═0ba46c00-da69-4b93-ae7a-83c76be57a09
# ╠═0c921266-57d0-4ae4-b24c-842555ce8a5c
# ╠═885b819e-46a5-4902-964b-2fd36e8d141d
# ╠═d2295335-23f5-4007-98ce-4c61a00b96bb
