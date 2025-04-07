### A Pluto.jl notebook ###
# v0.20.5

using Markdown
using InteractiveUtils

# ╔═╡ 80dbe216-9dc8-4843-b7e4-a8b6d94944ba
# ╠═╡ show_logs = false
begin
    using Pkg
    Pkg.activate("/home/fabian/Documents/network/Saxophone.jl")
    using Revise  # For automatic reloading of code changes
    
    # External packages we need
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

# ╔═╡ cddc7866-765b-48c3-93d5-4eead7fd8982
begin
    include("../src/types.jl")
    include("../src/utils.jl")
    include("../src/visualization.jl")
end

# ╔═╡ d34aa23d-6966-494d-986e-9b0c905c9c2e
system = System(5, 1.0, 42, 2.0, 0.35)

# ╔═╡ f6b89261-908a-487a-a52a-e86d8c8abf3b
initialize!(system)

# ╔═╡ Cell order:
# ╠═2e260a2d-6ebc-436d-a080-af402a16cfef
# ╠═80dbe216-9dc8-4843-b7e4-a8b6d94944ba
# ╠═cddc7866-765b-48c3-93d5-4eead7fd8982
# ╠═d34aa23d-6966-494d-986e-9b0c905c9c2e
# ╠═f6b89261-908a-487a-a52a-e86d8c8abf3b
