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

# ╔═╡ fc514192-6da5-4971-b7f0-2822c25c7624
begin
	# src/energies.jl
	using StaticArrays
	using Zygote # For gradients
	
	# Type aliases for better performance
	const Point2D = SVector{2, Float64}
	const Points2D = Vector{Point2D}
	
	function constrained_force_fn(R::Matrix{Float64}, energy_fn::Function, mask::Matrix{Float64})
	    """
	    Calculates forces with frozen edges.
	    """
	    function new_force_fn(R::Matrix{Float64})
	        # Compute force as negative gradient of energy
	        forces = -Zygote.gradient(energy_fn, R)[1]
	        # Apply mask to zero out forces on frozen nodes
	        forces .*= mask
	        return forces
	    end
	
	    return new_force_fn
	end
	
	function compute_force_norm(fire_state)
	    """Compute the norm of the force vector"""
	    return norm(fire_state.force)
	end
	
	# Convert between standard Matrix and Points2D
	function matrix_to_points(positions::Matrix{Float64})::Points2D
	    n = size(positions, 1)
	    points = Vector{Point2D}(undef, n)
	    @inbounds for i in 1:n
	        points[i] = Point2D(positions[i, 1], positions[i, 2])
	    end
	    return points
	end
	
	function angle_energy(system, positions::Matrix{Float64})::Float64
	    """
	    Calculates the harmonic angle energy for all triplets.
	    """
	    # Convert to Points2D for better performance
	    points = matrix_to_points(positions)
	    return angle_energy_points(system, points)
	end
	
	function angle_energy_points(system, points::Points2D)::Float64
	    """
	    Version of angle_energy that works directly with Points2D.
	    """
	    total_energy = 0.0
	    
	    @inbounds for i in 1:size(system.angle_triplets, 1)
	        triplet = view(system.angle_triplets, i, :)
	        pi = points[triplet[1]]
	        pj = points[triplet[2]]
	        pk = points[triplet[3]]
	        
	        # Calculate current angle - optimized version
	        v1 = pi - pj
	        v2 = pk - pj
	        
	        # Fast normalized dot product
	        n1 = sqrt(v1[1]^2 + v1[2]^2)
	        n2 = sqrt(v2[1]^2 + v2[2]^2)
	        dot_prod = v1[1]*v2[1] + v1[2]*v2[2]
	        
	        cos_angle = dot_prod / (n1 * n2)
	        # Clamp to avoid numerical issues
	        cos_angle = clamp(cos_angle, -1.0, 1.0)
	        current_angle = acos(cos_angle)
	        
	        # Get equilibrium angle
	        eq_angle = system.initial_angles[i]
	        
	        # Calculate harmonic energy
	        k_angle = system.k_angles[i]
	        energy = 0.5 * k_angle * (current_angle - eq_angle)^2
	        
	        # Add crossing penalty if needed
	        if current_angle < system.crossing_penalty_threshold
	            energy += bond_crossing_penalty(system, current_angle)
	        end
	        
	        total_energy += energy
	    end
	    
	    return total_energy
	end
	
	function bond_crossing_penalty(system, angle::Float64)::Float64
	    """
	    Calculate penalty for small angles (potential bond crossings)
	    """
	    da = angle / system.crossing_penalty_threshold
	    
	    if da < 1.0
	        # Apply soft sphere like potential
	        soft_potential = system.crossing_penalty_strength / 2 * (1.0 - da)^2 + 1e-3
	        
	        # Apply sigmoid to smooth transition
	        sigmoid = 1.0 / (1.0 + exp(50.0 * (da - 1.0)))
	        
	        return soft_potential * sigmoid
	    else
	        return 0.0
	    end
	end
	
	function bond_energy(system, positions::Matrix{Float64})::Float64
	    """
	    Calculate spring energy between connected nodes
	    """
	    # Convert to Points2D for better performance
	    points = matrix_to_points(positions)
	    return bond_energy_points(system, points)
	end
	
	function bond_energy_points(system, points::Points2D)::Float64
	    """
	    Version of bond_energy that works directly with Points2D.
	    """
	    total_energy = 0.0
	    
	    @inbounds for i in 1:size(system.E, 1)
	        n1, n2 = system.E[i, :]
	        r1 = points[n1]
	        r2 = points[n2]
	        
	        # Fast distance calculation
	        dx = r1[1] - r2[1]
	        dy = r1[2] - r2[2]
	        dist = sqrt(dx*dx + dy*dy)
	        
	        # Equilibrium distance
	        eq_dist = system.distances[i]
	        
	        # Spring constant
	        k = system.spring_constants[i]
	        
	        # Harmonic energy
	        energy = 0.5 * k * (dist - eq_dist)^2
	        
	        total_energy += energy
	    end
	    
	    return total_energy
	end
	
	function soft_sphere_energy(system, positions::Matrix{Float64})::Float64
	    """
	    Calculate soft sphere repulsion between nodes
	    """
	    # Convert to Points2D for better performance
	    points = matrix_to_points(positions)
	    return soft_sphere_energy_points(system, points)
	end
	
	function soft_sphere_energy_points(system, points::Points2D)::Float64
	    """
	    Version of soft_sphere_energy that works directly with Points2D.
	    """
	    total_energy = 0.0
	    sigma = system.soft_sphere_sigma
	    epsilon = system.soft_sphere_epsilon
	    sigma_sq = sigma^2
	    
	    @inbounds for i in 1:system.N-1
	        for j in (i+1):system.N
	            r_i = points[i]
	            r_j = points[j]
	            
	            # Fast squared distance calculation
	            dx = r_i[1] - r_j[1]
	            dy = r_i[2] - r_j[2]
	            r_sq = dx*dx + dy*dy
	            
	            # Apply soft sphere potential if nodes are closer than sigma
	            if r_sq < sigma_sq
	                r = sqrt(r_sq)
	                ratio = sigma/r
	                ratio6 = ratio^6
	                energy = epsilon * (ratio6^2 - 2*ratio6 + 1)
	                total_energy += energy
	            end
	        end
	    end
	    
	    return total_energy
	end
	
	function total_energy(system, positions::Matrix{Float64})::Float64
	    """
	    Calculate the total energy of the system
	    """
	    # Convert to Points2D for better performance
	    points = matrix_to_points(positions)
	    
	    # Sum all energy components
	    energy = bond_energy_points(system, points) + 
	             angle_energy_points(system, points) +
	             soft_sphere_energy_points(system, points)
	             
	    return energy
	end
	
	function penalty_energy(system, positions::Matrix{Float64})::Float64
	    """
	    Calculate penalty energies only (for minimization)
	    """
	    # Convert to Points2D for better performance
	    points = matrix_to_points(positions)
	    
	    # Calculate crossing penalty
	    crossing_penalty = 0.0
	    @inbounds for i in 1:size(system.angle_triplets, 1)
	        triplet = view(system.angle_triplets, i, :)
	        pi = points[triplet[1]]
	        pj = points[triplet[2]]
	        pk = points[triplet[3]]
	        
	        # Fast angle calculation
	        v1 = pi - pj
	        v2 = pk - pj
	        n1 = sqrt(v1[1]^2 + v1[2]^2)
	        n2 = sqrt(v2[1]^2 + v2[2]^2)
	        dot_prod = v1[1]*v2[1] + v1[2]*v2[2]
	        
	        cos_angle = dot_prod / (n1 * n2)
	        cos_angle = clamp(cos_angle, -1.0, 1.0)
	        angle = acos(cos_angle)
	        
	        if angle < system.crossing_penalty_threshold
	            crossing_penalty += bond_crossing_penalty(system, angle)
	        end
	    end
	    
	    # Calculate soft sphere repulsion
	    node_energy = soft_sphere_energy_points(system, points)
	    
	    # Return normalized penalty
	    return (crossing_penalty + node_energy) / system.N
	end
end

# ╔═╡ 6b4a5dfe-ca91-49eb-83f1-c6344bde0e8c
begin
	# src/simulation.jl
	using Optim
	using BenchmarkTools
	using Profile
	using ProfileView
	using Base.Threads
	
	# Cache for reusing allocated arrays
	mutable struct SimulationCache
	    free_masks::Dict{Vector{Int}, BitVector}
	    position_buffers::Vector{Matrix{Float64}}
	    param_buffers::Vector{Vector{Float64}}
	    
	    function SimulationCache()
	        return new(Dict{Vector{Int}, BitVector}(), 
	                  Vector{Matrix{Float64}}(), 
	                  Vector{Vector{Float64}}())
	    end
	end
	
	const SIM_CACHE = SimulationCache()
	
	function get_free_mask(cache::SimulationCache, n::Int, fixed_nodes::Vector{Int})::BitVector
	    key = sort(fixed_nodes)
	    
	    if haskey(cache.free_masks, key)
	        return cache.free_masks[key]
	    else
	        mask = ones(Bool, n)
	        mask[fixed_nodes] .= false
	        cache.free_masks[key] = mask
	        return mask
	    end
	end
	
	function get_position_buffer(cache::SimulationCache, n::Int)::Matrix{Float64}
	    for (i, buf) in enumerate(cache.position_buffers)
	        if size(buf, 1) == n
	            # Remove from cache and return
	            deleteat!(cache.position_buffers, i)
	            return buf
	        end
	    end
	    
	    # Create new buffer
	    return zeros(Float64, n, 2)
	end
	
	function release_position_buffer!(cache::SimulationCache, buf::Matrix{Float64})
	    push!(cache.position_buffers, buf)
	end
	
	function get_param_buffer(cache::SimulationCache, n::Int)::Vector{Float64}
	    for (i, buf) in enumerate(cache.param_buffers)
	        if length(buf) == n
	            # Remove from cache and return
	            deleteat!(cache.param_buffers, i)
	            return buf
	        end
	    end
	    
	    # Create new buffer
	    return zeros(Float64, n)
	end
	
	function release_param_buffer!(cache::SimulationCache, buf::Vector{Float64})
	    push!(cache.param_buffers, buf)
	end
	
	function extract_free_params!(params::Vector{Float64}, positions::Matrix{Float64}, 
	                            free_mask::BitVector)
	    # Clear params
	    resize!(params, 0)
	    
	    # Extract free parameters
	    @inbounds for i in 1:size(positions, 1)
	        if free_mask[i]
	            push!(params, positions[i, 1])
	            push!(params, positions[i, 2])
	        end
	    end
	    
	    return params
	end
	
	function update_positions!(positions::Matrix{Float64}, params::Vector{Float64}, 
	                         free_mask::BitVector)
	    idx = 1
	    @inbounds for i in 1:size(positions, 1)
	        if free_mask[i]
	            positions[i, 1] = params[idx]
	            positions[i, 2] = params[idx+1]
	            idx += 2
	        end
	    end
	    
	    return positions
	end
	
	function simulate_minimize_penalty(system)
	    """
	    Minimizes the penalty energy (prevents overlaps and crossings)
	    """
	    # Get initial positions
	    R_init = copy(system.X)
	    
	    # Identify surface nodes (fixed during minimization)
	    fixed_nodes = Int[]
	    for values in values(system.surface_nodes)
	        append!(fixed_nodes, values)
	    end
	    unique!(fixed_nodes)
	    
	    # Create a mask for non-fixed nodes
	    free_mask = get_free_mask(SIM_CACHE, system.N, fixed_nodes)
	    
	    # Get position buffer for optimization
	    positions_buffer = get_position_buffer(SIM_CACHE, system.N)
	    copyto!(positions_buffer, R_init)
	    
	    # Get parameter buffer
	    params_buffer = get_param_buffer(SIM_CACHE, 2*sum(free_mask))
	    extract_free_params!(params_buffer, R_init, free_mask)
	    
	    # Function to calculate energy from flattened positions
	    function energy_fn(x)
	        # Update positions
	        update_positions!(positions_buffer, x, free_mask)
	        
	        # Calculate energy
	        return penalty_energy(system, positions_buffer)
	    end
	    
	    # Optimize
	    result = optimize(energy_fn, params_buffer, ConjugateGradient(), 
	                     Optim.Options(iterations=system.steps, 
	                                  g_tol=1e-5))
	    
	    # Reconstruct optimized positions
	    R_final = copy(R_init)
	    update_positions!(R_final, Optim.minimizer(result), free_mask)
	    
	    # Return buffers to cache
	    release_position_buffer!(SIM_CACHE, positions_buffer)
	    release_param_buffer!(SIM_CACHE, params_buffer)
	    
	    return R_init, R_final
	end
	
	function simulate_auxetic(system)
	    """
	    Simulates the auxetic compression process
	    """
	    # Ensure minimized initial configuration
	    R_init, _ = simulate_minimize_penalty(system)
	    system.X = R_init
	    
	    # Initialize parameters
	    num_iterations = ceil(Int, abs(system.perturbation / system.delta_perturbation))
	    
	    # Get surface nodes
	    top_indices = system.surface_nodes["top"]
	    bottom_indices = system.surface_nodes["bottom"]
	    left_indices = system.surface_nodes["left"]
	    right_indices = system.surface_nodes["right"]
	    
	    # Pre-allocate arrays for position log
	    position_log = Vector{Matrix{Float64}}(undef, num_iterations + 1)
	    position_log[1] = copy(R_init)
	    
	    # Get position buffer for optimization
	    R_current = get_position_buffer(SIM_CACHE, system.N)
	    copyto!(R_current, R_init)
	    
	    # Create fixed nodes array once
	    fixed_nodes = vcat(left_indices, right_indices)
	    
	    # Get mask for fixed nodes
	    free_mask = get_free_mask(SIM_CACHE, system.N, fixed_nodes)
	    
	    # Get parameter buffer
	    num_free_params = 2 * (system.N - length(fixed_nodes))
	    params_buffer = get_param_buffer(SIM_CACHE, num_free_params)
	    
	    # For each compression step
	    for step in 1:num_iterations
	        # Apply perturbation to left edge
	        @inbounds for idx in left_indices
	            R_current[idx, 1] += system.delta_perturbation
	        end
	        
	        # Extract free parameters
	        extract_free_params!(params_buffer, R_current, free_mask)
	        
	        # Function to calculate total energy from flattened positions
	        function energy_fn(x)
	            # Update positions
	            update_positions!(R_current, x, free_mask)
	            
	            # Calculate total energy
	            return total_energy(system, R_current)
	        end
	        
	        # Optimize
	        result = optimize(energy_fn, params_buffer, ConjugateGradient(), 
	                         Optim.Options(iterations=system.steps, 
	                                      g_tol=1e-5))
	        
	        # Update positions with optimized values
	        update_positions!(R_current, Optim.minimizer(result), free_mask)
	        
	        # Store the current positions
	        position_log[step + 1] = copy(R_current)
	    end
	    
	    # Final positions
	    R_final = position_log[end]
	    
	    # Calculate Poisson's ratio
	    # Initial dimensions
	    initial_horizontal = mean(R_init[right_indices[2:end-1], 1]) - mean(R_init[left_indices[2:end-1], 1])
	    initial_vertical = mean(R_init[top_indices[2:end-1], 2]) - mean(R_init[bottom_indices[2:end-1], 2])
	    
	    # Final dimensions
	    final_horizontal = mean(R_final[right_indices[2:end-1], 1]) - mean(R_final[left_indices[2:end-1], 1])
	    final_vertical = mean(R_final[top_indices[2:end-1], 2]) - mean(R_final[bottom_indices[2:end-1], 2])
	    
	    # Calculate the Poisson ratio
	    poisson = poisson_ratio(initial_horizontal, initial_vertical, final_horizontal, final_vertical)
	    
	    # Return buffers to cache
	    release_position_buffer!(SIM_CACHE, R_current)
	    release_param_buffer!(SIM_CACHE, params_buffer)
	    
	    return poisson, R_init, R_final, position_log
	end
	
	"""
	Profiling utilities
	"""
	function profile_simulation(system, samples=5)
	    # Clear any existing profiling data
	    Profile.clear()
	    
	    # Start profiling
	    @profile begin
	        for _ in 1:samples
	            poisson, R_init, R_final, log = simulate_auxetic(system)
	        end
	    end
	    
	    # Print text summary
	    Profile.print(format=:flat)
	    
	    # Visualize profile (requires ProfileView)
	    ProfileView.view()
	end
	
	function benchmark_simulation(system)
	    return @benchmark simulate_auxetic($system)
	end
	
	function benchmark_energy_functions(system)
	    # Get a sample position matrix
	    positions = system.X
	    
	    # Benchmark each energy function
	    bond_bench = @benchmark bond_energy($system, $positions)
	    angle_bench = @benchmark angle_energy($system, $positions)
	    sphere_bench = @benchmark soft_sphere_energy($system, $positions)
	    total_bench = @benchmark total_energy($system, $positions)
	    
	    # Print results
	    println("Bond energy:        ", summary(bond_bench))
	    println("Angle energy:       ", summary(angle_bench))
	    println("Soft sphere energy: ", summary(sphere_bench))
	    println("Total energy:       ", summary(total_bench))
	    
	    return (bond=bond_bench, angle=angle_bench, sphere=sphere_bench, total=total_bench)
	end
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

# ╔═╡ 49206f58-688b-49fa-a59d-1ee1f5483c40
md"""
## Energies 
"""

# ╔═╡ 40db0e36-18df-4ce8-9547-d5eeaf985813
md"""
## Simulate
"""

# ╔═╡ d95cfa17-60af-46e0-a47a-639b78f37949
# src/dynamics.jl

function create_compatibility(system, positions)
    """
    Create compatibility matrix relating bond extensions to nodal displacements
    
    system: System with connectivity information
    positions: Matrix of node positions
    
    Returns: Compatibility matrix
    """
    N_b = size(system.E, 1)  # Number of bonds
    
    # Initialize compatibility matrix
    C = zeros(2 * system.N, N_b)
    
    # For each bond
    for i in 1:N_b
        # Get nodes connected by this bond
        n1, n2 = system.E[i, :]
        
        # Compute bond vector
        b_vec = positions[n1, :] - positions[n2, :]
        
        # Normalize
        b_vec_norm = norm(b_vec)
        b_vec_normalized = b_vec / b_vec_norm
        
        # Update compatibility matrix
        C[2*n1-1:2*n1, i] .+= b_vec_normalized
        C[2*n2-1:2*n2, i] .-= b_vec_normalized
    end
    
    return C
end

# ╔═╡ 3406051d-74fa-4e8e-9671-c1a21b7eee02
function get_forbidden_states(system, C, spring_constants)
    """
    Calculate eigenvalues and eigenvectors of the dynamical matrix
    
    system: System with mass matrix
    C: Compatibility matrix
    spring_constants: Vector of spring constants
    
    Returns: Eigenvalues, eigenvectors, forbidden states count, frequencies
    """
    # Create diagonal matrix of spring constants
    k_diag = Diagonal(spring_constants)
    
    # Construct stiffness matrix
    K = C * k_diag * C'
    
    # Construct dynamical matrix
    D_mat = inv(system.mass) * K
    
    # Calculate eigenvalues and eigenvectors
    eigen_result = eigen(Hermitian(D_mat))
    D = eigen_result.values
    V = eigen_result.vectors
    
    # Calculate frequencies
    frequency = sqrt.(abs.(D))
    
    # Count forbidden states (modes within the bandgap)
    w_low = system.frequency_center - system.frequency_width/2
    w_high = system.frequency_center + system.frequency_width/2
    forbidden_states = count(w -> w_low < w < w_high, frequency)
    
    return D, V, forbidden_states, frequency
end

# ╔═╡ 03e128ed-93b5-4f8a-bacb-516cebbb3162
function forbidden_states_compression(system)
    """
    Analyze forbidden modes before and after compression
    
    system: System with all parameters
    
    Returns: Initial and final mode data, Poisson ratio
    """
    # Perform compression simulation
    poisson, R_init, R_final, log = simulate_auxetic(system)
    
    # Create compatibility matrices
    C_init = create_compatibility(system, R_init)
    C_final = create_compatibility(system, R_final)
    
    # Calculate eigenvalues and eigenvectors
    D_init, V_init, forbidden_states_init, frequency_init = 
        get_forbidden_states(system, C_init, system.spring_constants)
    
    D_final, V_final, forbidden_states_final, frequency_final = 
        get_forbidden_states(system, C_final, system.spring_constants)
    
    # Organize results
    result = (
        D_init = D_init,
        V_init = V_init,
        C_init = C_init,
        forbidden_states_init = forbidden_states_init,
        frequency_init = frequency_init,
        R_init = R_init,
        D_final = D_final,
        V_final = V_final,
        C_final = C_final,
        forbidden_states_final = forbidden_states_final,
        frequency_final = frequency_final,
        R_final = R_final,
        log = log,
        poisson = poisson
    )
    
    return result
end

# ╔═╡ f2474468-71e5-4f00-98fe-7aceb66e9be0
begin
	# Add to src/visualization.jl
	
	function visualize_compression(system, R_init, R_final)
	    """
	    Create a visualization showing initial and final (compressed) states
	    
	    system: System containing network data
	    R_init: Initial positions
	    R_final: Final positions
	    
	    Returns: Plot with both states
	    """
	    p1 = plot(
	        aspect_ratio=:equal,
	        xlim=(0, system.nr_points+1),
	        ylim=(0, system.nr_points+1),
	        legend=false,
	        grid=false,
	        title="Initial Configuration"
	    )
	    
	    p2 = plot(
	        aspect_ratio=:equal,
	        xlim=(0, system.nr_points+1),
	        ylim=(0, system.nr_points+1),
	        legend=false,
	        grid=false,
	        title="Compressed Configuration"
	    )
	    
	    # Plot initial state
	    for i in 1:size(system.E, 1)
	        n1, n2 = system.E[i, :]
	        x1, y1 = R_init[n1, :]
	        x2, y2 = R_init[n2, :]
	        
	        # Line thickness proportional to spring constant
	        thickness = 2 * system.spring_constants[i] * system.distances[i]
	        plot!(p1, [x1, x2], [y1, y2], color=:black, alpha=0.6, linewidth=thickness)
	    end
	    
	    # Plot nodes
	    scatter!(p1, R_init[:,1], R_init[:,2], 
	        color=:black, 
	        alpha=0.25, 
	        markersize=system.soft_sphere_sigma*10
	    )
	    
	    # Plot final state
	    for i in 1:size(system.E, 1)
	        n1, n2 = system.E[i, :]
	        x1, y1 = R_final[n1, :]
	        x2, y2 = R_final[n2, :]
	        
	        # Line thickness proportional to spring constant
	        thickness = 2 * system.spring_constants[i] * system.distances[i]
	        plot!(p2, [x1, x2], [y1, y2], color=:black, alpha=0.6, linewidth=thickness)
	    end
	    
	    # Plot nodes
	    scatter!(p2, R_final[:,1], R_final[:,2], 
	        color=:black, 
	        alpha=0.25, 
	        markersize=system.soft_sphere_sigma*10
	    )
	    
	    # Combine plots
	    return plot(p1, p2, layout=(1,2), size=(1000, 400))
	end
	
	function visualize_density_of_states(system, frequencies; bins=50)
	    """
	    Visualize the density of states (frequency distribution)
	    
	    system: System with frequency data
	    frequencies: Vector of mode frequencies
	    bins: Number of histogram bins
	    
	    Returns: Histogram plot of frequencies
	    """
	    p = histogram(
	        frequencies,
	        bins=bins,
	        xlabel="Frequency",
	        ylabel="Number of modes",
	        title="Density of States",
	        legend=false,
	        alpha=0.7,
	        color=:blue
	    )
	    
	    # Mark bandgap region
	    w_low = system.frequency_center - system.frequency_width/2
	    w_high = system.frequency_center + system.frequency_width/2
	    
	    vspan!([w_low, w_high], color=:red, alpha=0.2, label="Bandgap")
	    
	    return p
	end
end

# ╔═╡ 95da9dad-0973-448e-8191-0cef22fc94b0
md"""
## Test simulation
"""

# ╔═╡ 88d9556a-0a0a-48ab-a308-6247374f90e1
begin
    # Set acoustic and auxetic parameters
    system.frequency_center = 2.0
    system.frequency_width = 0.2
    system.perturbation = 1.0
    system.delta_perturbation = 0.2
    system.steps = 50
    
    # Run compression
    poisson, R_init, R_final, log = simulate_auxetic(system)
    
    println("Poisson's ratio: ", poisson)
    
    # Visualize
    visualize_compression(system, R_init, R_final)
end

# ╔═╡ Cell order:
# ╠═2e260a2d-6ebc-436d-a080-af402a16cfef
# ╠═3e6bc5c6-117d-4e42-8d2a-f973ea3fab65
# ╠═c5c2c26a-4d35-4690-8d9e-9615894d1b61
# ╠═0ba46c00-da69-4b93-ae7a-83c76be57a09
# ╠═0c921266-57d0-4ae4-b24c-842555ce8a5c
# ╠═885b819e-46a5-4902-964b-2fd36e8d141d
# ╠═d2295335-23f5-4007-98ce-4c61a00b96bb
# ╟─49206f58-688b-49fa-a59d-1ee1f5483c40
# ╠═fc514192-6da5-4971-b7f0-2822c25c7624
# ╟─40db0e36-18df-4ce8-9547-d5eeaf985813
# ╠═6b4a5dfe-ca91-49eb-83f1-c6344bde0e8c
# ╠═d95cfa17-60af-46e0-a47a-639b78f37949
# ╠═3406051d-74fa-4e8e-9671-c1a21b7eee02
# ╠═03e128ed-93b5-4f8a-bacb-516cebbb3162
# ╠═f2474468-71e5-4f00-98fe-7aceb66e9be0
# ╟─95da9dad-0973-448e-8191-0cef22fc94b0
# ╠═88d9556a-0a0a-48ab-a308-6247374f90e1
