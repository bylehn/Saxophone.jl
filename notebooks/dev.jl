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

# ╔═╡ 49206f58-688b-49fa-a59d-1ee1f5483c40
md"""
## Energies 
"""

# ╔═╡ fc514192-6da5-4971-b7f0-2822c25c7624
begin
	# src/energies.jl
	
	function constrained_force_fn(R, energy_fn, mask)
	    """
	    Calculates forces with frozen edges.
	
	    R: position matrix
	    energy_fn: energy function
	    mask: mask for frozen edges (1 for free, 0 for frozen)
	
	    Returns: A function that computes the total force with frozen edges
	    """
	    function new_force_fn(R)
	        # Compute force as negative gradient of energy
	        forces = -gradient(energy_fn, R)
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
	
	function angle_energy(system, positions)
	    """
	    Calculates the harmonic angle energy for all triplets.
	    
	    system: System containing angle data
	    positions: Matrix of node positions
	    
	    Returns: Total angle energy
	    """
	    total_energy = 0.0
	    
	    for i in 1:size(system.angle_triplets, 1)
	        triplet = system.angle_triplets[i, :]
	        pi = positions[triplet[1], :]
	        pj = positions[triplet[2], :]
	        pk = positions[triplet[3], :]
	        
	        # Calculate current angle
	        current_angle = compute_angle_between_triplet(pi, pj, pk)
	        
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
	
	function bond_crossing_penalty(system, angle)
	    """
	    Calculate penalty for small angles (potential bond crossings)
	    
	    system: System containing penalty parameters
	    angle: Current angle in radians
	    
	    Returns: Penalty energy
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
	
	function bond_energy(system, positions)
	    """
	    Calculate spring energy between connected nodes
	    
	    system: System containing connectivity and spring constants
	    positions: Matrix of node positions
	    
	    Returns: Total bond energy
	    """
	    total_energy = 0.0
	    
	    for i in 1:size(system.E, 1)
	        n1, n2 = system.E[i, :]
	        r1 = positions[n1, :]
	        r2 = positions[n2, :]
	        
	        # Current distance
	        dist = norm(r1 - r2)
	        
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
	
	function soft_sphere_energy(system, positions)
	    """
	    Calculate soft sphere repulsion between nodes
	    
	    system: System with soft sphere parameters
	    positions: Matrix of node positions
	    
	    Returns: Total repulsion energy
	    """
	    total_energy = 0.0
	    sigma = system.soft_sphere_sigma
	    epsilon = system.soft_sphere_epsilon
	    
	    # For all pairs of nodes
	    for i in 1:system.N
	        for j in (i+1):system.N
	            r_i = positions[i, :]
	            r_j = positions[j, :]
	            
	            r = norm(r_i - r_j)
	            
	            # Apply soft sphere potential if nodes are closer than sigma
	            if r < sigma
	                energy = epsilon * ((sigma/r)^12 - 2*(sigma/r)^6 + 1)
	                total_energy += energy
	            end
	        end
	    end
	    
	    return total_energy
	end
	
	function total_energy(system, positions)
	    """
	    Calculate the total energy of the system
	    
	    system: System containing all parameters
	    positions: Matrix of node positions
	    
	    Returns: Total energy
	    """
	    # Sum all energy components
	    energy = bond_energy(system, positions) + 
	             angle_energy(system, positions) +
	             soft_sphere_energy(system, positions)
	             
	    return energy
	end
	
	function penalty_energy(system, positions)
	    """
	    Calculate penalty energies only (for minimization)
	    
	    system: System containing parameters
	    positions: Matrix of node positions
	    
	    Returns: Penalty energy per node
	    """
	    # Calculate crossing penalty
	    crossing_penalty = 0.0
	    for i in 1:size(system.angle_triplets, 1)
	        triplet = system.angle_triplets[i, :]
	        pi = positions[triplet[1], :]
	        pj = positions[triplet[2], :]
	        pk = positions[triplet[3], :]
	        
	        angle = compute_angle_between_triplet(pi, pj, pk)
	        
	        if angle < system.crossing_penalty_threshold
	            crossing_penalty += bond_crossing_penalty(system, angle)
	        end
	    end
	    
	    # Calculate soft sphere repulsion
	    node_energy = soft_sphere_energy(system, positions)
	    
	    # Return normalized penalty
	    return (crossing_penalty + node_energy) / system.N
	end
end

# ╔═╡ 6b4a5dfe-ca91-49eb-83f1-c6344bde0e8c
begin
	# src/simulation.jl
	using Optim  # For optimization
	
	function simulate_minimize_penalty(system)
	    """
	    Minimizes the penalty energy (prevents overlaps and crossings)
	    
	    system: System containing state and properties
	    
	    Returns: Optimized positions
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
	    free_mask = ones(Bool, system.N)
	    free_mask[fixed_nodes] .= false
	    
	    # Create flattened representation
	    x0 = vec(R_init')  # Flatten the array column-major
	    
	    # Function to calculate energy from flattened positions
	    function energy_fn(x)
	        # Reshape to position matrix, preserving fixed nodes
	        positions = copy(R_init)
	        idx = 1
	        for i in 1:system.N
	            if free_mask[i]
	                positions[i, 1] = x[idx]
	                positions[i, 2] = x[idx+1]
	                idx += 2
	            end
	        end
	        
	        # Calculate energy
	        return penalty_energy(system, positions)
	    end
	    
	    # Extract only free parameters
	    free_params = Float64[]
	    for i in 1:system.N
	        if free_mask[i]
	            push!(free_params, R_init[i, 1])
	            push!(free_params, R_init[i, 2])
	        end
	    end
	    
	    # Optimize
	    result = optimize(energy_fn, free_params, LBFGS(), 
	                     Optim.Options(iterations=system.steps))
	    
	    # Reconstruct optimized positions
	    R_final = copy(R_init)
	    x_opt = Optim.minimizer(result)
	    idx = 1
	    for i in 1:system.N
	        if free_mask[i]
	            R_final[i, 1] = x_opt[idx]
	            R_final[i, 2] = x_opt[idx+1]
	            idx += 2
	        end
	    end
	    
	    return R_init, R_final
	end
	
	function simulate_auxetic(system)
	    """
	    Simulates the auxetic compression process
	    
	    system: System containing network properties
	    
	    Returns: Poisson ratio, initial and final positions
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
	    
	    # Create a list to store positions at each step
	    position_log = [copy(R_init)]
	    
	    # Copy the initial positions
	    R_current = copy(R_init)
	    
	    # For each compression step
	    for i in 1:num_iterations
	        # Apply perturbation to left edge
	        for idx in left_indices
	            R_current[idx, 1] += system.delta_perturbation
	        end
	        
	        # Create mask for fixed nodes (left and right edges)
	        fixed_nodes = vcat(left_indices, right_indices)
	        free_mask = ones(Bool, system.N)
	        free_mask[fixed_nodes] .= false
	        
	        # Minimize energy with fixed left and right edges
	        free_params = Float64[]
	        for i in 1:system.N
	            if free_mask[i]
	                push!(free_params, R_current[i, 1])
	                push!(free_params, R_current[i, 2])
	            end
	        end
	        
	        # Function to calculate total energy from flattened positions
	        function energy_fn(x)
	            # Reshape to position matrix, preserving fixed nodes
	            positions = copy(R_current)
	            idx = 1
	            for i in 1:system.N
	                if free_mask[i]
	                    positions[i, 1] = x[idx]
	                    positions[i, 2] = x[idx+1]
	                    idx += 2
	                end
	            end
	            
	            # Calculate total energy
	            return total_energy(system, positions)
	        end
	        
	        # Optimize
	        result = optimize(energy_fn, free_params, LBFGS(), 
	                         Optim.Options(iterations=system.steps))
	        
	        # Reconstruct optimized positions
	        x_opt = Optim.minimizer(result)
	        idx = 1
	        for i in 1:system.N
	            if free_mask[i]
	                R_current[i, 1] = x_opt[idx]
	                R_current[i, 2] = x_opt[idx+1]
	                idx += 2
	            end
	        end
	        
	        # Store the current positions
	        push!(position_log, copy(R_current))
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
	    
	    return poisson, R_init, R_final, position_log
	end
end

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
    system.delta_perturbation = 0.5
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
