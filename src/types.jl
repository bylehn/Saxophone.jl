mutable struct System
    # Basic parameters
    nr_points::Int
    random_seed::Int
    r_circle::Float64
    dx::Float64
    k_angle::Float64
    
    # Node and bond properties
    soft_sphere_sigma::Float64  # Diameter
    soft_sphere_epsilon::Float64
    
    # Crossing penalty attributes
    crossing_penalty_threshold::Float64  # radians
    crossing_penalty_strength::Float64   # epsilon of soft angle strength
    
    # Scaling parameters
    penalty_scale::Float64
    k_std_threshold::Float64
    k_std_strength::Float64
    
    # System state
    N::Union{Int, Nothing}
    G::Union{SimpleGraph, Nothing}
    X::Union{Matrix{Float64}, Nothing}
    E::Union{Matrix{Int}, Nothing}
    L::Union{Vector{Float64}, Nothing}
    surface_nodes::Union{Dict{String, Vector{Int}}, Nothing}
    surface_mask::Union{BitVector, Nothing}
    surface_bond_mask::Union{BitVector, Nothing}
    mass::Union{Matrix{Float64}, Nothing}
    spring_constants::Union{Vector{Float64}, Nothing}
    distances::Union{Vector{Float64}, Nothing}
    angle_triplets::Union{Matrix{Int}, Nothing}
    initial_angles::Union{Vector{Float64}, Nothing}
    
    # Acoustic properties
    m::Union{Vector{Float64}, Nothing}
    frequency_center::Union{Float64, Nothing}
    frequency_width::Union{Float64, Nothing}
    ageing_rate::Float64
    success_fraction::Float64
    nr_trials::Union{Int, Nothing}
    degrees::Union{Vector{Int}, Nothing}
    
    # Auxetic properties
    perturbation::Float64
    delta_perturbation::Float64
    steps::Int
    write_every::Int
    
    # Constructor with defaults
    function System(nr_points::Int, k_angle::Float64, random_seed::Int, r_circle::Float64, dx::Float64)
        return new(
            # Basic parameters
            nr_points, random_seed, r_circle, dx, k_angle,
            
            # Node and bond properties
            0.3, 2.0,  # soft_sphere_sigma, soft_sphere_epsilon
            
            # Crossing penalty attributes
            0.3, 2.0,  # crossing_penalty_threshold, crossing_penalty_strength
            
            # Scaling parameters
            1e-5, 1.0, 2.0,  # penalty_scale, k_std_threshold, k_std_strength
            
            # System state (all initialized to nothing)
            nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing, nothing, nothing,
            
            # Acoustic properties
            nothing, nothing, nothing, 0.01, 0.05, nothing, nothing,
            
            # Auxetic properties
            1.0, 0.1, 50, 1
        )
    end
end

# In utils.jl
function create_delaunay_graph!(system::System)
    # Set the random seed for reproducibility
    Random.seed!(system.random_seed)
    
    # Generate grid points
    nr_points = system.nr_points
    X = zeros(Float64, nr_points^2, 2)
    idx = 1
    for y in 1:nr_points
        for x in 1:nr_points
            X[idx, :] = [x, y]
            idx += 1
        end
    end
    
    # Determine surface nodes
    surface_mask = falses(size(X, 1))
    for i in 1:size(X, 1)
        if X[i, 1] == 1 || X[i, 1] == nr_points || 
           X[i, 2] == 1 || X[i, 2] == nr_points
            surface_mask[i] = true
        end
    end
    
    # Add noise to non-surface points
    for i in 1:size(X, 1)
        if !surface_mask[i]
            X[i, :] += system.dx * (2 * rand(2) .- 1)
        end
    end
    
    # Create Delaunay triangulation
    points = [(X[i,1], X[i,2]) for i in 1:size(X,1)]
    tri = DelaunayTriangulation.triangulate(points)
    
    # Extract edges from triangulation
    edges = Set{Tuple{Int,Int}}()
    for triangle in get_triangles(tri)
        i, j, k = triangle
        # Ensure we're using 1-based indices
        push!(edges, minmax(i, j))
        push!(edges, minmax(j, k))
        push!(edges, minmax(i, k))
    end
    
    # Filter edges by distance
    filtered_edges = Tuple{Int,Int}[]
    for (i, j) in edges
        # Ensure indices are valid
        if i > 0 && j > 0 && i <= size(X, 1) && j <= size(X, 1)
            d = norm(X[i,:] - X[j,:])
            if d < system.r_circle
                push!(filtered_edges, (i, j))
            end
        end
    end
    
    # Create graph from edges
    N = size(X, 1)
    G = SimpleGraph(N)
    E = zeros(Int, length(filtered_edges), 2)
    L = zeros(Float64, length(filtered_edges))
    
    for (idx, (i, j)) in enumerate(filtered_edges)
        add_edge!(G, i, j)
        E[idx, :] = [i, j]
        L[idx] = norm(X[i,:] - X[j,:])
    end
    
    # Store results in system
    system.N = N
    system.G = G
    system.X = X
    system.E = E
    system.L = L
    system.surface_mask = surface_mask
    
    # Get additional properties
    get_surface_nodes!(system)
    extract_surface_bond_mask!(system)
    get_mass!(system)
    system.degrees = [degree(G, v) for v in 1:N]
    
    return system
end

function get_surface_nodes!(system::System)
    if isnothing(system.G)
        error("Graph not created. Call create_delaunay_graph! first.")
    end
    
    nodes = collect(1:system.N)
    x_values = [Int(mod(n-1, system.nr_points))+1 for n in nodes]
    y_values = [Int(floor((n-1) / system.nr_points))+1 for n in nodes]
    
    top_nodes = nodes[y_values .== system.nr_points]
    bottom_nodes = nodes[y_values .== 1]
    left_nodes = nodes[x_values .== 1]
    right_nodes = nodes[x_values .== system.nr_points]
    
    system.surface_nodes = Dict(
        "top" => top_nodes,
        "bottom" => bottom_nodes,
        "left" => left_nodes,
        "right" => right_nodes
    )
    
    return system
end

function extract_surface_bond_mask!(system::System)
    if isnothing(system.surface_nodes)
        get_surface_nodes!(system)
    end
    
    # Combine all surface nodes into a single set
    surface_nodes_set = Set{Int}()
    for (key, nodes) in system.surface_nodes
        union!(surface_nodes_set, nodes)
    end
    
    # Create a boolean mask for edges between surface nodes
    system.surface_bond_mask = BitVector([
        (edge[1] in surface_nodes_set) && (edge[2] in surface_nodes_set)
        for edge in eachrow(system.E)
    ])
    
    return system
end

function get_mass!(system::System)
    if isnothing(system.G) || isnothing(system.N)
        error("Graph not created. Call create_delaunay_graph! first.")
    end
    
    m = ones(system.N)
    
    # Create the mass matrix
    m2 = zeros(2 * system.N)
    m2[1:2:end] = m
    m2[2:2:end] = m
    system.mass = diagm(m2)
    
    system.m = m
    
    return system
end

function create_spring_constants!(system::System, k_1::Float64=1.0)
    if isnothing(system.X) || isnothing(system.E)
        error("Graph properties not set. Call create_delaunay_graph! first.")
    end
    
    displacements = zeros(Float64, size(system.E, 1), 2)
    for i in 1:size(system.E, 1)
        displacements[i, :] = system.X[system.E[i, 1], :] - system.X[system.E[i, 2], :]
    end
    
    distances = [norm(displacements[i, :]) for i in 1:size(displacements, 1)]
    system.spring_constants = k_1 ./ distances
    system.distances = distances
    
    return system
end

# In utils.jl
function calculate_angle_triplets(E::Matrix{Int})
    """
    Calculates the triplets of nodes that form angles.
    This considers ALL angles at a node, not just the planar subset.
    
    Returns: triplets of nodes that form angles
    """
    edges = [(E[i,1], E[i,2]) for i in 1:size(E,1)]
    triplets = Tuple{Int,Int,Int}[]
    
    for i in 1:length(edges)
        for j in i+1:length(edges)
            # Check if edges share a vertex
            if edges[i][1] == edges[j][1]
                push!(triplets, (edges[i][2], edges[i][1], edges[j][2]))
            elseif edges[i][1] == edges[j][2]
                push!(triplets, (edges[i][2], edges[i][1], edges[j][1]))
            elseif edges[i][2] == edges[j][1]
                push!(triplets, (edges[i][1], edges[i][2], edges[j][2]))
            elseif edges[i][2] == edges[j][2]
                push!(triplets, (edges[i][1], edges[i][2], edges[j][1]))
            end
        end
    end
    
    # Convert to matrix format
    angle_triplets = zeros(Int, length(triplets), 3)
    for (idx, (i, j, k)) in enumerate(triplets)
        angle_triplets[idx, :] = [i, j, k]
    end
    
    return angle_triplets
end

function compute_angle_between_triplet(pi::Vector{Float64}, pj::Vector{Float64}, pk::Vector{Float64})
    """
    Computes the angle formed by three points.
    
    Returns: Angle in radians
    """
    d_ij = pi - pj
    d_kj = pk - pj
    u_ij = d_ij / norm(d_ij)
    u_kj = d_kj / norm(d_kj)
    cos_theta = dot(u_ij, u_kj)
    return acos(clamp(cos_theta, -0.999, 0.999))
end

function calculate_initial_angles(positions::Matrix{Float64}, angle_triplets::Matrix{Int})
    """
    Calculates the initial angles for each triplet of nodes.
    
    Returns: Vector of initial angles
    """
    n_triplets = size(angle_triplets, 1)
    angles = zeros(n_triplets)
    
    for t in 1:n_triplets
        i, j, k = angle_triplets[t, :]
        pi = positions[i, :]
        pj = positions[j, :]
        pk = positions[k, :]
        angles[t] = compute_angle_between_triplet(pi, pj, pk)
    end
    
    return angles
end

function calculate_angle_triplets!(system::System)
    """
    Calculates the triplets of nodes that form angles and stores them in system.
    """
    system.angle_triplets = calculate_angle_triplets(system.E)
    return system
end

function calculate_initial_angles!(system::System)
    """
    Calculates the initial angles for each triplet of nodes and stores them.
    """
    if isnothing(system.angle_triplets)
        calculate_angle_triplets!(system)
    end
    
    system.initial_angles = calculate_initial_angles(system.X, system.angle_triplets)
    return system
end

function poisson_ratio(initial_horizontal::Float64, initial_vertical::Float64, 
                      final_horizontal::Float64, final_vertical::Float64)
    """
    Calculate the Poisson ratio based on average edge positions.
    
    Returns: Poisson ratio
    """
    delta_horizontal = final_horizontal - initial_horizontal
    delta_vertical = final_vertical - initial_vertical
    
    return -delta_vertical / delta_horizontal
end

function initialize!(system::System)
    """
    Initializes the system by setting up the graph, calculating necessary properties,
    and preparing the system for simulation.
    """
    create_delaunay_graph!(system)
    create_spring_constants!(system)
    calculate_angle_triplets!(system)
    calculate_initial_angles!(system)
    
    # Calculate k_angles - just ensure triplet_centers indices are valid
    if !isnothing(system.angle_triplets) && !isnothing(system.degrees)
        triplet_centers = system.angle_triplets[:,2]
        degrees = system.degrees
        
        # Create k_angles array
        n_triplets = size(system.angle_triplets, 1)
        system.k_angles = zeros(Float64, n_triplets)
        
        for i in 1:n_triplets
            center = triplet_centers[i]
            if degrees[center] > 1  # Avoid division by zero
                system.k_angles[i] = 2*system.k_angle / (degrees[center] * (degrees[center] - 1))
            else
                system.k_angles[i] = system.k_angle  # Default if degree is 1
            end
        end
    end
    
    return system
end