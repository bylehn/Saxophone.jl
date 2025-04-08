# System type definition
mutable struct System
    # Basic parameters
    nr_points::Int
    random_seed::Int
    r_circle::Float64
    dx::Float64
    k_angle::Float64
    
    # Node and bond properties
    soft_sphere_sigma::Float64
    soft_sphere_epsilon::Float64
    
    # Crossing penalty attributes
    crossing_penalty_threshold::Float64
    crossing_penalty_strength::Float64
    
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
    k_angles::Union{Vector{Float64}, Nothing}
    
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
            0.3, 2.0,
            
            # Crossing penalty attributes
            0.3, 2.0,
            
            # Scaling parameters
            1e-5, 1.0, 2.0,
            
            # System state (all initialized to nothing)
            nothing, nothing, nothing, nothing, nothing, 
            nothing, nothing, nothing, nothing, nothing, 
            nothing, nothing, nothing, nothing,
            
            # Acoustic properties
            nothing, nothing, nothing, 0.01, 0.05, nothing, nothing,
            
            # Auxetic properties
            1.0, 0.1, 50, 1
        )
    end
end