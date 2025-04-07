using Plots
function visualize_network(system::System)
    
    # Extract node positions
    X = system.X
    
    # Create plot with appropriate dimensions
    p = plot(
        aspect_ratio=:equal,
        xlim=(0, system.nr_points+1),
        ylim=(0, system.nr_points+1),
        legend=false,
        grid=false
    )
    
    # Plot edges
    for i in 1:size(system.E, 1)
        n1, n2 = system.E[i, :]
        x1, y1 = X[n1, :]
        x2, y2 = X[n2, :]
        
        # Line thickness proportional to spring constant
        thickness = 2 * system.spring_constants[i] * system.distances[i]
        plot!(p, [x1, x2], [y1, y2], color=:black, alpha=0.6, linewidth=thickness)
    end
    
    # Plot nodes
    scatter!(p, X[:,1], X[:,2], 
        color=:black, 
        alpha=0.25, 
        markersize=system.soft_sphere_sigma*10
    )
    
    # Highlight surface nodes in different colors
    if !isnothing(system.surface_nodes)
        for (key, nodes) in system.surface_nodes
            color = if key == "top"
                :red
            elseif key == "bottom"
                :blue
            elseif key == "left"
                :green
            else # right
                :purple
            end
            
            scatter!(p, X[nodes,1], X[nodes,2], 
                color=color, 
                alpha=0.5, 
                markersize=system.soft_sphere_sigma*15
            )
        end
    end
    
    return p
end