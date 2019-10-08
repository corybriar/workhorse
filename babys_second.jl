using Pkg
using LinearAlgebra
using Plots
using Distributions
using Parameters, Random



"""
BABY'S FIRST SPATIAL GEN-EQ SOLVER
------------------------------------
For a given set of parameters, this program estimates a dynamic spatial equilibrium for n agents over T < ∞ periods using Lee (2008): "An Estimable Dynamic General Equilibrium Model of Work, Schooling and Occupation Choice."
"""

@with_kw struct parameters
    σ1::Float64 = 0.5   # Weight on log(wage)
    σ2::Float64 = -0.5  # Weight on -log(prices)
    ρ::Float64 = 0.04   # Inverse elasticity of rent
    ω::Float64 = -0.03  # Inverse elasticity of wage
    T::Int64 = 2       # Number of time periods
    n::Int64 = 5000     # Number of agents
    J::Int64 = 2       # Number of cities
    τ::Float64 = 5      # Utility cost of moving
    δ::Float64 = 0.9    # Expontential discount rate
    β::Float64 = 1      # Present bias
end

para = parameters()

"""
eq_solver(): General equilibrium function
    - Takes parameters, the initial distribution of agents, an initial guess at the prices, the technology multipliers for inverse demand for housing and labor, the amenity levels for each city A_jt in a given period, and the matrix of agent shocks e_jt
    - Contains the functions:
        + J_sum() -- Computes EMAX value for a time t
        + EMAXer() -- Calculates EMAX value for all cities for all periods given a guess of prices
        + choice_t() -- Simulates agent decisions given EMAX values
        + initial_sort() -- Simulates agent decisions for first period
        + prices() -- Calculates rents and wages resulting from agent choices for a given set of guessed prices
    - Structure:
        1) First declare functions for later use
        2) Nested while loop structure to compute equilibrium prices
            + Inner loops that establish a fixed point in each year given a current guess of prices and the resulting EMAX values for each city
            + Outer loop that checks whether behavior induced by expectations produces prices sufficiently close to the guess and updates expectations accordingly
"""



function eq_solver(para, guess, R_j, W_j, A_jt, e_jt, max_iter, inner_maxit)
    @unpack σ1, σ2, ρ, ω, T, n, J, τ, δ, β = para

    """
    J_sum(): Function to calculate the time t EMAX value for each city j (hence "J-sum").
        - Takes a given time t, the parameters, the existing EMAX matrix, city amenity levels A_jt, and the most recent guess of prices, prices_new
        - Structure:
            1) Creates a blank vector EMAX_t to store results
            2) Loops through each city j:
                a) Creates matricies prices_noj and A_noj that omit city j and pulls out these values for city j
                b) Calculates j's contribution to EMAX_t, emax_j
                c) Calculates each non-j city's contribution to EMAX_t and sums them up, emax_noj
                d) Adds emax_j and emax_noj and takes the log
            3) Returns EMAX_t
    """
    function J_sum(t, para, EMAX, A_jt, prices_new)
        @unpack σ1, σ2, ρ, ω, T, n, J, τ, δ, β = para
        # Blank matrix to store EMAX_t's
        EMAX_t = zeros(1,J)
        # Iterate by j, calculate entries in EMAX[j]
        for j in 1:J
            # Initialize matricies to delete obs. from
            prices_noj = prices_new[t,:,:]
            A_noj = A_jt[t,:]

            # Items to be filtered out:
            prices_j = prices_new[t,:,j]
            Ajt = A_jt[t,j]

            # Create filtered matricies
            prices_noj = prices_noj[:,1:size(prices_noj,2) .!= j]
            A_noj = A_noj[1:size(A_noj,1) .!= j]

            #Caculate EMAX_t for city j
            emax_j = exp(σ1*prices_j[1,1] + σ2*prices_j[2,1] + Ajt + δ*EMAX[t+1,j])
            EMAX_noj = zeros(1,J-1)
            for j2 in 1:(J-1)
                EMAX_noj[1,j2] += exp(σ1*prices_noj[1,j2] + σ2*prices_noj[2,j2] + A_noj[j2] + δ*EMAX[t+1, j2] - τ)
            end
            emax_noj = sum(EMAX_noj)
            EMAX_t[j] = log(emax_j + emax_noj)

        end # j-loop

        return EMAX_t
    end # J_sum

    """
    EMAXer(): Calculates EMAX value for all cities j and all periods t
        - Takes the parameters, the city amenity levels A_jt, and the most recent guess of prices, prices_new
        - Only fucking called EMAXer because EMAX was impossible to delete from memory while I was writing it
        - Structure:
            1) Creates blank matrix EMAX to store results
            2) Initializes matrix for time T using J_sum
            3) Loops backwards from time T-1 to 1 calculating EMAX's for each city j using J_sum
            4) Returns EMAX
    """
    function EMAXer(para, A_jt, prices_new)
        @unpack σ1, σ2, ρ, ω, T, n, J, τ, δ, β = para

        # Initialize EMAX matrix
        EMAX = zeros(T+1,J)
        EMAX[T,:] = J_sum(T, para, EMAX, A_jt, prices_new)
        # Loop by t to obtain EMAX_j for every time period
        for t in (T-1):-1:1
            EMAX[t,:] = J_sum(t, para, EMAX, A_jt, prices_new)
        end # t-loop
        # Delete T+1th period
        return EMAX
    end # EMAXer

    """
    choice_t() -- Calculates utility maximizing locational choice for each agent at time t
        - Takes parameters, the time period in question, amenity levels, EV1 shocks, history of agent decisions and EMAX values
        - Structure:
            1) Establishes blank matricies to hold choices
            2) Iterates by i then j to determine present period going-forward utility value of each location to i
                + First establishes agent i's location
                + Compares this location to each other location j.
                + If loc_i == j, then the vector of utilties is calculate accordingly
                + Finds the maximum entry of the vector of going forward utilities, records as the ith entry in choice
            3) Returns the vector of utility maximizing locations for all agents in period t
    """

    function choice_t(para, t, prices_new, A_jt, e_jt, agent_choices, EMAX)
        @unpack σ1, σ2, n, J, τ, δ, β = para
        # Create blank matricies to hold agent choices
        choice = zeros(n)
        for i in 1:n
            loc_i = agent_choices[t-1,i]
            # Loop by j to determine agent location, define utility values at different locations to determine utility maximizing action
            u_ijt = zeros(J)
            for j in 1:J
                if loc_i == j
                    # Create vector of time t utility values for each location
                    u_ijt = σ1.*prices_new[1,:] + σ2.*prices_new[2,:] + A_jt[t,:] + e_jt[t,i,:] + β*δ.*EMAX[t+1,:] .- τ

                    # Replace the jth element of u_ijt with the value of staying in location j rather than moving
                    u_ijt[j] = σ1*prices_new[1,j] + σ2*prices_new[2,j] + A_jt[t,j] + e_jt[t,i,j] + β*δ*EMAX[t+1,j]
                end # End if loc_i == j statment
            end #j-loop
            # Find agent i's new location
            choice[i] = findmax(u_ijt)[2]
        end # End i-loop
        return choice
    end # End choices_t()

    function initial_sort(para, prices_new, A_jt, e_jt, EMAX)
        @unpack σ1, σ2, n, δ, β = para
        # Create blank matricies to hold agent choices
        choice = zeros(n)
        for i in 1:n
            # Compute time 1 utility in each location for agent i
            u_ijt = σ1.*prices_new[1,:] + σ2.*prices_new[2,:] + A_jt[1,:] + e_jt[1,i,:] + β*δ.*EMAX[2,:]
            # Find agent i's initial location
            choice[i] = findmax(u_ijt)[2]
        end # End i-loop
        return choice
    end # End choices_t()

    """
    prices(): Function to calculate prices across locations after agent choices have been made
    """
    function prices(t, pop_jt, W_j, R_j, para)
        @unpack ρ, ω, T, J = para
        wages = W_j.*(pop_jt[t,:] .+ 1).^ω # Inverse labor supply
        rents = R_j.*(pop_jt[t,:] .+ 1).^ρ # Inverse housing supply
        prices_calc = zeros(2,J)
        prices_calc[1,:] = wages
        prices_calc[2,:] = rents
        replace!(prices_calc, Inf => 0)
        return prices_calc
    end

"""------------ BEGIN CONVERGENCE LOOP ------------"""
    # Initialize while loop
    tolerance = 10e-6
    iter = 0
    prices_current = guess
    differ = 10e+6
    prices_act = zeros(T,2,J)
    pop_jt = zeros(T,J)
    EMAX_track = zeros(T+1,J,max_iter)
    eq_prices = zeros(T,2,J,max_iter + 3)
    pop_track = zeros(T,J,max_iter)

    # Outer While loop to acheive convergence
    @time while (differ > tolerance) & (iter < max_iter)
        # Add to iteration tracker
        iter += 1
        # Blank matrix to hold agent choices
        agent_choices = zeros(T,n)
        # Form EMAX values given current guess of prices in outer loop
        EMAX = EMAXer(para, A_jt, prices_current)
        EMAX_track[:,:,iter] = EMAX
        # Initialize while loop for first period
        innerdiff = 10e+6
        init = 0
        innertol = 10e-6
        # Blank matrix to store innerloop progress
        prices_temp = zeros(2, J, inner_maxit + 3)
        prices_temp[:,:,3] = prices_current[1,:,:]
        #"""---------- INNER LOOP: FIRST PERIOD -----------"""
        while (innerdiff > innertol) & (init < inner_maxit)
            init += 1
            # Simulate agent actions
            agent_choices[1,:] = initial_sort(para, prices_temp[:,:,init+2], A_jt, e_jt, EMAX)
            # Calculate resulting prices
            for j in 1:J
                # Count population for each j in time t
                pop_jt[1,j] = count(i -> i == j, agent_choices[1,:])
            end # j-loop

            # Store resulting wages and rents
            prices_act[1,:,:] = prices(1,pop_jt, W_j, R_j, para)

            # Calculate difference
            innerdiff = findmax(abs.(prices_act[1,:,:] .- prices_temp[:,:,init+2]))[1]

            # Check if fixed point has been found
            if innerdiff > innertol
                prices_temp[:, :, init+3] = 0.25*(prices_act[1,:,:] + prices_temp[:,:, init+2] + prices_temp[:,:, init+1] + prices_temp[:,:, init])
                println(innerdiff)
            end # Convergence check if statement
        end # End initial period while loop


        """---------- INNER LOOP: PERIODS 2 THROUGH T -----------"""
        # Loop through time periods, achieve fixed point in each
        for t in 2:T
            # Initiate while/for loop for all other periods
            innerdiff = 10e+6
            init = 0
            # Blank matricies to hold progress
            prices_temp = zeros(2, J, inner_maxit + 3)
            prices_temp[:,:,3] = prices_current[t,:,:]
            while (innerdiff > innertol) & (init < inner_maxit)
                init += 1
                # Simulate agent actions
                agent_choices[t,:] = choice_t(para, t, prices_temp[:,:,init + 2], A_jt, e_jt, agent_choices, EMAX)

                # Calculate resulting prices
                for j in 1:J
                    # Count population for each j in time t
                    pop_jt[t,j] = count(i -> i ==j, agent_choices[t,:])
                end # j-loop
                # Store resulting wages and rents
                prices_act[t,:,:] = prices(t, pop_jt, W_j, R_j, para)

                # Calculate difference
                innerdiff = findmax(abs.(prices_act[t,:,:] - prices_temp[:,:,init+2]))[1]
                # Check if fixed point has been found
                if innerdiff > innertol
                    prices_temp[:, :, init+3] = 0.25*(prices_act[t,:,:] + prices_temp[:,:,init+2] + prices_temp[:,:,init+1] + prices_temp[:,:,init])
                    println(innerdiff)
                end # Convergence check if statement
            end # End period-t while loop
        end # t-loop

        # Store results for iteration
        pop_track[:,:,iter] = pop_jt
        eq_prices[:,:,:,iter+3] = prices_act

        # Calculate differ
        differ = findmax(abs.(prices_act - prices_current))[1]
        # Check if convergence criteria are met
        if differ < tolerance
            println("Convergence acheived after $iter iterations. Diff = $differ, Tolerance = $tolerance")
        else
            println("Iteration = $iter, Diff = $differ")
            prices_current = 0.25*(prices_act + prices_current + eq_prices[:,:,:,iter+1] + eq_prices[:,:,:,iter])
        end # End convergence check if statement
    end # while
"""------------- END CONVERGENCE LOOP -------------"""
    return eq_prices, pop_track, EMAX_track
end # gen_eq

"""
RUNNING THE MODEL
The next few lines of code generate the matrix of idiosyncratic preference shocks for agents, the technology mulipliers for labor and housing, the amenity levels for each city (as constants or a function), and the initial guess for the prices in each city and period. These are taken out of eq_solver to allow the user some control over these values if desired. Note, however, that this code is all written with the assumption that preference shocks are Extreme Value Type I, and solutions will only be valid if a distribution of this form is used.
"""
# Generate matrix of agent shocks. (Thanks John)
function AgentShockmatrix(para)
    @unpack n, T, J = para
    dist = GeneralizedExtremeValue(0,1,0)
    A = rand(dist, T, n, J)
    return A
end # AgentShockmatrix

# Draw technology levels using uniform distribution
function tech_levels(para,Wmax,Rmax)
    @unpack J = para
    dist_w = Uniform(1, Wmax)
    dist_r = Uniform(1, Rmax)
    W = rand(dist_w, J)
    R = rand(dist_r, J)
    return W, R
end # tech_levels

# Draw amenity levels
function amenities(para, ϕ = 1, shock_var = 1, type = "constant")
    @unpack J, T = para
    dist_a = Normal(0,1.5)
    A_1 = rand(dist_a, 1, J)
    # Populate amenities matrix based on A_1
    A_jt = zeros(T,J)
    A_jt[1,:] = A_1
    for t in 2:T
        """Optional functional forms here"""
        if type == "trend"
            A_jt[t,:] = ϕ.*A_jt[t-1,:]
        elseif type == "persitence"
            dist_ξ = Normal(0,shock_var)
            ξ_jt = rand(dist_ξ, 1, J)
            A_jt[t,:] = ϕ.*A_jt[t-1] + ξ_jt
        else
            A_jt[t,:] = A_jt[t-1,:]
        end
    end # t-loop
    return A_jt
end # amenities

# Function for generating random initial guess
function guesser(para)
    @unpack J, T = para
    guess = ones(T,2,J)
    return guess
end #guesser

"""------------- MODEL SETUP ---------------"""
e_jt = AgentShockmatrix(para)
#W_j, R_j = tech_levels(para,10,10)
#A_jt = amenities(para)
W_j = ones(2)
R_j = 2*ones(2)
A_jt = zeros(2,2)
guess = guesser(para)


eq_prices, pop_track, EMAX_track = eq_solver(para, guess, R_j, W_j, A_jt, e_jt, 100, 200)
"""-----------------------------------------"""
