# [===============================================================================]

mutable struct DoubleML #<: MLJModelInterface.Unsupervised
    model::MLJModelInterface.Model  # Model to be used for the first stage
    model2::MLJModelInterface.Model # Model to be used for the second stage
    data::DataFrame                 # Data
    n_folds::Int                    # Number of folds for cross-validation
    n_rep::Int                      # Number of repetitions for cross-validation
    n_jobs::Int                     # Number of jobs for parallelization
    verbose::Bool                   # Whether to print the progress
end

DoubleML(model, model2, data) = DoubleML(model, model2, data, 5, 1, 1, false)



# [===============================================================================]
function naive_DML(X, d, y, model, model2)
    # Fit the first stage model
    mach = machine(model, X, d, scitype_check_level=0)
    fit!(mach)
    d_hat = predict(mach, X)

    # Fit the second stage model
    mach2 = machine(model2, X, y, scitype_check_level=0)
    fit!(mach2)
    y_hat = predict(mach2, X)

    # residuals
    v_hat = d .- d_hat # Error in the first stage
    e_hat = y .- y_hat # Error in the second stage

    #dot(v_hat, e_hat)/ dot(v_hat, v_hat)
    theta_est = dot(v_hat, e_hat) / dot(v_hat, d) 



    return theta_est
end


# [===============================================================================]
# Cross fitting DML
function crossfit_DML(X, d, y, model, model2, n_folds, n_rep)
    # Split the data into n_folds
    n = size(X, 1)
    n_fold = Int(floor(n/n_folds)) # Number of observations in each fold
    n_rem = n - n_fold * n_folds # Number of observations in the last fold

    # Randomly permute the data
    idx = randperm(n)

    # Split the data into n_folds
    idx_split = [idx[1:n_fold]]
    for i in 2:n_folds
        push!(idx_split, idx[(i-1)*n_fold+1:i*n_fold])
    end
    if n_rem > 0
        push!(idx_split, idx[end-n_rem+1:end])
    end

    # Repeat the cross-fitting n_rep times
    theta_est = zeros(n_rep)
    for i in 1:n_rep
        theta_est[i] = 0
        for j in 1:n_folds
            # Split the data into training and test sets
            idx_train = vcat(idx_split[1:j-1]..., idx_split[j+1:end]...)
            idx_test = idx_split[j]

            # Fit the first stage model
            mach = machine(model, X[idx_train, :], d[idx_train], scitype_check_level=0)
            fit!(mach)
            d_hat = predict(mach, X[idx_test, :])

            # Fit the second stage model
            mach2 = machine(model2, X[idx_train, :], y[idx_train], scitype_check_level=0)
            fit!(mach2)
            y_hat = predict(mach2, X[idx_test, :])

            # residuals
            v_hat = d[idx_test] .- d_hat # Error in the first stage
            e_hat = y[idx_test] .- y_hat # Error in the second stage

            #dot(v_hat, e_hat)/ dot(v_hat, v_hat)
            theta_est[i] += dot(v_hat, e_hat) / dot(v_hat, d[idx_test]) 
        end
        theta_est[i] /= n_folds
    end

    return theta_est
end





# [===============================================================================]
# Partially Linear IV (PLIV)

# [===============================================================================]
# Monte-Carlo Simulation
function MC_sim(n_reps)
    # Set the parameters
    theta = 0.8
    n_folds = 10


    # Set the models
    RF = @load RandomForestRegressor pkg = DecisionTree


    # Set the results
    theta_est_naive = zeros(n_reps)
    theta_est_crossfit = zeros(n_reps)

    # Run the simulation
    for i in 1:n_reps
        println("Iteration: ", i)
        # Generate the data
        X, d, y = gen_data(1000, 10)

        # Estimate the treatment effect
        model1_naive = RF()
        model2_naive = RF()
        theta_est_naive[i] = naive_DML(X, d, y, model1_naive, model2_naive)
        
        model1_cross = RF()
        model2_cross = RF()
        theta_est_crossfit[i] = crossfit_DML(X, d, y, model1_cross, model2_cross, n_folds, 1)[1]
    end

    # Print the results
    println("The true treatment effect is ", theta)
    println("The estimated treatment effect by naive DML is ", mean(theta_est_naive))
    println("The bias by naive DML is ", theta - mean(theta_est_naive))
    println("The estimated treatment effect by crossfit DML is ", mean(theta_est_crossfit))
    println("The bias by crossfit DML is ", theta - mean(theta_est_crossfit))

    Gadfly.with_theme(:dark) do 
        plot(
            layer(x = theta_est_naive, Geom.density,Theme(default_color = "blue")),
            layer(x = theta_est_crossfit, Geom.density, Theme(default_color = "red")),
            Guide.manual_color_key("Method", ["Naive DML", "Crossfit DML"], ["blue", "red"]),
        )
    end

    return theta_est_naive, theta_est_crossfit
end
function MC_sim(n_reps, seed)
    Random.seed!(seed)
    return MC_sim(n_reps)
end


# # Working with formulas
# formula = @formula(x3 ~ 1 + x1 + x2)

# f = apply_schema(formula, StatsModels.schema(formula, data))

# f

# y, X = modelcols(f, data)

