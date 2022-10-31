function gen_plm(n, p)
    θ = 0.8 # Treatment effect
    d = rand(n) # Treatment variable

    X = randn(n, p)
    β = repeat([0.1, .9], inner = p ÷ 2)

    # Generate treatment
    m(x) = 1/(2*π) * (sinh(1) / (cosh(1)-cos(x)))
    d = m.(X * β) .+ randn(n)

    # Generate outcome variable
    y = [θ] .* d .+ sin.(X * β) .+ randn(n)


    return X, d, y, β, θ
end

function gen_plm(n, p, seed)
    Random.seed!(seed)
    gen_data(n, p)
end