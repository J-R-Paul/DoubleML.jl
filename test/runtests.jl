using DoubleML
using Test

@testset "DoubleML.jl" begin
    theta_est_naive, theta_est_crossfit = MC_sim(10, 10);
    println("The estimated treatment effect by naive DML is ", mean(theta_est_naive))
    # @test 
end
