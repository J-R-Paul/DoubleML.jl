using DoubleML
using Test

@testset "DoubleML.jl" begin
    theta_est_naive, theta_est_crossfit = MC_sim(1, 10);
    println(theta_est_naive[1])
    @test theta_est_naive[1] == 0.3757114922230331
end