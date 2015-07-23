using NewTR

f(x) = (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2
∇f(x) = [2*(x[1]-1) - 400*x[1]*(x[2]-x[1]^2);200*(x[2]-x[1]^2)]
∇²f(x) = [2-400*(x[2]-x[1]^2)+800*x[1]^2 -400*x[1];-400*x[1] 200]

x0 = [-1.2;1.0]

x, fx, ∇fx, k, ef, el_time = NewTR.solve(f, ∇f, ∇²f, x0)

println("x = $x")
println("fx = $fx")
println("|∇fx| = $(norm(∇fx))")
println("Iterations: $k")
println("EXIT: $(NewTR.flags[ef])")
println("Elapsed time: $el_time")
