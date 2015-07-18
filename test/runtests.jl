using NewTR

f(x) = (x[1]-1)^2 + 100*(x[2]-x[1]^2)^2
∇f(x) = [2*(x[1]-1) - 400*x[1]*(x[2]-x[1]^2);200*(x[2]-x[1]^2)]
∇²f(x) = [2-400*(x[2]-x[1]^2)+800*x[1]^2 -400*x[1];-400*x[1] 200]

x0 = [-3.0;-4.0]

P(x) = x

x, ∇fx, k, ef = NewTR.solve(f, ∇f, ∇²f, P, x0)

println("x = $x")
println("|∇fx| = $(norm(∇fx))")
println("Iterations: $k")
println("EXIT: $(NewTR.flags[ef])")
