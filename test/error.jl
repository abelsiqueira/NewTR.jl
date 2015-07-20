using NewTR

include("error_data.jl")

f(x) = 0.5*dot(α,(x-c).^2)
∇f(x) = diagm(α)*(x-c)
∇²f(x) = diagm(α)

P(x) = Float64[x[i] < l[i] ? l[i] : (x[i] > u[i] ? u[i] : x[i]) for i = 1:n]

x, fx, ∇fx, k, ef = NewTR.solve(f, ∇f, ∇²f, P, x0, verbose=true)

if n < 5
  println("α = $α")
  println("c = $c")
  println("l = $l")
  println("x = $x")
  println("u = $u")
end
println("fx = $fx")
println("|∇fx| = $(norm(∇fx))")
println("Iterations: $k")
println("EXIT: $(NewTR.flags[ef])")
