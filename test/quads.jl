using NewTR
using Base.Test

println("Testing Quadratic bounded problems")

N = 100
a = -1
b = 2

for i = 1:N
  n = rand(10:100)

  c = rand(n)*(b-a)+a
  α = rand(n)*100 + 1
  l = rand(n)*(b-a)+a
  u = l + rand(n)

  f(x) = 0.5*dot(α,(x-c).^2)
  ∇f(x) = diagm(α)*(x-c)
  ∇²f(x) = diagm(α)

  x0 = Float64[rand(linspace(l[i],u[i],100)) for i = 1:n]

  P(x) = Float64[x[i] < l[i] ? l[i] : (x[i] > u[i] ? u[i] : x[i]) for i=1:n]

  x, fx, ∇fx, k, ef = NewTR.solve(f, ∇f, ∇²f, P, x0)
  sol = P(c)

  @test ef == 0
  @test norm(P(x-∇fx)-x, Inf) < 1e-5
end
