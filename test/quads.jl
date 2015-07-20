using NewTR
using Base.Test

println("Testing Quadratic bounded problems")

N = 100000
a = -1
b = 2

total = 0
success = 0

for i = 1:N
  n = 3

  c = rand(n)*(b-a)+a
  α = rand(n)*100 + 0.1
  l = rand(n)*(b-a)+a
  u = l + rand(n)

  f(x) = 0.5*dot(α,(x-c).^2)
  ∇f(x) = diagm(α)*(x-c)
  ∇²f(x) = diagm(α)

  x0 = Float64[rand(linspace(l[i],u[i],100)) for i = 1:n]

  P(x) = Float64[x[i] < l[i] ? l[i] : (x[i] > u[i] ? u[i] : x[i]) for i=1:n]

  x, fx, ∇fx, k, ef = NewTR.solve(f, ∇f, ∇²f, P, x0)
  sol = P(c)

  if ef != 0
    file = open("test/error_data.jl", "w")
    write(file, "#ERROR data\n")
    write(file, "n = $n\n")
    write(file, "c = $c\n")
    write(file, "α = $α\n")
    write(file, "l = $l\n")
    write(file, "u = $u\n")
    write(file, "x0 = $x0\n")
    write(file, "#x = $x\n")
    write(file, "#fx = $fx\n")
    write(file, "#|P[x-∇fx]-x| = $(norm(P(x-∇fx)-x, Inf))\n")
    close(file)
  end
  total += 1
  if ef == 0
    @test norm(P(x-∇fx)-x, Inf) < 1e-5
    success += 1
  else
    @test k == 10000
  end
end

@test success/total > 0.95
