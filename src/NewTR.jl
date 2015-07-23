module NewTR

include("Steps.jl")

macro verbose(str)
  :(verbose && println($(str)))
end

using Compat

type Options
  ϵ::Real
  η₀::Real
  η₁::Real
  η₂::Real
  σ₁::Real
  σ₂::Real
  σ₃::Real
  kmax::Int
  Δmin::Real
  Δmax::Real
  verbose::Bool
  max_time::Real
  function Options(ϵ::Real = 1e-5, η₀::Real = 1e-3, η₁::Real = 0.25, η₂::Real =
      0.75, σ₁::Real = 0.25, σ₂::Real = 0.5, σ₃::Real = 4.0, kmax::Int = 10000,
      Δmin::Real = 1e-12, Δmax::Real = 1e20, verbose::Bool = false,
      max_time = 600)
    return new(ϵ, η₀, η₁, η₂, σ₁, σ₂, σ₃, kmax, Δmin, Δmax, verbose, max_time)
  end
end

@compat flags = Dict(
    0=> "Convergence criteria satisfied",
    1=> "Maximum number of iterations",
    2=> "Time limit exceeded")

function solve(f::Function, ∇f::Function, ∇²f::Function, x₀::Vector;
    η₁ = 0.25, η₂ = 0.75, μ = 1.0, α = 0.5, β = 0.5, σ₁ = 1/6, σ₂ = 6,
    kmax = 1000, max_time = 60)
  # Unconstrained problem.

  ef = 0
  st_time = time()

  x = copy(x₀)
  fx = f(x)
  ∇fx = ∇f(x)
  B = eye(length(x))
  ngrad = norm(∇fx)
  r = ngrad

  k = 0
  el_time = time() - st_time
  while ngrad > 1e-8 && el_time < max_time
    # Step 2
    d = more_sorensen(r, ∇fx, B)

    # Step 3
    ∇fx₀ = copy(∇fx)
    m = fx + dot(∇fx,d) + 0.5*dot(d,B*d)
    x⁺ = x + d
    fx⁺ = f(x⁺)
    ∇fx⁺ = ∇f(x⁺)
    Ared = fx - fx⁺
    Pred = fx - m
    ρ = Ared/Pred
    if ρ > η₁
      x = x⁺
      fx = fx⁺
      ∇fx = ∇fx⁺
      ngrad = norm(∇fx)
    end

    # Step 4
    if ρ < η₂
      μ = σ₁*μ
    else
      if norm(d) > r/2
        μ = σ₂*μ
      end
    end
    s = μ^α
    t = ngrad^β
    r = s*t

    # Step 5
    if ρ > η₁
      y = ∇fx - ∇fx₀
      a = dot(d,y)
      if a > 0
        b = dot(d,B*d)
        v = B*d
        B = B + (1/a)*y*y' - (1/b)*v*v'
      end
    end
    k += 1
    if k >= kmax
      ef = 1
      break
    end
    el_time = time() - st_time
  end # while

  if el_time > max_time
    ef = 2
  end

  return x, f(x), ∇fx, k, ef, el_time
end

function solve(f::Function, ∇f::Function, ∇²f::Function, l::Vector, u::Vector,
    x0::Vector; options::Options = Options())
  n = length(x0)
  P(x) = Float64[x[i] < l[i] ? l[i] : (x[i] > u[i] ? u[i] : x[i]) for i = 1:n]

  return solve(f, ∇f, ∇²f, P, x0, options=options)
end

function solve(f::Function, ∇f::Function, ∇²f::Function, P::Function,
    x0::Vector; options::Options = Options())
  for field in fieldnames(options)
    @eval $field = $(options).$(field)
  end

  ef = 0
  st_time = time()

  x = P(x0)
  if x != x0
    println("Warning: x0 was not feasible. Using P[x0] as initial point")
  end
  ∇fx = ∇f(x)
  B = ∇²f(x)
  s = ones(x)
  Δ = norm(∇fx)
  k = 0

  el_time = time() - st_time
  while norm(P(x-∇fx)-x, Inf) > ϵ && el_time < max_time
    @verbose("x = $x")
    @verbose("∇fx = $(∇fx)")
    @verbose("Δ = $Δ")
    # Compute the model ψ
    ψ(w) = dot(∇fx,w) + 0.5*dot(w,B*w)
    # Compute the Cauchy step sC
    sC = cauchyStep(x, ∇fx, B, P, Δ, verbose=verbose)
    # Compute the step s that satisfies (2.5)
    s = sC
    @verbose("s = $s")
    @verbose("ψ(s) = $(ψ(s))")
    # Compute the ratio ρ and update x by (2.2)
    ψ(s) >= 0 && error("ψ(s) = $(ψ(s)) >= 0")
    ρ = ( f(x+s)-f(x) )/ψ(s)
    @verbose("ρ = $ρ")
    # Update Δ according to (2.3)
    if ρ <= η₁
      Δnew = σ₂*Δ
      if Δnew < σ₁*norm(s)
        Δnew = σ₁*norm(s)
      end
    elseif ρ < η₂
      Δnew = Δ
    else
      Δnew = σ₃*Δ
    end
    Δ = max(min(Δnew, Δmax), Δmin)
    # Update x
    if ρ > η₀
      x = x + s
    end
    ∇fx = ∇f(x)
    B = ∇²f(x)

    k += 1
    @verbose("####################### k = $k")
    if k > kmax
      ef = 1
      break
    end
    el_time = time() - st_time
  end # while norm(s) > ϵ

  if el_time >= max_time
    ef = 2
  end

  return x, f(x), ∇fx, k, ef, el_time
end # function solve

# s(α) = P[x - α∇fx] - x
# (2.4) ψ(s) ≦ μ₀∇f(x)⋅s
#       |s| ≦ μ₁Δ
function cauchyStep(x::Vector, ∇fx::Vector, B::Matrix, P::Function, Δ::Real;
    ϵ::Real = 1e-5, μ₀::Real = 1e-2, μ₁::Real = 1.0, kmax = 50,
    verbose::Bool = false)
  α = 1.0
  s(α) = P(x-α*∇fx)-x
  sα = s(α)
  if norm(sα) < ϵ
    return sα
  end
  ψ(w) = dot(∇fx,w) + 0.5*dot(w,B*w)
  @verbose("cauchyStep")
  @verbose("  sα = $(sα)")
  @verbose("  ψ(sα) = $(ψ(sα))")
  @verbose("  ∇fx⋅sα = $(dot(∇fx, sα))")
  k = 0
  if ψ(sα) <= μ₀*dot(∇fx, sα) && norm(sα) <= μ₁*Δ
    αplus = 2*α
    splus = s(αplus)
    while ψ(splus) <= μ₀*dot(∇fx, splus) && norm(splus) <= μ₁*Δ && splus != sα
      α = αplus
      sα = splus
      αplus = 2*α
      splus = s(αplus)
      k += 1
      if k > kmax
        @verbose("  |s⁺| = $(norm(splus))")
        @verbose("  α⁺ = $(αplus)")
      end
      k > kmax && error("kmax on cauchyStep")
    end
  else
    αplus = 0.5*α
    splus = s(αplus)
    while ψ(splus) > μ₀*dot(∇fx, splus) || norm(splus) > μ₁*Δ
      α = αplus
      sα = splus
      αplus = 0.5*α
      splus = s(αplus)
      k > kmax && error("kmax on cauchyStep")
    end
    sα = splus
  end
  @verbose("  α⁺ = $(αplus)")
  return sα
end

end
