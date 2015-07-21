module NewTR

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
  function Options(ϵ::Real = 1e-5, η₀::Real = 1e-3, η₁::Real = 0.25, η₂::Real =
      0.75, σ₁::Real = 0.25, σ₂::Real = 0.5, σ₃::Real = 4.0, kmax::Int = 10000,
      Δmin::Real = 1e-12, Δmax::Real = 1e20, verbose::Bool = false)
    return new(ϵ, η₀, η₁, η₂, σ₁, σ₂, σ₃, kmax, Δmin, Δmax, verbose)
  end
end

@compat flags = Dict(
    0=> "Convergence criteria satisfied",
    1=> "Maximum number of iterations" )

function solve(f::Function, ∇f::Function, ∇²f::Function, l::Vector, u::Vector,
    x0::Vector; options::Options = Options())
  n = length(x0)
  P(x) = Float64[x[i] < l[i] ? l[i] : (x[i] > u[i] ? u[i] : x[i]) for i = 1:n]

  return solve(f, ∇f, ∇²f, P, x0, options=options)
end

function solve(f::Function, ∇f::Function, ∇²f::Function, P::Function,
    x0::Vector; options::Options = Options())
  ϵ = options.ϵ
  η₀ = options.η₀
  η₁ = options.η₁
  η₂ = options.η₂
  σ₂ = options.σ₂
  σ₁ = options.σ₁
  σ₃ = options.σ₃
  kmax = options.kmax
  Δmax = options.Δmax
  Δmin = options.Δmin
  verbose = options.verbose

  x = P(x0)
  if x != x0
    println("Warning: x0 was not feasible. Using P[x0] as initial point")
  end
  ∇fx = ∇f(x)
  B = ∇²f(x)
  s = ones(x)
  Δ = norm(∇fx)
  k = 0

  while norm(P(x-∇fx)-x, Inf) > ϵ
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
    k > kmax && return x, f(x), ∇fx, kmax, 1
  end # while norm(s) > ϵ

  return x, f(x), ∇fx, k, 0
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
