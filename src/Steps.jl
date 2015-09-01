# Definition of inner steps, or minor iterates.

function more_sorensen(Δ::Float64, g::Vector, B::Matrix;
    kmax = 300)
  k = 0
  n = length(g)

  d = zeros(n)
  λ = 0
  while k < kmax
    H = B + λ*eye(n)
    d = -H\g
    normd = norm(d)
    if abs(λ) <= 1e-3 && normd <= Δ
      break
    elseif abs(normd-Δ)/Δ <= 1e-3
      break
    end
    u = H\d
    λ = max(0, λ + normd^2*(normd-Δ)/(dot(d,u)*Δ))
    k += 1
  end

  if k >= kmax
    error("KMAX")
  end

  return d
end

function steihaug_toint(Δ::Float64, g::Vector, B::Matrix;
    kmax = 1000, tol = 1e-6, verbose = false)
  n = length(g)
  z = zeros(n)
  D = diag(B)
  κmax = 1e8
  M = max(abs(D),norm(D)/κmax)
  r = copy(g)
  t = r./M
  rᵀt = dot(r,t)
  tol² = tol^2
  if norm(r) < tol
      @verbose("  exit: ‖r₀‖ < ϵ")
    return z
  end
  d = -t
  k = 0
  while k < kmax
    Bd = B*d
    dᵀBd = dot(d,Bd); dᵀd = dot(d,d)
    @verbose("  dᵀBd = $(dᵀBd)")
    if dᵀBd <= 1e-12*dᵀd
      # Find τ such that p = z + τd minimizes ϕ(τ) = m(p)-m(z) and ‖p‖ = Δ
      #dᵀd = dot(d,M.*d); dᵀz = dot(d,M.*z); zᵀz = dot(z,M.*z)
      @verbose("  dᵀBd/dᵀd = $(dᵀBd/dᵀd)")
      dᵀd = dot(d,d); dᵀz = dot(d,z); zᵀz = dot(z,z)
      D = sqrt(dᵀz^2 - dᵀd*(zᵀz - Δ^2))
      τ₊ = (-dᵀz + D)/dᵀd
      τ₋ = (-dᵀz - D)/dᵀd
      b = dot(B*z+g,g)
      ϕ(τ) = τ*b + 0.5*τ^2*dᵀBd
      if ϕ(τ₊) < ϕ(τ₋)
        τ = τ₊
      else
        τ = τ₋
      end
      @verbose("  exit: Negative curvature")
      return z + τ*d
    end
    α = rᵀt/dᵀBd
    z₊ = z + α*d
    @verbose("  ‖z₊‖ = $(norm(z₊))")
    #if dot(z₊,M.*z₊) > Δ^2
    if dot(z₊,z₊) > Δ^2
      # Find τ ≧ 0 such that p = z + τd satisfies ‖p‖ = Δ
      #dᵀd = dot(d,M.*d); dᵀz = dot(d,M.*z); zᵀz = dot(z,M.*z)
      dᵀd = dot(d,d); dᵀz = dot(d,z); zᵀz = dot(z,z)
      D = sqrt(dᵀz^2 - dᵀd*(zᵀz - Δ^2))
      τ = (-dᵀz + D)/dᵀd
      @verbose("  exit: Step larger than trust region")
      return z + τ*d
    end
        copy!(z, z₊)
    r = r + α*Bd
    t = r./M
    rᵀt₊ = dot(r,t)
    if norm(r)/norm(g) < tol
      @verbose("  exit: ‖r‖/‖g‖ < ϵ")
      return z
    end
    β = rᵀt₊/rᵀt
    d = -t + β*d
    rᵀt = rᵀt₊
    k += 1
  end

  return z
end
