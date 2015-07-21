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
