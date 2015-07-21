using NewTR
using CUTEst

function cutest_solve(nlp::CUTEstModel)
  if nlp.meta.ncon > 0
    error("ERROR: Can't solve constrained problems yet")
  else
    f(x) = jl_ufn(nlp, x)
    ∇f(x) = jl_ugr(nlp, x)
    ∇²f(x) = jl_udh(nlp, x, nlp.meta.nvar)

    opt = NewTR.Options()
    bounded = any([l > -1e20 for l in nlp.meta.lvar]) ||
        any([u < 1e20 for u in nlp.meta.uvar])

    if bounded
      function P(x::Array{Float64,1})
        y = copy(x)
        n = nlp.meta.nvar
        for i = 1:n
          if y[i] < nlp.meta.lvar[i]
            y[i] = nlp.meta.lvar[i]
          elseif y[i] > nlp.meta.uvar[i]
            y[i] = nlp.meta.uvar[i]
          end
        end
        return y
      end # P

      @time x, fx, ∇fx, k, ef = NewTR.solve(f, ∇f, ∇²f, P, nlp.meta.x0)
    else
      @time x, fx, ∇fx, k, ef = NewTR.solve(f, ∇f, ∇²f, nlp.meta.x0)
    end
  end

  println("#########")
  println("EXIT: $(NewTR.flags[ef])")
  if nlp.meta.nvar < 10
    println("x = $x")
  end
  println("fx = $fx")
  if bounded
    println("|P[x-∇fx]-x| = $(norm(P(x-∇fx)-x))")
  else
    println("|∇fx| = $(norm(∇fx))")
  end
  println("Iterations: $k")
end

if length(ARGS) < 1
  println("ERROR: Missing problem name")
  println("Usage\n  julia cutest.jl PROBLEM")
end

nlp = CUTEstModel(ascii(ARGS[1]))

cutest_solve(nlp)

