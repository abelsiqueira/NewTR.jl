using NewTR
using CUTEst

function cutest_solve(nlp::CUTEstModel)
  if nlp.meta.ncon > 0
    error("ERROR: Can't solve constrained problems yet")
  else
    f(x) = jl_ufn(nlp, x)
    ∇f(x) = jl_ugr(nlp, x)
    ∇²f(x) = jl_udh(nlp, x, nlp.meta.nvar)

    function P (x::Array{Float64,1})
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

    @time x, fx, ∇fx, k, ef = NewTR.solve(f, ∇f, ∇²f, P, nlp.meta.x0, verbose=false,
    kmax=1000)
  end

  println("#########")
  println("EXIT: $(NewTR.flags[ef])")
  if nlp.meta.nvar < 10
    println("x = $x")
  end
  println("fx = $fx")
  println("|∇fx| = $(norm(∇fx))")
  println("|P[x-∇fx]-x| = $(norm(P(x-∇fx)-x))")
  println("Iterations: $k")
end

if length(ARGS) < 1
  println("ERROR: Missing problem name")
  println("Usage\n  julia cutest.jl PROBLEM")
end

nlp = CUTEstModel(ascii(ARGS[1]))

cutest_solve(nlp)

