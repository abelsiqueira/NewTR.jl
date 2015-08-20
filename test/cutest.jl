using NewTR
using CUTEst

function cutest_solve(nlp::CUTEstModel, options_file::String = "")
  if nlp.meta.ncon > 0
    error("ERROR: Can't solve constrained problems yet")
  else
    if options_file != ""
      options = NewTR.ReadOptions(options_file)
    else
      options = NewTR.Options()
    end
    f(x) = ufn(nlp, x)
    ∇f(x) = ugr(nlp, x)
    ∇f!(x, g) = ugr!(nlp, x, g)
    ∇²f(x) = udh(nlp, x, nlp.meta.nvar)

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

      @time x, fx, ∇fx, k, ef, el_time = NewTR.solve(f, ∇f, ∇²f, P, nlp.meta.x0,
          options=options)
    else
      @time x, fx, ∇fx, k, ef, el_time = NewTR.solve(f, ∇f!, ∇²f, nlp.meta.x0,
          options=options)
    end
  end

  println("#########")
  println("EXIT $ef: $(NewTR.flags[ef])")
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
  println("Elapsed time: $el_time")
end

if length(ARGS) < 1
  println("ERROR: Missing problem name")
  println("Usage\n  julia cutest.jl PROBLEM")
end

if length(ARGS) > 1
  options_file = ARGS[2]
else
  options_file = ""
end

nlp = CUTEstModel(ascii(ARGS[1]))

cutest_solve(nlp, options_file)

