# This creates a configuration file for NewtR, then runs it  with a list of
# CUTEst problems and prints a evaluation cost based on the amount of time
# it took for convergence to be obtained.

max_time = 60
kmax = 5000

# The arguments needed are η₁, η₂, δ₀, α, β, σ₁ and σ₂
if length(ARGS) < 7
  error("ERROR: Need 7 arguments to run: η₁ η₂ δ₀ α β σ₁ σ₃")
end

args = ["η₁", "η₂", "δ₀", "α", "β", "σ₁", "σ₃"]

file = open("test/cutest.param","w")
for i = 1:7
  write(file, "$(args[i]) $(ARGS[i])\n")
end
write(file, "max_time $max_time\n")
write(file, "kmax $kmax\n")
close(file)

test_file = "test/cutest-test.list"
#test_file = "test/cutest-2var-free.list"

objfun = 0.0

file = open(test_file)
for line in readlines(file)
  out = readall(`make cutest PROBLEM=$(chomp(line)) OPTS=test/cutest.param`)
  converged = contains(out, "EXIT 0")
  time = parse(match(r"Elapsed time: ([0-9\.]*)", out).captures[1])
  if converged
    objfun += time
  else
    objfun += kmax
  end
end
println(objfun)
