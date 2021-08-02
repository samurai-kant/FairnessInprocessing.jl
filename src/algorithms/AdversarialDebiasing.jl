### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 32fccb86-dc43-11eb-1615-dd78fd427d06
begin
	using Pkg
	Pkg.activate("Inprocessing_5")
	Pkg.add("Flux")
	Pkg.add("Plots")
	Pkg.add("MLJ")
	Pkg.add("LinearAlgebra")
end

# ╔═╡ c8d134f2-dd05-11eb-172f-3fb55f0042fb
using Fairness

# ╔═╡ fd2b214e-dc43-11eb-085b-1ff6e98b0fad
begin
	using Flux
	using Plots
	using MLJ
	using LinearAlgebra
end

# ╔═╡ bf0a32f2-dd05-11eb-0178-290b127637b1
Pkg.add(url="https://github.com/samurai-kant/Fairness.jl.git")

# ╔═╡ 635e5f7c-dd06-11eb-1f96-e1652adf1b42
data = @load_adult

# ╔═╡ 6cc77846-dd06-11eb-12ba-e92c26c06476
X, y = data

# ╔═╡ 07e8f378-dd0b-11eb-13c9-7193b5d9d49c
v = Matrix(X)

# ╔═╡ 85229626-dd0d-11eb-279d-9f6f35f380a8
g = Int64(X[:,2])

# ╔═╡ 19bf21cc-dc44-11eb-1d19-d3ad82d294f2
function classifier_model(features_dim, classifier_num_hidden_units, keep_prob)
	layer1 = Dense(features_dim,classifier_num_hidden_units; bias=true)
	layer2 = Dense(classifier_num_hidden_units, 1, bias = true)
	model = Chain(layer1, LayerNorm(classifier_num_hidden_units, relu), layer2)
	return model
end

# ╔═╡ d7209d48-dc44-11eb-1239-f79884d3fcbb
function adversary_model()
	layer1 = Dense(3,1;bias=true)
	return layer1
end

# ╔═╡ e63b61b8-dc45-11eb-150d-17c3ff716ca9
loss(ŷ,y) = Flux.logitbinarycrossentropy(ŷ, y; agg = mean)

# ╔═╡ 465d8aee-dc46-11eb-0b1d-43c11a27d886
rand(5,1000)

# ╔═╡ 56fa88a2-dc46-11eb-0928-03de70a237c9
classifier_net1 = classifier_model(5,3,1)

# ╔═╡ 5dac88d0-dc46-11eb-04b9-b3e135f36e18
classifier_net1(rand(5,1000))

# ╔═╡ 7da2af1e-dc4d-11eb-12f2-c3777766fc62
true_labels = rand(Bool,1,10000)

# ╔═╡ 3f8fc0dc-dc51-11eb-30b9-4946f8de483c
Xtrain = rand(5,10000)

# ╔═╡ a2aaf2ca-dcdf-11eb-29b2-59bf788e615c
Ytrain = rand(Bool,10000)*1

# ╔═╡ fe5391a0-dcd9-11eb-284d-e3735d38cb82
train_loader = Flux.Data.DataLoader((Xtrain, Ytrain), batchsize=100, shuffle=true)

# ╔═╡ 8256786e-dc53-11eb-2f92-d318d12a1607
normalise(temp) = temp/(norm(temp)+ 1e-8)

# ╔═╡ a37538c8-dc53-11eb-3a8c-ddabadfcfc08
classifier_opt = Flux.Optimise.Optimiser(Flux.Optimise.ExpDecay(0.001, 0.1, 1000, 1e-4), Flux.Optimise.ADAM())

# ╔═╡ a97bd004-dc53-11eb-1878-83107ffdbf8a
adversary_opt = Flux.Optimise.Optimiser(Flux.Optimise.ExpDecay(0.001, 0.1, 1000, 1e-4), Flux.Optimise.ADAM())

# ╔═╡ 8a294e12-dc54-11eb-2ec5-537de3c650e1
losses = []

# ╔═╡ 1148586a-dce2-11eb-1ad7-05c6ae2b40a2
begin
	a = 0
	b = 0
	for (x,y) in train_loader
	a=x
	b=y
	end
end

# ╔═╡ 1fe32a94-dce2-11eb-2f93-f5d6b88b8bd7
a

# ╔═╡ 2169f6ae-dce2-11eb-1fb4-b707cac2cdb2
b

# ╔═╡ 7ad6fc98-dce5-11eb-2353-995a06ecd7be
1 .- b

# ╔═╡ 5171a414-dce5-11eb-19d8-3959caa96908
# (2*classifier_net(a)).*(1 .- b)'

# ╔═╡ e983e43c-dce8-11eb-0c67-c3dc271278bd
# σ.(adversary_net(cat(σ.(2*classifier_net(a)), σ.(2*classifier_net(a)).*b',σ.(2*classifier_net(a)).*(1 .- b)';dims=1)))

# ╔═╡ 775ee5f2-dce1-11eb-032b-a7196b859e5e
# adversary_grad2 = gradient(() -> loss(σ.(adversary_net(cat(σ.(2*classifier_net(a)), σ.(2*classifier_net(a)).*b',σ.(2*classifier_net(a)).*(1 .- b)';dims=1))), rand(1,100)), classifier_vars)

# ╔═╡ fe4e9314-de5d-11eb-06ca-1daa0ca85ed0
losses_adversary = []

# ╔═╡ cd4a80a8-dcdb-11eb-2618-53206637b422
begin
	classifier_net = classifier_model(5,3,1)
	adversary_net = adversary_model()
	classifier_vars = Flux.params(classifier_net)
	adversary_vars = Flux.params(adversary_net)
	for (x,y) in train_loader
		classifier_grad = gradient(() -> loss(σ.(classifier_net(x)),y), classifier_vars)
		adversary_grad = gradient(() -> loss(σ.(adversary_net(cat(σ.(2*classifier_net(x)), σ.(2*classifier_net(x)).*y',σ.(2*classifier_net(x)).*(1 .- y)';dims=1))), rand(1,100)), classifier_vars)
		for var in classifier_vars
			unit_adversary_grad = normalise(adversary_grad[var])
			classifier_grad[var] = classifier_grad[var]-sum(classifier_grad[var].*unit_adversary_grad)*unit_adversary_grad
			classifier_grad[var] = classifier_grad[var] - 0.01*adversary_grad[var]
		end
		adversary_grad2 = gradient(() -> loss(σ.(adversary_net(cat(σ.(2*classifier_net(x)), σ.(2*classifier_net(x)).*y',σ.(2*classifier_net(x)).*(1 .- y)';dims=1))), rand(1,100)), adversary_vars)
		Flux.Optimise.update!(classifier_opt, classifier_vars, classifier_grad)
		Flux.Optimise.update!(adversary_opt, adversary_vars, adversary_grad2)
		append!(losses, loss(σ.(classifier_net(x)),y))
		append!(losses_adversary, loss(σ.(adversary_net(cat(σ.(2*classifier_net(x)), σ.(2*classifier_net(x)).*y',σ.(2*classifier_net(x)).*(1 .- y)';dims=1))), rand(1,100)))
		
	end
end

# ╔═╡ b50239a0-dc54-11eb-3386-3b7fe1da6dfe
length(losses)

# ╔═╡ 1ec0fa56-de5e-11eb-2314-7ddaff5d2282
losses_adversary

# ╔═╡ d25e5006-dc54-11eb-26b8-49dd0b291d17
plot(1:2100, losses)

# ╔═╡ 25c58ad8-de5e-11eb-2d73-d31840f1e6f2
plot(1:200, losses_adversary)

# ╔═╡ Cell order:
# ╠═32fccb86-dc43-11eb-1615-dd78fd427d06
# ╠═bf0a32f2-dd05-11eb-0178-290b127637b1
# ╠═c8d134f2-dd05-11eb-172f-3fb55f0042fb
# ╠═635e5f7c-dd06-11eb-1f96-e1652adf1b42
# ╠═6cc77846-dd06-11eb-12ba-e92c26c06476
# ╠═07e8f378-dd0b-11eb-13c9-7193b5d9d49c
# ╠═85229626-dd0d-11eb-279d-9f6f35f380a8
# ╠═fd2b214e-dc43-11eb-085b-1ff6e98b0fad
# ╠═19bf21cc-dc44-11eb-1d19-d3ad82d294f2
# ╠═d7209d48-dc44-11eb-1239-f79884d3fcbb
# ╠═e63b61b8-dc45-11eb-150d-17c3ff716ca9
# ╠═465d8aee-dc46-11eb-0b1d-43c11a27d886
# ╠═56fa88a2-dc46-11eb-0928-03de70a237c9
# ╠═5dac88d0-dc46-11eb-04b9-b3e135f36e18
# ╠═7da2af1e-dc4d-11eb-12f2-c3777766fc62
# ╠═3f8fc0dc-dc51-11eb-30b9-4946f8de483c
# ╠═a2aaf2ca-dcdf-11eb-29b2-59bf788e615c
# ╠═fe5391a0-dcd9-11eb-284d-e3735d38cb82
# ╠═8256786e-dc53-11eb-2f92-d318d12a1607
# ╠═a37538c8-dc53-11eb-3a8c-ddabadfcfc08
# ╠═a97bd004-dc53-11eb-1878-83107ffdbf8a
# ╠═8a294e12-dc54-11eb-2ec5-537de3c650e1
# ╠═1148586a-dce2-11eb-1ad7-05c6ae2b40a2
# ╠═1fe32a94-dce2-11eb-2f93-f5d6b88b8bd7
# ╠═2169f6ae-dce2-11eb-1fb4-b707cac2cdb2
# ╠═7ad6fc98-dce5-11eb-2353-995a06ecd7be
# ╠═5171a414-dce5-11eb-19d8-3959caa96908
# ╠═e983e43c-dce8-11eb-0c67-c3dc271278bd
# ╠═775ee5f2-dce1-11eb-032b-a7196b859e5e
# ╠═fe4e9314-de5d-11eb-06ca-1daa0ca85ed0
# ╠═cd4a80a8-dcdb-11eb-2618-53206637b422
# ╠═b50239a0-dc54-11eb-3386-3b7fe1da6dfe
# ╠═1ec0fa56-de5e-11eb-2314-7ddaff5d2282
# ╠═d25e5006-dc54-11eb-26b8-49dd0b291d17
# ╠═25c58ad8-de5e-11eb-2d73-d31840f1e6f2
