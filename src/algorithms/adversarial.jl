struct AdversarialDebiasing{M<:MLJBase.Model}<: Deterministic
    # grp::Symbol
    classifier::M
    # alpha::Float64
end

function AdversarialDebiasing(; classifier::MLJBase.Model = nothing)
    model = AdversarialDebiasing(classifier)
    message = MLJBase.clean!(model)
    isempty(message)||@warn message
    return model
end

function MLJBase.clean!(model::AdversarialDebiasing)
    warning = ""
    model.classifier!=nothing || (warning *= "No classifier specified in model\n")
    target_scitype(model) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported\n")
    # (model.alpha>=0 && model.alpha<=1) || (warning*="alpha should be between 0 and 1 (inclusive)\n")
    return warning
end

function classifier_model(features_dim, classifier_num_hidden_units, keep_prob)
	layer1 = Dense(features_dim,classifier_num_hidden_units; bias=true)
	layer2 = Dense(classifier_num_hidden_units, 1, bias = true, σ)
	model = Chain(layer1, LayerNorm(classifier_num_hidden_units, relu), layer2)
	return model
end

function adversary_model()
	layer1 = Dense(3,1;bias=true)
	return layer1
end

loss(ŷ,y) = Flux.logitbinarycrossentropy(ŷ, y; agg = mean)

classifier_opt = Flux.Optimise.Optimiser(Flux.Optimise.ExpDecay(0.001, 0.1, 1000, 1e-4), Flux.Optimise.ADAM())
classifier_opt2 = Flux.Optimise.Optimiser(Flux.Optimise.ExpDecay(0.001, 0.1, 1000, 1e-4), Flux.Optimise.ADAM())

normalise(temp) = temp/(norm(temp)+ 1e-8)

function MMI.fit(model::AdversarialDebiasing, verbosity::Int,X,y)
    # 	@show "Hey i am being called"
        df_n1 = DataFrames.DataFrame()
        for name in names(X)
            if typeof(X[:,name])==CategoricalArray{String,1,UInt32,String,CategoricalValue{String,UInt32},Union{}}
                df_n1[:,name] = coerce(X[:,name], Count, verbosity=0)
            else
                df_n1[:,name] = X[:,name]
            end
        end
        
        @show "this was called"
        
        train_loader = Flux.Data.DataLoader((Matrix(df_n1)', rand(Bool,6907)*1), batchsize=100, shuffle=true)
        
        # a3 = @pipeline ContinuousEncoder  NeuralNetworkClassifier
        # @show model.classifier
        # mch = machine(model.classifier, X, y)
        # @show mch
        # fit!(mch)
        classifier_net = classifier_model(8,5,1)
        adversary_net = adversary_model()
        classifier_vars = Flux.params(classifier_net)
        adversary_vars = Flux.params(adversary_net)
        for (x,y) in train_loader
            classifier_grad = gradient(() -> loss(σ.(classifier_net(x)),y), classifier_vars)
            adversary_grad = gradient(() -> loss(σ.(adversary_net(cat(σ.(2*classifier_net(x)), σ.(2*classifier_net(x)).*y',σ.(2*classifier_net(x)).*(1 .- y)';dims=1))),x[4,:]), classifier_vars)
            for var in classifier_vars
                unit_adversary_grad = normalise(adversary_grad[var])
                classifier_grad[var] = classifier_grad[var]-sum(classifier_grad[var].*unit_adversary_grad)*unit_adversary_grad
                classifier_grad[var] = classifier_grad[var] - 0.01*adversary_grad[var]
            end
            adversary_grad2 = gradient(() -> loss(σ.(adversary_net(cat(σ.(2*classifier_net(x)), σ.(2*classifier_net(x)).*y',σ.(2*classifier_net(x)).*(1 .- y)';dims=1))), x[4,:]), adversary_vars)
            Flux.Optimise.update!(classifier_opt, classifier_vars, classifier_grad)
            Flux.Optimise.update!(adversary_opt, adversary_vars, adversary_grad2)
            # append!(losses, loss(σ.(classifier_net(x)),y))
            # append!(losses_adversary, loss(σ.(adversary_net(cat(σ.(2*classifier_net(x)), σ.(2*classifier_net(x)).*y',σ.(2*classifier_net(x)).*(1 .- y)';dims=1))), rand(1,100)))
        end
        @show Flux.params(classifier_net)
        fitresult =  Flux.params(classifier_net)
        return fitresult
end

function MMI.predict(model::AdversarialDebiasing, fitresult, X)
	df_n1 = DataFrames.DataFrame()
	for name in names(X)
		if typeof(X[:,name])==CategoricalArray{String,1,UInt32,String,CategoricalValue{String,UInt32},Union{}}
			df_n1[:,name] = coerce(X[:,name], Count, verbosity=0)
		else
			df_n1[:,name] = X[:,name]
		end
	end
	
	@show size(fitresult)
	
	@show "this was called"
	classifier_net2 = classifier_model(8,5,1)
	@show Flux.params(classifier_net2)
	Flux.loadparams!(classifier_net2, fitted_params(mch2))
	ŷ = classifier_net2(Matrix(df_n1)')
	ŷ=round.(ŷ)
	return ŷ
end

MMI.input_scitype(::Type{<:AdversarialDebiasing{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:AdversarialDebiasing{M}}) where M = AbstractVector{<:Finite{2}}