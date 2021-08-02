function DisparateImpactRemover(X, protected_attribute, select_grp)
	grps = levels(X[:,protected_attribute])
	for grp in grps
		select_grp = X[:,protected_attribute] .== grp
		category = X[select_grp,:]
		all_grps[grp] = nquantile(category[:,repair_column],10000)
	end
	medians = Dict()
	for index in 1:10000
		medians[index] = []
	end
	for index in 1:10000	
		for grp in grps
			append!(medians[index],all_grps[grp][index])
		end
	end
	for grp in grps
		select_grp = X[:,protected_attribute] .== grp
		category = X[select_grp,:]
		for (index,var) in enumerate(category[:,select_grp])
			try
				category[index,select_grp]=sort!(medians[convert(Int64,floor(cdf(var)*10000))])[4]
            catch
				println("Couldn't replace")
			end
		end
	end
end

struct DisparateImpactRemoverWrapper{M<:MLJBase.Model} <: DeterministicComposite
	grp::Symbol
	classifier::M
	protected_attribute::Int64
	select_grp::Int64
end

function DisparateImpactRemoverWrapper(; classifier::MLJBase.Model=nothing, grp::Symbol=:class,protect_attribute::Int64, select_grp::Int64)
    model = DisparateImpactRemoverWrapper(grp, classifier, protected_attribute, select_grp)
    message = MLJBase.clean!(model)
    isempty(message)||@warn message
    return model
end

function MLJBase.clean!(model::DisparateImpactRemoverWrapper)
    warning = ""
    model.classifier!=nothing || (warning *= "No classifier specified in model\n")
    target_scitype(model) <: AbstractVector{<:Finite} || (warning *= "Only Binary Classifiers are supported\n")
    supports_weights(model.classifier) || (warning *= "Classifier provided does not support weights\n")
    return warning
end

function MLJBase.fit(model::DisparateImpactRemoverWrapper, verbosity::Int, X,y)
	grps = X[:, model.grp]
	
	X_repaired = DisparateImpactRemover(X, protected_attribute, select_grp)
	
	classifier = model.classifier
    mach1 = machine(classifier, X_repaired, y)
	fit!(mach1)
	return mach1
end

MMI.input_scitype(::Type{<:DisparateImpactRemoverWrapper{M}}) where M = input_scitype(M)
MMI.target_scitype(::Type{<:DisparateImpactRemoverWrapper{M}}) where M = AbstractVector{<:Finite{2}}
istype(model::DisparateImpactRemoverWrapper, type) = istype(model.classifier, type)

