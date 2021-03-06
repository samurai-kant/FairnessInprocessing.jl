### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 0f2b2e38-f2bb-11eb-2f1a-eb1c29869308
begin
	using Pkg
	Pkg.add(url="https://github.com/samurai-kant/Fairness.jl.git")
	Pkg.add("Flux")
	Pkg.add("Downloads")
	Pkg.add("CSV")
	Pkg.add("DataFrames")
	Pkg.add("Statistics")
	Pkg.add("LinearAlgebra")
end

# ╔═╡ 8ca89ec2-f2bb-11eb-3ffe-0130194a8f43
begin
	using Downloads
	using Fairness
	using Flux
	using CSV
	using DataFrames
	using Statistics
	using LinearAlgebra
end

# ╔═╡ 4d38534e-f2bc-11eb-2b66-2f694e3e2410
begin
	const MODULE_DIR = "/Users/agango/Desktop/Datasets"
	const DATA_DIR = joinpath(MODULE_DIR)
	function ensure_download(url::String, file::String)
	    cd(DATA_DIR) # This is to ensue that the dataset is not downloaded to /tmp instead of ./data
	    fpath = joinpath(DATA_DIR, file)
	    if !isfile(fpath)
	        Downloads.download(url, file)
	    end
	end
end

# ╔═╡ be81eefa-f2bb-11eb-368d-775fef1a0a89
begin
	url = "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
	fname = "communities_crime.data"
	cols = ["state", "county", "community", "communityname", "fold", "population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian", "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWWage", "pctWFarmSelf", "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc", "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade", "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce", "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids", "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5", "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous", "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian", "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState", "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop", "ViolentCrimesPerPop"]
	ensure_download(url, fname)
	fpath = joinpath(DATA_DIR, fname)
	df = DataFrame(CSV.File(fname, header=cols, silencewarnings=true, delim=","); copycols = false)
	df = dropmissing(df, names(df))
	X1 = df[!, names(df)[1:127]]
	y1 = df[!, names(df)[128]] .> 0.7
	y1= categorical(y1)
	(X1, y1)
end

# ╔═╡ ebaea712-f2bc-11eb-01a9-f1f933afac79
begin
	df_cleaned = df[:,eltype.(eachcol(df)).!=String]
	df_cleaned = select!(df_cleaned, Not([:"state",:"fold"]))
end

# ╔═╡ c7151616-f2bc-11eb-148d-2578d6582076
begin
	crime_rate_70_percentile = quantile!(df[:,128], 0.7)
	race_black_70_percentile = quantile!(df[:,"racepctblack"],0.7)
	
	crimes_df = df[:,"ViolentCrimesPerPop"] .> crime_rate_70_percentile
	groups_df = df[:,"racepctblack"] .> race_black_70_percentile
end

# ╔═╡ 820c7cf6-f2be-11eb-338e-c9d4758793f4
begin
	df_cleaned.mergekey = zeros(1994)
	df_pos = df_cleaned[crimes_df,:]
	df_neg = df_cleaned[crimes_df.==false,:]
end

# ╔═╡ e6c0f142-f2cb-11eb-2bfb-7943a7231cd7
queries = rand(1:10, 
822613)

# ╔═╡ 9a530fc2-f2bf-11eb-0d6f-4d4cf253221e
begin
	pair_features = outerjoin(df_pos, df_neg, on=:"mergekey",makeunique=true)
	pair_features = select!(pair_features, Not([:"mergekey"]))
	pos_groups = pair_features[:,"racepctblack"] .> race_black_70_percentile
	neg_groups = pair_features[:,"racepctblack_1"] .> race_black_70_percentile
	pair_features_pos = Matrix(pair_features[:,1:convert(Int64,size(pair_features)[2]/2)])
	pair_features_neg = Matrix(pair_features[:,convert(Int64,(size(pair_features)[2]/2+1)):convert(Int64,size(pair_features)[2])])
	pair_features_pos = convert(Array{Float64,2}, pair_features_pos)
	pair_features_neg = convert(Array{Float64,2}, pair_features_neg)
end

# ╔═╡ e18a1a3c-f2c1-11eb-3b91-f104bd2247e8
begin
	W = zeros(1,100)
	predict(x) = W*x'
	function loss(x1, x2)
		sum(max.(0,(predict(x1) .- predict(x2))))
	end
	function lossbin(x1, x2)
		sum(predict(x1) .> predict(x2))
	end
end

# ╔═╡ bf04688e-f2cc-11eb-0978-3d287ac97dfa
select_all = ones(
822613)

# ╔═╡ e621bd38-f2d1-11eb-07a5-87ed8cdc286d
.~neg_groups

# ╔═╡ 15e9fac6-f2cd-11eb-3cd5-1d2b9c9349ff
begin
	mask = queries.==5
	mask_pos = mask .* (pos_groups .* .~neg_groups)
	mask_neg = mask .* (.~pos_groups .* neg_groups)
end

# ╔═╡ c4f94664-f301-11eb-2fb8-d9b9b7838e93
ϵ = 0.01

# ╔═╡ b2768480-f34b-11eb-1f9b-fd342bb6269b
M = ones(4,4)*0.25

# ╔═╡ db96504c-f356-11eb-2698-3bc73a8e0e57
M

# ╔═╡ 109ef703-768e-46c7-be37-18f9c7b68601
λ2 = eigvecs(M)[findmax(abs.(eigvals(M)))[2],:]

# ╔═╡ 33f47df7-988a-4e7d-90f8-05a8aff5019f
findmax(λ2)[1]

# ╔═╡ 9cc956ee-9527-48e2-b47c-11f32dcc075f
opt = Descent(0.1)

# ╔═╡ f2944635-f285-4b87-b5ee-26be6d5d035e
λ=randn(4)

# ╔═╡ b717b880-f2c2-11eb-35ab-1b1df15ac2bb
grads = gradient(()->λ[1]*loss(pair_features_pos[mask,:], pair_features_neg[mask,:])+λ[2]*loss(pair_features_pos[mask_pos,:], pair_features_neg[mask_pos,:])+λ[3]*loss(pair_features_pos[mask_neg,:], pair_features_neg[mask_neg,:])+λ[4], Flux.params(W))

# ╔═╡ a88a06c2-21a4-4090-a4c8-e18589376e02
grads[W]

# ╔═╡ aaa1de28-f2fe-11eb-17c4-bfffadf1e873
grads_λ = gradient(()->λ[2]*(lossbin(pair_features_pos[mask_pos,:], pair_features_neg[mask_pos,:])-ϵ)+λ[3]*(lossbin(pair_features_pos[mask_neg,:], pair_features_neg[mask_neg,:])-ϵ)+λ[4], Flux.params(λ))

# ╔═╡ 00583ace-f2c3-11eb-0c44-0915bf9d683b
M_update = grads_λ[λ]*λ'

# ╔═╡ 155049ba-f351-11eb-28b2-47f133b8fae5
M_new = M.^(M_update*0.001)

# ╔═╡ ea3e4730-f354-11eb-3399-dff8b0275932
begin
	M[:,1] = M_new[:,1]/sum(abs.(M_new[:,1]))
	M[:,2] = M_new[:,2]/sum(abs.(M_new[:,2]))
	M[:,3] = M_new[:,3]/sum(abs.(M_new[:,3]))
	M[:,4] = M_new[:,4]/sum(abs.(M_new[:,4]))
end

# ╔═╡ 61697053-7bf9-4a47-8f29-b9367488b123
grads_λ[λ]

# ╔═╡ ecce427a-a17e-4278-80dd-5ffb9f3700e9
eigvecs(M)[:,4]
	

# ╔═╡ b4cb96ea-ab14-4c07-8c0f-f26fc4fb988b
M*eigvecs(M)[:,4]
	

# ╔═╡ 84a755fb-671f-41ca-baa7-b9e05acb3104
λ

# ╔═╡ 58b24dd7-6198-4895-8ead-3e888b5d656a
begin
	for j in 1:10
		λ = eigvecs(M)[:,findmax(abs.(eigvals(M)))[2]]
		grads = gradient(()->λ[1]*loss(pair_features_pos[mask,:], pair_features_neg[mask,:])+λ[2]*loss(pair_features_pos[mask_pos,:], pair_features_neg[mask_pos,:])+λ[3]*loss(pair_features_pos[mask_neg,:], pair_features_neg[mask_neg,:])+λ[4], Flux.params(W))
		grads_λ = gradient(()->λ[2]*(lossbin(pair_features_pos[mask_pos,:], pair_features_neg[mask_pos,:])-ϵ)+λ[3]*(lossbin(pair_features_pos[mask_neg,:], pair_features_neg[mask_neg,:])-ϵ)+λ[4], Flux.params(λ))
		Flux.update!(opt, W, grads[W])
		M_update = grads_λ[λ]*λ'
		M_new = M.*(ℯ.^(M_update*0.001))
		M[:,1] = M_new[:,1]/sum(abs.(M_new[:,1]))
		M[:,2] = M_new[:,2]/sum(abs.(M_new[:,2]))
		M[:,3] = M_new[:,3]/sum(abs.(M_new[:,3]))
		M[:,4] = M_new[:,4]/sum(abs.(M_new[:,4]))
	end
end

# ╔═╡ 301fae52-f5c2-4fa2-a719-b300a47b897d
M

# ╔═╡ 2de75817-f6bc-48aa-9ca1-5fa5f25b38d2
W

# ╔═╡ b33a8bc8-a5a0-4f74-9beb-aab808be324c
M

# ╔═╡ e163f362-c0a4-4174-81f7-c4efc701b558
M*eigvecs(M)[:,findmax(abs.(eigvals(M)))[2]]

# ╔═╡ Cell order:
# ╠═0f2b2e38-f2bb-11eb-2f1a-eb1c29869308
# ╠═8ca89ec2-f2bb-11eb-3ffe-0130194a8f43
# ╠═4d38534e-f2bc-11eb-2b66-2f694e3e2410
# ╠═be81eefa-f2bb-11eb-368d-775fef1a0a89
# ╠═ebaea712-f2bc-11eb-01a9-f1f933afac79
# ╠═c7151616-f2bc-11eb-148d-2578d6582076
# ╠═820c7cf6-f2be-11eb-338e-c9d4758793f4
# ╠═e6c0f142-f2cb-11eb-2bfb-7943a7231cd7
# ╠═9a530fc2-f2bf-11eb-0d6f-4d4cf253221e
# ╠═e18a1a3c-f2c1-11eb-3b91-f104bd2247e8
# ╠═bf04688e-f2cc-11eb-0978-3d287ac97dfa
# ╠═e621bd38-f2d1-11eb-07a5-87ed8cdc286d
# ╠═15e9fac6-f2cd-11eb-3cd5-1d2b9c9349ff
# ╠═c4f94664-f301-11eb-2fb8-d9b9b7838e93
# ╠═b2768480-f34b-11eb-1f9b-fd342bb6269b
# ╠═b717b880-f2c2-11eb-35ab-1b1df15ac2bb
# ╠═aaa1de28-f2fe-11eb-17c4-bfffadf1e873
# ╠═00583ace-f2c3-11eb-0c44-0915bf9d683b
# ╠═155049ba-f351-11eb-28b2-47f133b8fae5
# ╠═ea3e4730-f354-11eb-3399-dff8b0275932
# ╠═db96504c-f356-11eb-2698-3bc73a8e0e57
# ╠═109ef703-768e-46c7-be37-18f9c7b68601
# ╠═33f47df7-988a-4e7d-90f8-05a8aff5019f
# ╠═a88a06c2-21a4-4090-a4c8-e18589376e02
# ╠═61697053-7bf9-4a47-8f29-b9367488b123
# ╠═9cc956ee-9527-48e2-b47c-11f32dcc075f
# ╠═f2944635-f285-4b87-b5ee-26be6d5d035e
# ╠═ecce427a-a17e-4278-80dd-5ffb9f3700e9
# ╠═b4cb96ea-ab14-4c07-8c0f-f26fc4fb988b
# ╠═84a755fb-671f-41ca-baa7-b9e05acb3104
# ╠═58b24dd7-6198-4895-8ead-3e888b5d656a
# ╠═301fae52-f5c2-4fa2-a719-b300a47b897d
# ╠═2de75817-f6bc-48aa-9ca1-5fa5f25b38d2
# ╠═b33a8bc8-a5a0-4f74-9beb-aab808be324c
# ╠═e163f362-c0a4-4174-81f7-c4efc701b558
