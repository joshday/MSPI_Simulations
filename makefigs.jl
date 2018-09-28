module Figures
using OnlineStats, Plots, ProgressMeter, LossFunctions, PenaltyFunctions, Statistics,
    SparseRegression
import Plots: plot

#-----------------------------------------------------------------------# data generators
beta(p) = collect(range(-1, stop=1, length=p))

function linregdata(n, p, β = beta(p), σ = 1.0)
    x = randn(n, p)
    y = x * β + σ .* randn(n)
    return x, y, β 
end

function logregdata(n, p, β = beta(p))
	x = rand(n, p)
	y = 2.0 .* [rand() <= 1 / (1 + exp(-η)) for η in x * β] .- 1.0
	return x, y, β
end

data(::DistanceLoss, n, p) = linregdata(n, p)
data(::MarginLoss, n, p) = logregdata(n, p)

#-----------------------------------------------------------------------# Ribbon
struct Ribbon
    low::Vector{Float64}
    mid::Vector{Float64}
    hi::Vector{Float64}
    label::String
end
function Ribbon(results::Matrix{Float64}, label::String)
    mat = mapslices(x -> quantile(x, [.05, .5, .95]), results, dims=2)
    Ribbon(mat[:,1], mat[:,2], mat[:,3], label)
end
@recipe function f(o::Ribbon)
    ribbon --> (o.low, o.hi)
    fillalpha --> .1
    w --> 2
    label --> o.label
    o.mid
end
# plot(o::Ribbon) = plot(o.mid, ribbon=(o.low, o.hi), fillalpha=.1, w=2, label=o.label)

struct Figure 
    ribbons::Vector{Ribbon}
    title::String
end
@recipe function f(o::Figure)
    title --> o.title
    for r in o.ribbons
        @series begin r end
    end
end


# function Ribbon(data::Matrix{<:Real}, label)
#     mid = median(data, 2)
#     low = mid - mapslices(x -> quantile(x, [.05]), data, 2)
#     high = mapslices(x -> quantile(x, [.95]), data, 2) - mid
#     new(vec(low), vec(mid), vec(high), label)
# end


#-----------------------------------------------------------------------# simulation
function simulation(n, p, nreps=100, seed=123; loss = .5 * L2DistLoss(), kw...)
    results = Dict(
        :SGD => Dict(
            :r5 => Matrix{Float64}(undef, n, nreps),
            :r7 => Matrix{Float64}(undef, n, nreps),
            :r9 => Matrix{Float64}(undef, n, nreps)
        ),
        :ADAGRAD => Dict(
            :r5 => Matrix{Float64}(undef, n, nreps),
            :r7 => Matrix{Float64}(undef, n, nreps),
            :r9 => Matrix{Float64}(undef, n, nreps)
        ),
        :MSPI => Dict(
            :r5 => Matrix{Float64}(undef, n, nreps),
            :r7 => Matrix{Float64}(undef, n, nreps),
            :r9 => Matrix{Float64}(undef, n, nreps)
        )
    )

    bestmean = Mean()

    @showprogress for r in 1:nreps
        x, y, beta = data(loss, n, p)
        smod = SModel(x, y, loss, NoPenalty())
        learn!(smod; verbose=false)
        best = value(smod)
        fit!(bestmean, best)
        sgd = Series(
            StatLearn(p, loss, NoPenalty(), SGD(); rate=LearningRate(.5)),
            StatLearn(p, loss, NoPenalty(), SGD(); rate=LearningRate(.7)),
            StatLearn(p, loss, NoPenalty(), SGD(); rate=LearningRate(.9)),
        )
        ada = Series(
            StatLearn(p, loss, NoPenalty(), ADAGRAD(); rate=LearningRate(.5)),
            StatLearn(p, loss, NoPenalty(), ADAGRAD(); rate=LearningRate(.7)),
            StatLearn(p, loss, NoPenalty(), ADAGRAD(); rate=LearningRate(.9)),
        )
        mspi = Series(
            StatLearn(p, loss, NoPenalty(), MSPI(); rate=LearningRate(.5)),
            StatLearn(p, loss, NoPenalty(), MSPI(); rate=LearningRate(.7)),
            StatLearn(p, loss, NoPenalty(), MSPI(); rate=LearningRate(.9)),
        )
        for i in 1:n
            xi, yi = @view(x[i, :]), y[i]
            fit!(sgd, (xi, yi))
            fit!(ada, (xi, yi))
            fit!(mspi, (xi, yi))
            results[:SGD][:r5][i, r]        = OnlineStats.objective(sgd.stats[1], x, y) - best
            results[:SGD][:r7][i, r]        = OnlineStats.objective(sgd.stats[2], x, y) - best
            results[:SGD][:r9][i, r]        = OnlineStats.objective(sgd.stats[3], x, y) - best
            results[:ADAGRAD][:r5][i, r]    = OnlineStats.objective(ada.stats[1], x, y) - best
            results[:ADAGRAD][:r7][i, r]    = OnlineStats.objective(ada.stats[2], x, y) - best
            results[:ADAGRAD][:r9][i, r]    = OnlineStats.objective(ada.stats[3], x, y) - best
            results[:MSPI][:r5][i, r]       = OnlineStats.objective(mspi.stats[1], x, y) - best
            results[:MSPI][:r7][i, r]       = OnlineStats.objective(mspi.stats[2], x, y) - best
            results[:MSPI][:r9][i, r]       = OnlineStats.objective(mspi.stats[3], x, y) - best
        end
    end
    ribbons = Dict(
        :SGD => Dict(
            :r5 => Ribbon(results[:SGD][:r5], "r = .5"),
            :r7 => Ribbon(results[:SGD][:r7], "r = .7"),
            :r9 => Ribbon(results[:SGD][:r9], "r = .9")
        ),
        :ADAGRAD => Dict(
            :r5 => Ribbon(results[:ADAGRAD][:r5], "r = .5"),
            :r7 => Ribbon(results[:ADAGRAD][:r7], "r = .7"),
            :r9 => Ribbon(results[:ADAGRAD][:r9], "r = .9")
        ),
        :MSPI => Dict(
            :r5 => Ribbon(results[:MSPI][:r5], "r = .5"),
            :r7 => Ribbon(results[:MSPI][:r7], "r = .7"),
            :r9 => Ribbon(results[:MSPI][:r9], "r = .9")
        )
    )
    ylim = (0, 10 * value(bestmean))
    plots = Dict(
        :SGD => plot(Figure(collect(values(ribbons[:SGD])), "SGD"); ylim=ylim, kw...),
        :ADAGRAD => plot(Figure(collect(values(ribbons[:ADAGRAD])), "ADAGRAD"); ylim=ylim, kw...),
        :MSPI => plot(Figure(collect(values(ribbons[:MSPI])), "MSPI"); ylim=ylim, kw...),
    )
end

end #module
