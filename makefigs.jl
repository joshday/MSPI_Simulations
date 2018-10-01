module Figures
using OnlineStats, Plots, ProgressMeter, LossFunctions, PenaltyFunctions, Statistics,
    SparseRegression, Random
import Plots: plot

#-----------------------------------------------------------------------# data generators
beta(p) = collect(range(-1, stop=1, length=p))

function linregdata!(x, β = beta(size(x, 2)), σ = 1.0)
    randn!(x)
    y = x * β + σ .* randn(size(x, 1))
    x, y, β
end

function logregdata(x, β = beta(size(x, 2)))
	randn!(x)
	y = [2.0 * (rand() <= 1 / (1 + exp(-η))) - 1.0 for η in x * β]
	return x, y, β
end

data!(::DistanceLoss, x) = linregdata!(x)
data!(::MarginLoss, x) = logregdata!(x)

#-----------------------------------------------------------------------# Ribbon
struct Ribbon
    low::Vector{Float64}
    mid::Vector{Float64}
    hi::Vector{Float64}
end
function Ribbon(results::Matrix{Float64})
    mat = mapslices(x -> quantile(x, [.05, .5, .95]), results, dims=2)
    Ribbon(mat[:,1], mat[:,2], mat[:,3])
end
@recipe function f(o::Ribbon)
    ribbon --> (o.low, o.hi)
    fillalpha --> .1
    w --> 2
    label --> ""
    o.mid
end

struct Figure 
    ribbons
    title::String
end
@recipe function f(o::Figure)
    title --> o.title
    for r in o.ribbons
        @series begin r end
    end
end

#-----------------------------------------------------------------------# simulation
function simulation(n, p, nreps=100, seed=123; loss = .5 * L2DistLoss(), kw...)
    models = make_models(loss, p)
    results = make_results(models, n, nreps)
    stat = Series(models...)
    x = randn(n, p)
    @showprogress for r in 1:nreps 
        # generate data
        x, y, beta = data!(loss, x)
        # get offline solution 
        sm = SModel(x, y, loss, NoPenalty())
        learn!(sm, strategy(AdaptiveProxGrad(sm), MaxIter(200), Converged(coef)))
        best = value(sm)
        # online stuff
        stat_r = deepcopy(stat)
        for (i, xy) in enumerate(eachrow(x, y))
            fit!(stat_r, xy)
            for (s, res) in zip(stat_r.stats, results)
                res[i, r] = OnlineStats.objective(s, x, y) - best
            end
        end
    end
    ribbons = map(Ribbon, results)
    ymax = reduce((a,b) -> min(minimum(a), minimum(b)), results) * 5
    plots = (
        sgd = plot(Figure([ribbons.sgd_r5, ribbons.sgd_r7, ribbons.sgd_r9], "SGD"); kw...),
        adagrad = plot(Figure([ribbons.adagrad_r5, ribbons.adagrad_r7, ribbons.adagrad_r9], "ADAGRAD"); kw...),
        adadelta = plot(Figure([ribbons.adadelta], "ADADELTA"); kw...),
        rmsprop = plot(Figure([ribbons.rmsprop_r5, ribbons.rmsprop_r7, ribbons.rmsprop_r9], "RMSPROP"); kw...),
        adam = plot(Figure([ribbons.adam_r5, ribbons.adam_r7, ribbons.adam_r9], "ADAM"); kw...),
        mspi = plot(Figure([ribbons.mspi_r5, ribbons.mspi_r7, ribbons.mspi_r9], "MSPI"); kw...),
    )
    for (nm, plt) in pairs(plots)
        png(plt, "output/$(nm)_$(loss)_.png")
    end
end


function make_models(loss, p::Int)
    (
        sgd_r5 = StatLearn(p, loss, SGD(); rate = LearningRate(.5)),
        sgd_r7 = StatLearn(p, loss, SGD(); rate = LearningRate(.7)),
        sgd_r9 = StatLearn(p, loss, SGD(); rate = LearningRate(.9)),
        # ADAGRAD
        adagrad_r5 = StatLearn(p, loss, ADAGRAD(); rate = LearningRate(.5)),
        adagrad_r7 = StatLearn(p, loss, ADAGRAD(); rate = LearningRate(.7)),
        adagrad_r9 = StatLearn(p, loss, ADAGRAD(); rate = LearningRate(.9)),
        # ADADELTA
        adadelta = StatLearn(p, loss, ADADELTA()),
        # RMSPROP
        rmsprop_r5 = StatLearn(p, loss, RMSPROP(); rate = LearningRate(.5)),
        rmsprop_r7 = StatLearn(p, loss, RMSPROP(); rate = LearningRate(.7)),
        rmsprop_r9 = StatLearn(p, loss, RMSPROP(); rate = LearningRate(.9)),
        # ADAM 
        adam_r5 = StatLearn(p, loss, ADAM(); rate = LearningRate(.5)),
        adam_r7 = StatLearn(p, loss, ADAM(); rate = LearningRate(.7)),
        adam_r9 = StatLearn(p, loss, ADAM(); rate = LearningRate(.9)),
        # MSPI
        mspi_r5 = StatLearn(p, loss, MSPI(); rate = LearningRate(.5)),
        mspi_r7 = StatLearn(p, loss, MSPI(); rate = LearningRate(.7)),
        mspi_r9 = StatLearn(p, loss, MSPI(); rate = LearningRate(.9)),
    )
end
make_results(models, n, nreps) = map(x -> Matrix{Float64}(undef, n, nreps), models)


#-----------------------------------------------------------------------# simulation settings 
n, p = 1000, 10

@info "Linear Regression Example"
simulation(n, p, loss = .5 * L2DistLoss(), ylim = (0, .6))


# #-----------------------------------------------------------------------# simulation
# function simulation(n, p, nreps=100, seed=123; loss = .5 * L2DistLoss(), kw...)
#     results = Dict(
#         :SGD => Dict(
#             :r5 => Matrix{Float64}(undef, n, nreps),
#             :r7 => Matrix{Float64}(undef, n, nreps),
#             :r9 => Matrix{Float64}(undef, n, nreps)
#         ),
#         :ADAGRAD => Dict(
#             :r5 => Matrix{Float64}(undef, n, nreps),
#             :r7 => Matrix{Float64}(undef, n, nreps),
#             :r9 => Matrix{Float64}(undef, n, nreps)
#         ),
#         :ADADELTA => Matrix{Float64}(undef, n, nreps),
#         :RMSPROP => Dict(
#             :r5 => Matrix{Float64}(undef, n, nreps),
#             :r7 => Matrix{Float64}(undef, n, nreps),
#             :r9 => Matrix{Float64}(undef, n, nreps)
#         ),
#         :MSPI => Dict(
#             :r5 => Matrix{Float64}(undef, n, nreps),
#             :r7 => Matrix{Float64}(undef, n, nreps),
#             :r9 => Matrix{Float64}(undef, n, nreps)
#         )
#     )

#     bestmean = Mean()

#     @showprogress for r in 1:nreps
#         x, y, beta = data(loss, n, p)
#         smod = SModel(x, y, loss, NoPenalty())
#         learn!(smod; verbose=false)
#         best = value(smod)
#         fit!(bestmean, best)
#         sgd = Series(
#             StatLearn(p, loss, NoPenalty(), SGD(); rate=LearningRate(.5)),
#             StatLearn(p, loss, NoPenalty(), SGD(); rate=LearningRate(.7)),
#             StatLearn(p, loss, NoPenalty(), SGD(); rate=LearningRate(.9)),
#         )
#         ada = Series(
#             StatLearn(p, loss, NoPenalty(), ADAGRAD(); rate=LearningRate(.5)),
#             StatLearn(p, loss, NoPenalty(), ADAGRAD(); rate=LearningRate(.7)),
#             StatLearn(p, loss, NoPenalty(), ADAGRAD(); rate=LearningRate(.9)),
#             StatLearn(p, loss, NoPenalty(), ADADELTA())
#         )
#         rms = Series(
#             StatLearn(p, loss, NoPenalty(), RMSPROP(); rate=LearningRate(.5)),
#             StatLearn(p, loss, NoPenalty(), RMSPROP(); rate=LearningRate(.7)),
#             StatLearn(p, loss, NoPenalty(), RMSPROP(); rate=LearningRate(.9)),
#         )
#         mspi = Series(
#             StatLearn(p, loss, NoPenalty(), MSPI(); rate=LearningRate(.5)),
#             StatLearn(p, loss, NoPenalty(), MSPI(); rate=LearningRate(.7)),
#             StatLearn(p, loss, NoPenalty(), MSPI(); rate=LearningRate(.9)),
#         )
#         for i in 1:n
#             ob = @view(x[i, :]), y[i]
#             fit!(sgd, ob)
#             fit!(ada, ob)
#             fit!(rms, ob)
#             fit!(mspi, ob)
#             results[:SGD][:r5][i, r]        = OnlineStats.objective(sgd.stats[1], x, y) - best
#             results[:SGD][:r7][i, r]        = OnlineStats.objective(sgd.stats[2], x, y) - best
#             results[:SGD][:r9][i, r]        = OnlineStats.objective(sgd.stats[3], x, y) - best
#             results[:ADAGRAD][:r5][i, r]    = OnlineStats.objective(ada.stats[1], x, y) - best
#             results[:ADAGRAD][:r7][i, r]    = OnlineStats.objective(ada.stats[2], x, y) - best
#             results[:ADAGRAD][:r9][i, r]    = OnlineStats.objective(ada.stats[3], x, y) - best
#             results[:ADADELTA][i, r]        = OnlineStats.objective(ada.stats[4], x, y) - best
#             results[:RMSPROP][:r5][i, r]    = OnlineStats.objective(rms.stats[1], x, y) - best
#             results[:RMSPROP][:r7][i, r]    = OnlineStats.objective(rms.stats[2], x, y) - best
#             results[:RMSPROP][:r9][i, r]    = OnlineStats.objective(rms.stats[3], x, y) - best
#             results[:MSPI][:r5][i, r]       = OnlineStats.objective(mspi.stats[1], x, y) - best
#             results[:MSPI][:r7][i, r]       = OnlineStats.objective(mspi.stats[2], x, y) - best
#             results[:MSPI][:r9][i, r]       = OnlineStats.objective(mspi.stats[3], x, y) - best
#         end
#     end
#     ribbons = Dict(
#         :SGD => Dict(
#             :r5 => Ribbon(results[:SGD][:r5], "r = .5"),
#             :r7 => Ribbon(results[:SGD][:r7], "r = .7"),
#             :r9 => Ribbon(results[:SGD][:r9], "r = .9")
#         ),
#         :ADAGRAD => Dict(
#             :r5 => Ribbon(results[:ADAGRAD][:r5], "r = .5"),
#             :r7 => Ribbon(results[:ADAGRAD][:r7], "r = .7"),
#             :r9 => Ribbon(results[:ADAGRAD][:r9], "r = .9")
#         ),
#         :RMSPROP => Dict(
#             :r5 => Ribbon(results[:RMSPROP][:r5], "r = .5"),
#             :r7 => Ribbon(results[:RMSPROP][:r7], "r = .7"),
#             :r9 => Ribbon(results[:RMSPROP][:r9], "r = .9")
#         ),
#         :MSPI => Dict(
#             :r5 => Ribbon(results[:MSPI][:r5], "r = .5"),
#             :r7 => Ribbon(results[:MSPI][:r7], "r = .7"),
#             :r9 => Ribbon(results[:MSPI][:r9], "r = .9")
#         )
#     )
#     ylim = (0, 10 * value(bestmean))
#     plots = Dict(
#         :SGD => plot(Figure(collect(values(ribbons[:SGD])), "SGD"); ylim=ylim, kw...),
#         :ADAGRAD => plot(Figure(collect(values(ribbons[:ADAGRAD])), "ADAGRAD"); ylim=ylim, kw...),
#         :RMSPROP => plot(Figure(collect(values(ribbons[:RMSPROP])), "RMSPROP"); ylim=ylim, kw...),
#         :MSPI => plot(Figure(collect(values(ribbons[:MSPI])), "MSPI"); ylim=ylim, kw...),
#     )
# end

end #module
