module UnobservedComponentsGAS

    using JuMP
    using Ipopt
    using LinearAlgebra
    using Parameters
    using Statistics
    using Distributions
    using Random 
    using SpecialFunctions
    using StateSpaceModels
    # For residuals diagnostics
    using Plots
    using HypothesisTests
    using StatsPlots
    using StatsBase
    using ARCHModels
     
    # # include("../NonParametricStructuralModels/src/NonParametricStructuralModels.jl")

    include("structures.jl")
    include("distributions/common.jl")
    include("distributions/normal.jl")
    include("distributions/t_location_scale.jl")
    include("distributions/log_normal.jl")
    include("distributions/gamma.jl")
    include("distributions/gamma_log_link.jl")
    include("initialization.jl")
    include("fit.jl")
    include("utils.jl")
    include("components_dynamics.jl")
    include("optimization.jl")
    include("forecast.jl")
    include("residuals_diagnostics.jl")

    const DICT_CODE = Dict(1 => "Normal",
                           2 => "tLocationScale",
                           3 => "Gamma",
                           4 => "GammaLogLink")

    const DICT_SCORE = Dict("Normal"         => score_normal,
                            "tLocationScale" => score_tlocationscale,
                            "Gamma"          => score_gamma,
                            "GammaLogLink"   => score_gamma_log_link)

    const DICT_FISHER_INFORMATION = Dict("Normal"         => fisher_information_normal,
                                         "tLocationScale" => fisher_information_tlocationscale,
                                         "Gamma"          => fisher_information_gamma,
                                         "GammaLogLink"   => fisher_information_gamma_log_link)

    const DICT_LOGPDF = Dict("Normal"         => logpdf_normal,
                             "tLocationScale" => logpdf_tlocationscale,
                             "Gamma"          => logpdf_gamma,
                             "GammaLogLink"   => logpdf_gamma_log_link)

    const DICT_CDF = Dict("Normal"         => cdf_normal,
                          "tLocationScale" => cdf_tlocationscale,
                          "Gamma"          => cdf_gamma,
                          "GammaLogLink"   => cdf_gamma_log_link)
end 
