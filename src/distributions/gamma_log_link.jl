"
Defines a Gamma distribution with parameters α λ.
From a shape (α) and ratio (β) parametrization, we obtain our parametrization making λ = α/β
"
mutable struct GammaDistributionLogLink <: ScoreDrivenDistribution
    λ::Union{Missing, Float64}
    α::Union{Missing, Float64}
end

"
Outer constructor for the Normal distribution.
"
function GammaDistributionLogLink()
    return GammaDistributionLogLink(missing, missing)
end

"
Gamma Function Γ(x)
"
function Γ(x)
    return SpecialFunctions.gamma(x)
end
    

"Auxiliar function Ψ1(α)"
function Ψ1(α)
    return SpecialFunctions.digamma(α)
end


"Auxiliar function Ψ2(α)"
function Ψ2(α)
    return SpecialFunctions.trigamma(α)
end


"
Evaluate the score of a Normal distribution with mean μ and variance σ², in observation y.
Colocar link aqui
"
function score_gamma_log_link(λ, α, y) 
  
    α <= 0 ? α = 1e-2 : nothing
    
    eλ_frac = isinf(exp(λ)) ? 0.0 : (y/exp(λ)) 

    ∇_α =  log(y) - eλ_frac + log(α) - Ψ1(α) - λ + 1
    ∇_λ = α * (eλ_frac - 1)

    return [∇_λ; ∇_α]
end

"
Evaluate the fisher information of a Normal distribution with mean μ and variance σ².
Colocar link aqui
"
function fisher_information_gamma_log_link(λ, α) 

    α <= 0 ? α = 1e-2 : nothing
    eλ = isinf(exp(λ)) ? 1e6 : exp(λ)
    
    I_λ = α
    I_α = α

    return [I_λ 0; 0 I_α]
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
"
function logpdf_gamma_log_link(λ, α, y)

    return logpdf_gamma_log_link([λ, α], y)
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
    param[1] = α
    param[2] = λ
"
function logpdf_gamma_log_link(param, y)

    param[2] > 0 ? α = param[2] : α = 1e-2
    λ = param[1]

    member = isinf(exp(λ)) ? 0.0 : α/(exp(λ))
   
    return -log(Γ(α)) + α*log(α) + α*λ + α*log(y) - log(y) - member*y 
end

"
Evaluate the cdf of a Gamma distribution with α,λ, in observation y.
"
function cdf_gamma_log_link(param::Vector{Float64}, y::Fl) where Fl

    param[2] > 0 ? α = param[2] : α = 1e-2
    param[1] > 0 ? λ = param[1] : λ = 1e-2

    eλ = isinf(exp(λ)) ? 1e6 : exp(λ)

    return Distributions.cdf(Distributions.Gamma(α, eλ/α), y)
end

"
Returns the code of the Normal distribution. Is the key of DICT_CODE.
"
function get_dist_code(dist::GammaDistributionLogLink)
    return 4
end

"
Returns the number of parameters of the Normal distribution.
"
function get_num_params(dist::GammaDistributionLogLink)
    return 2
end

"
Simulates a value from a given Normal distribution.
    param[1] = λ
    param[2] = α  
"
function sample_dist(param::Vector{Float64}, dist::GammaDistributionLogLink)
    
    "A Gamma do pacote Distributions é parametrizada com shape α e scale θ"
    "Como θ = 1/β e β = α/λ, segue-se que θ = λ/α"
    param[2] > 0 ? α = param[2] : α = 1e-2
    λ = param[1]

    l = λ > 0 ? λ : 1e-2

    eλ = isinf(exp(λ)) ? 1e6 : exp(λ)
    
    return rand(Distributions.Gamma(α, l/α))
end

"
Indicates which parameters of the Normal distribution must be positive.
"
function check_positive_constrainst(dist::GammaDistributionLogLink)
    return [true, true]
end


function get_initial_α(y::Vector{Float64})

    T = length(y)
    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, α >= 1e-2)  # Ensure α is positive
    @variable(model, λ[1:T])
    register(model, :Γ, 1, Γ; autodiff = true)
    @NLobjective(model, Max, sum(-log(Γ(α)) - α*log(1/α) - α*λ[i] +(α-1)*log(y[i]) - (α/exp(λ[i]))*y[i] for i in 1:T))
    optimize!(model)
    if has_values(model)
        return JuMP.value.(α)
    else
        return fit_mle(Gamma, y).α
    end 
end

"
Returns a dictionary with the initial values of the parameters of Normal distribution that will be used in the model initialization.
"
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::GammaDistributionLogLink, seasonality::Dict{Int64, Union{Bool, Int64}}) where Fl

    T         = length(y)
    dist_code = get_dist_code(dist)
    seasonal_period = get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict()
    
    # param[2] = λ = média
    if time_varying_params[1]
        initial_params[1] = log.(y)
    end

    # param[1] = α
    if time_varying_params[2]
        initial_params[2] = get_seasonal_var(y, maximum(seasonal_period), dist)#(scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        initial_params[2] = get_initial_α(y)#mean(y)^2/var((y)) 
    end
    
    return initial_params
end
 
 
function get_seasonal_var(y::Vector{Fl}, seasonal_period::Int64, dist::GammaDistributionLogLink) where Fl

    num_periods = ceil(Int, length(y) / seasonal_period)
    seasonal_variances = zeros(Fl, length(y))
    
    for i in 1:seasonal_period
        month_data = y[i:seasonal_period:end]
        num_observations = length(month_data)
        if num_observations > 0
            g = Distributions.fit(Gamma, month_data)
            α = g.α
            θ = g.θ
            variance = α*(θ^2) 
            for j in 1:num_observations
                seasonal_variances[i + (j - 1) * seasonal_period] = variance
            end
        end
    end
    return seasonal_variances
end
 