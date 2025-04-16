"
Defines a Gamma distribution with parameters α λ.
From a shape (α) and ratio (β) parametrization, we obtain our parametrization making λ = α/β
"
mutable struct GeneralizedBetaDistribution <: ScoreDrivenDistribution
    a::Union{Missing, Float64}
    b::Union{Missing, Float64}
    c::Union{Missing, Float64}
    p::Union{Missing, Float64}
    q::Union{Missing, Float64}
end

"
Outer constructor for the Normal distribution.
"
function GeneralizedBetaDistribution()
    return GeneralizedBetaDistribution(missing, missing, missing, missing, missing)
end

"
Gamma Function B(p, q)
"
function B(p, q)
    return SpecialFunctions.beta(p, q)
end

"
Evaluate the score of a Normal distribution with mean μ and variance σ², in observation y.
Colocar link aqui
"
function score_genbeta(a, b, c, p, q, y) 
  
    a, b, p, q, c = d.a, d.b, d.p, d.q, d.c
    if y <= a || y >= b
        return [NaN, NaN, NaN, NaN, NaN]
    else
        ∂a = (c*p - 1)/(x - a) + (c*(p + q) - 1)/(b - a)
        ∂b = (c*q - 1)/(b - y) - (c*(p + q) - 1)/(b - a)
        ∂p = c*log(x - a) - digamma(p) + digamma(p + q) - c*log(b - a)
        ∂q = c*log(b - y) - digamma(q) + digamma(p + q) - c*log(b - a)
        ∂c = 1/c + p*log(y - a) + q*log(b - y) - (p + q)*log(b - a)
        return [∂a, ∂b, ∂p, ∂q, ∂c]
    end
end

"
Evaluate the fisher information of a Normal distribution with mean μ and variance σ².
Colocar link aqui
"
function fisher_information_genbeta(a, b, c, p, q) 

    diff = b - a
    cpq = c*(p + q)
    I_aa = (cpq - 1)*(c*p - 1) / diff^2
    I_bb = (cpq - 1)*(c*q - 1) / diff^2
    I_pp = trigamma(p) - trigamma(p + q)
    I_qq = trigamma(q) - trigamma(p + q)
    I_cc = 1 / c^2
    I_ab = -(cpq - 1) / diff^2
    I_ap = c / diff
    I_aq = c / diff
    I_ac = p / diff
    I_bp = -c / diff
    I_bq = -c / diff
    I_bc = q / diff
    I_pq = -trigamma(p + q)
    I_pc = log(b - a) - mean(log(x - a))  # Requires data expectation
    I_qc = log(b - a) - mean(log(b - x))  # Requires data expectation

    return [I_aa  I_ab  I_ap  I_aq  I_ac;
            I_ab  I_bb  I_bp  I_bq  I_bc;
            I_ap  I_bp  I_pp  I_pq  I_pc;
            I_aq  I_bq  I_pq  I_qq  I_qc;
            I_ac  I_bc  I_pc  I_qc  I_cc]
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
"
function logpdf_genbeta(a, b, c, p, q, y)

    return logpdf_genbeta([a, b, c, p, q], y)
end

"
Evaluate the log pdf of a Normal distribution with mean μ and variance σ², in observation y.
    param[1] = α
    param[2] = λ
"
function logpdf_genbeta(param, y)

    a, b, p, q, c = param.a, param.b, param.p, param.q, param.c
    if y <= a || y >= b
        return -Inf
    else
        return log(c) + (c*p - 1)*log(y - a) + (c*q - 1)*log(b - y) - 
               logbeta(p, q) - (c*(p + q) - 1)*log(b - a)
end

"
Evaluate the cdf of a Gamma distribution with α,λ, in observation y.
"
function cdf_gamma(param::Vector{Float64}, y::Fl) where Fl

    if y <= a
        return 0.0
    elseif y >= b
        return 1.0
    else
        z = ((y - a) / (b - a))^c
        return incbeta(p, q, z)
    end
end

"
Returns the code of the Normal distribution. Is the key of DICT_CODE.
"
function get_dist_code(dist::GeneralizedBetaDistribution)
    return 5
end

"
Returns the number of parameters of the Normal distribution.
"
function get_num_params(dist::GeneralizedBetaDistribution)
    return 5
end

"
Simulates a value from a given Normal distribution.
    param[1] = λ
    param[2] = α  
"
function sample_dist(param::Vector{Float64}, dist::GeneralizedBetaDistribution)
    
    a, b, p, q, c = param[1], param[2], param[3], param[4], param[5]
    u = rand()
    z = invincbeta(p, q, u)
    return a + (b - a) * z^(1/c)
end

"
Indicates which parameters of the Normal distribution must be positive.
"
function check_positive_constrainst(dist::GeneralizedBetaDistribution)
    return [true, true, true, true, true]
end


function get_initial_α(y::Vector{Float64})

    T = length(y)
    model = JuMP.Model(Ipopt.Optimizer)
    set_silent(model)
    @variable(model, α >= 1e-2)  # Ensure α is positive
    @variable(model, λ[1:T] .>= 1e-4)
    register(model, :Γ, 1, Γ; autodiff = true)
    @NLobjective(model, Max, sum(-log(Γ(α)) - α*log(1/α) - α*log(λ[i]) +(α-1)*log(y[i]) - (α/λ[i])*y[i] for i in 1:T))
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
function get_initial_params(y::Vector{Fl}, time_varying_params::Vector{Bool}, dist::GammaDistribution, seasonality::Dict{Int64, Union{Bool, Int64}}) where Fl

    T         = length(y)
    dist_code = get_dist_code(dist)
    seasonal_period = get_num_harmonic_and_seasonal_period(seasonality)[2]

    initial_params = Dict()
    fitted_distribution = fit_mle(Gamma, y)
    
    # param[2] = λ = média
    if time_varying_params[1]
        initial_params[1] = y
    else
        initial_params[1] = fitted_distribution.α*fitted_distribution.θ
    end

    # param[1] = α
    if time_varying_params[2]
        initial_params[2] = get_seasonal_var(y, maximum(seasonal_period), dist)#(scaled_score.(y, ones(T) * var(diff(y)) , y, 0.5, dist_code, 2)).^2
    else
        initial_params[2] = get_initial_α(y)#mean(y)^2/var((y)) 
    end
    
    return initial_params
end
 
 
function get_seasonal_var(y::Vector{Fl}, seasonal_period::Int64, dist::GammaDistribution) where Fl

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
 