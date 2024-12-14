
import Pkg; Pkg.add(["Parameters", "StateSpaceModels", "ARCHModels"])

using Distributions, LinearAlgebra, Random, Optim, JuMP, ForwardDiff, SpecialFunctions, CSV, DataFrames, Gurobi, Plots, UnPack

 include("src/structures.jl")
#     include("src/distributions/common.jl")
#     include("src/distributions/normal.jl")
#     include("src/distributions/t_location_scale.jl")
#     include("src/distributions/log_normal.jl")
     include("src/distributions/gamma.jl")
#     include("src/distributions/gamma_log_link.jl")
#     include("src/initialization.jl")
#     include("src/fit.jl")
#     include("src/utils.jl")
     include("src/components_dynamics.jl")
#     include("src/optimization.jl")
     include("src/forecast.jl")
#     include("src/residuals_diagnostics.jl")

include("src/UnobservedComponentsGAS.jl")
# include("src/utils.jl")

has_HS_seas = has_HS_seasonality(seasonality)

df = CSV.read("C:/Users/ilang/OneDrive/Documentos/Ilan/academia/2024.2/SDM/trabalho/code/data/input/ipeadata-consumo-energia-NE.csv", DataFrame)
orig_data = float.(df[:, 2])
tam =  length(orig_data)
y_train = orig_data[1:tam - 12]
y_test = orig_data[(tam - 11):tam]
y_all = orig_data

dist = Main.UnobservedComponentsGAS.GammaDistributionLogLink()  #.GammaDistributionLogLink()

time_varying_params = [true, false];
d                   = 0.0;
level               = ["random walk slope", ""];
seasonality         = ["stochastic 12", ""];
ar                  = Vector{Union{Missing}}([missing, missing])
#sample_robustness   = 1;

# has_seasonality(seasonality, 1) && !has_HS_seasonality(seasonality, 1)
# !isempty(seasonality[1]) && split(seasonality[1])[1] in ["HS", "Harrison Stevens", "Harrison & Stevens", "Harrison and Stevens"]

# initializing the models

modeld0 = UnobservedComponentsGAS.GASModel(dist, time_varying_params, 0.0, level, seasonality, ar)
modeld05 = UnobservedComponentsGAS.GASModel(dist, time_varying_params, 0.5, level, seasonality, ar)
modeld1 = UnobservedComponentsGAS.GASModel(dist, time_varying_params, 1.0, level, seasonality, ar)

# fitting models to train set

fitted_modeld0_train = UnobservedComponentsGAS.fit(modeld0, y_train; α = 0.0, robust = false, initial_values = missing);
fitted_modeld05_train = UnobservedComponentsGAS.fit(modeld05, y_train; α = 0.0, robust = false, initial_values = missing);
fitted_modeld1_train = UnobservedComponentsGAS.fit(modeld1, y_train; α = 0.0, robust = false, initial_values = missing);

fitted_modeld0_train.components["param_1"]["level"]["value"]

# predicting test set

steps_ahead    = 12;
num_scenarious = 1;

fitted_modeld0_test = UnobservedComponentsGAS.predict(modeld0, fitted_modeld0_train, y_train , steps_ahead, num_scenarious, [0.95])
fitted_modeld05_test = UnobservedComponentsGAS.predict(modeld05, fitted_modeld05_train, y_train , steps_ahead, num_scenarious, [0.95])
fitted_modeld1_test = UnobservedComponentsGAS.predict(modeld1, fitted_modeld1_train, y_train , steps_ahead, num_scenarious, [0.95])


fitted_modeld0_test[2]["params"][1, 539:550, 1]

# fitting models to full sample

fitted_modeld0 = UnobservedComponentsGAS.fit(modeld0, y_all; α = 0.0, robust = false, initial_values = missing);
fitted_modeld05 = UnobservedComponentsGAS.fit(modeld05, y_all; α = 0.0, robust = false, initial_values = missing);
fitted_modeld1 = UnobservedComponentsGAS.fit(modeld1, y_all; α = 0.0, robust = false, initial_values = missing);
fitted_modeld0_test
# train set eval
fitted_modeld0_train.components["param_1"]["slope"]["hyperparameters"]
function plot_model(y_in_sample, fit, y_validation, mean, upper_ci, lower_ci)
     T_in_sample = 1:length(y_in_sample)
     T_validation = length(y_in_sample)+1:(length(y_validation) + length(y_in_sample))
     println(length(y_in_sample), " ", length(y_validation), " ", length(y_in_sample)+length(y_validation))
     plot(T_in_sample,y_in_sample, label = "y_train")
     plot!(T_in_sample, fit, label = "fit in sample")
     plot!(T_validation, y_validation, label = "y_test, 12 steps ahead")
     # Add shaded confidence intervals
     plot!(T_validation, mean, label="prediction 12 steps ahead", color=:blue, alpha=0.2)
     plot!(T_validation, upper_ci, fill_between=(lower_ci), color=:blue, alpha=0.2, label="95% CI")
end

println(length(y_train), " ", length(y_test), " ", length(y_train)+length(y_test))

upper_95_model0 = fitted_modeld0_test[1]["intervals"]["95"]["upper"]
lower_95_model0 = fitted_modeld0_test[1]["intervals"]["95"]["lower"]
mean_model0 = fitted_modeld0_test[1]["mean"]
plot_model(y_train[2:end], fitted_modeld0_train.fit_in_sample[2:end], y_test, mean_model0, upper_95_model0, lower_95_model0)
savefig("modeld0_train_test.png")

plot(y_train)
plot!(539:550, y_test)
plot!(539:550, fitted_modeld0_test[1])

exp(2000)

plot!(fitted_modeld0_train.fitted_params["param_1"])

exp.(fitted_modeld0_train.fitted_params["param_1"])

sum(fitted_modeld1_train.components["param_1"]["seasonality"]["value"][1:12])

plot(y_train)
plot!(fitted_modeld0_train.fit_in_sample)

plot(y_train)
upper_95_model05 = fitted_modeld05_test["intervals"]["95"]["upper"]
lower_95_model05 = fitted_modeld05_test["intervals"]["95"]["lower"]
mean_model05 = fitted_modeld05_test["mean"]
plot_model(y_train, fitted_modeld05_train.fit_in_sample, y_test, mean_model05, upper_95_model05, lower_95_model05)

plot(fitted_modeld05_train.components["param_1"]["slope"]["value"])

fitted_modeld05_train.components["param_1"]["level"]["value"][1:5]
fitted_modeld05_train.components["param_1"]["level"]

l1 = fitted_modeld05_train.components["param_1"]["level"]["value"][1] + fitted_modeld05_train.components["param_1"]["seasonality"]["value"][1]
Φ = fitted_modeld05_train.components["param_2"]["level"]["value"][2] + fitted_modeld05_train.components["param_1"]["seasonality"]["value"][2]

scaled_score(parameters[t-1, 1], parameters[t-1, 2], y[t-1], 0.5, 4, 1)


upper_95_model1 = fitted_modeld1_test["intervals"]["95"]["upper"]
lower_95_model1 = fitted_modeld1_test["intervals"]["95"]["lower"]
mean_model1 = fitted_modeld1_test["mean"]
plot_model(y_train, fitted_modeld1_train.fit_in_sample, y_test, mean_model1, upper_95_model1, lower_95_model1)
savefig("modeld1_train_test111.png")


plot(y_train)
plot!(fitted_modeld05_train.fit_in_sample)
plot!(y_test)


plot(y_train)
plot!(fitted_modeld1_train.fit_in_sample)
plot!(y_test)
plot!(fitted_modeld1_test.fit_out_of_sample)


# full series

plot(y_all)
plot!(fitted_modeld0.fit_in_sample)

plot(y_all)
plot!(fitted_modeld05.fit_in_sample)

plot(y_all)
plot!(fitted_modeld1.fit_in_sample)

savefig("C:/Users/ilang/OneDrive/Documentos/Ilan/academia/2024.2/SDM/trabalho/HS_NE.png")

α = 0.8052321435188204
v1 = α^2 * (trigamma(α) - (1/α))
v2 = α

v2^(-0.5)

