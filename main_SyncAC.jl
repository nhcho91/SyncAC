## Load Packages
using LinearAlgebra
using DifferentialEquations
using Optimization, OptimizationOptimJL
using Convex, Mosek, MosekTools
using Plots
using ControlSystems
using ComponentArrays

## Simulation Parameters
# Model Parameters
nₓ = 3
nᵤ = 1
nᵩ = 3

A = [-1.0189 0.9051 0.0
    0.8223 -1.0774 0.0
    1.0 0.0 0.0]
b = [-0.0022
    -0.1756
    0.0]
bᵣ = [0.0
    0.0
    -1.0]
c = [1.0
    0.0
    0.0]
θ = [-4.6839
    -9.8197
    1.0]
α_c = pi / 90.0
σ = 0.0233

Φ(x) = [x[1]
    x[2]
    exp(-(x[1] - α_c)^2 / (2.0 * σ^2))]

# Baseline Control Law Parameters
Q_base = diagm([200.0, 0.0, 100.0])
R_base = 1.0
kₘ = lqr(Continuous, A, b, Q_base, R_base)'
Aₘ = A - b * kₘ'
kᵣ = 0.0
bₘ = -b * kᵣ + bᵣ

# Reference Generator
function r_gen(t)
    r_bp = deg2rad.([0.0, 5.0, -5.0, 0.0, 2.5, -2.5, 0.0])
    t_bp = [1.0, 11.0, 22.0, 41.0, 51.0, 62.0, 100.0]
    i_bp = minimum(findall(x -> x > 0, t_bp .- t))
    return r_bp[i_bp]
end

# Adaptation Law Parameters
struct OptimJL
    i_OPT::Int64
end
struct CVX
    i_OPT::Int64
end

OPT = OptimJL(1) 
# OPT = CVX(2)
i_J = "J_pert"
# i_J = "J_input"

i_J == "J_pert" ? W = diagm([1.0, 1.0, 1.0]) : W = diagm([1.0, 1.0, 1.0, 1.0])
norm_p = 2
kₚ_list = [0.0, 1.0, 10.0, 100.0]
μ_list = [0.0, 0.5, 1.0]
Γ = diagm([400.0, 400.0, 20.0])
Q = diagm([1.0, 800.0, 0.1])

## Run Simulation
Fig_dir = "./Simulation/Figures"
if ~isdir(Fig_dir)
    mkdir(Fig_dir)
    mkdir(string(Fig_dir,"/coll"))
end

f_list = Matrix(undef, length(kₚ_list), length(μ_list))
for i_kₚ in eachindex(kₚ_list)
    for i_μ in eachindex(μ_list)
        kₚ = kₚ_list[i_kₚ]
        μ = μ_list[i_μ]
        P = lyap(Aₘ' - kₚ * I(nₓ), Q)        

        if typeof(OPT) == OptimJL
            function J_opt(u_opt, p_opt)
                u_c = u_opt[nᵤ]
                Uₘ = u_opt[nᵤ+1:nᵤ+nₓ]
                i_J = p_opt.i_J

                if i_J == "J_pert"
                    return norm(W * (μ * b * u_c + (1.0 - μ) * Uₘ), norm_p)
                elseif i_J == "J_input"
                    return norm(W * [u_c; Uₘ], norm_p) # norm(u_opt, 2)
                else
                    return norm(u_c, norm_p)
                end
            end

            function cons_opt(res, u_opt, p_opt)
                u_c = u_opt[nᵤ]
                Uₘ = u_opt[nᵤ+1:nᵤ+nₓ]
                U_c = p_opt.U_c
                return res .= U_c - (Uₘ - b * u_c)
            end

            f_opt = OptimizationFunction(J_opt, Optimization.AutoForwardDiff(), cons=cons_opt)
            U_c = 0.1 * ones(nₓ)
            u0_opt = [0.0; U_c]
            p_opt = ComponentArray(U_c=U_c, i_J=i_J)
            prob_opt = OptimizationProblem(f_opt, u0_opt, p_opt, lcons=zeros(nₓ), ucons=zeros(nₓ))
            
            function save_output_CL(p, OPT::OptimJL)
                return function (u, t, integrator)
                    # (; x, xₘ, θ̂) = u
                    x = u.x
                    xₘ = u.xₘ
                    θ̂ = u.θ̂
                    (kₘ, Γ, kₚ) = p

                    r = r_gen(t)
                    Δ = dot(Φ(x), θ) # Φ(x)' * θ
                    e = xₘ - x
                    U_c = -kₚ * e

                    u0_opt[2:4] = U_c
                    p_opt.U_c = U_c
                    p_opt.i_J = i_J
                    prob_alloc = remake(prob_opt, u0=u0_opt, p=p_opt)
                    sol_opt = solve(prob_alloc, IPNewton())
                    u_c = sol_opt.u[nᵤ]
                    Uₘ = sol_opt.u[nᵤ+1:nᵤ+nₓ]

                    u_base = -dot(kₘ, x) # -kₘ' * x
                    u_ad = -dot(Φ(x), θ̂) # -Φ(x)' * θ̂
                    u = u_base + u_ad + u_c
                    dx = A * x + b * (u + Δ) + bᵣ * r
                    dxₘ = Aₘ * xₘ + bₘ * r + Uₘ
                    dθ̂ = -Γ * Φ(x) * e' * P * b

                    V_e  = norm(e, 2)
                    V_θ  = norm(θ̂ - θ, 2)
                    V_dθ̂ = norm(dθ̂, 2)

                    return ComponentArray(
                        r=r,
                        Δ=Δ,
                        e=e,
                        U_c=U_c,
                        u_c=u_c,
                        Uₘ=Uₘ,
                        u_base=u_base,
                        u_ad=u_ad,
                        u=u,
                        dx=dx,
                        dxₘ=dxₘ,
                        dθ̂=dθ̂,
                        V_e=V_e,
                        V_θ=V_θ,
                        V_dθ̂=V_dθ̂
                    )
                end
            end

        elseif typeof(OPT) == CVX
            u_c_cvx = Variable(1)
            Uₘ_cvx = Variable(3)

            function J_opt(i_J)
                if i_J == "J_pert"
                    return norm(W * (μ * b * u_c + (1.0 - μ) * Uₘ), norm_p)
                elseif i_J == "J_input"
                    return norm(W * [u_c; Uₘ], norm_p) # norm(u_opt, 2)
                else
                    return norm(u_c, norm_p) # norm(Uₘ, norm_p)
                end
            end

            objective = J_opt(i_J)
 
            function save_output_CL(p, OPT::CVX)
                return function (u, t, integrator)
                    # (; x, xₘ, θ̂) = u
                    x = u.x
                    xₘ = u.xₘ
                    θ̂ = u.θ̂
                    (kₘ, Γ, kₚ) = p

                    r = r_gen(t)
                    Δ = dot(Φ(x), θ) # Φ(x)' * θ
                    e = xₘ - x
                    U_c = -kₚ * e

                    constraints = [U_c == Uₘ_cvx - b * u_c_cvx]
                    problem = minimize(objective, constraints)
                    Convex.solve!(problem, Mosek.Optimizer; silent_solver=true)
                    u_c = evaluate(u_c_cvx)
                    Uₘ = evaluate(Uₘ_cvx)

                    u_base = -dot(kₘ, x) # -kₘ' * x
                    u_ad = -dot(Φ(x), θ̂) # -Φ(x)' * θ̂
                    u = u_base + u_ad + u_c
                    dx = A * x + b * (u + Δ) + bᵣ * r
                    dxₘ = Aₘ * xₘ + bₘ * r + Uₘ
                    dθ̂ = -Γ * Φ(x) * e' * P * b

                    V_e  = norm(e, 2)
                    V_θ  = norm(θ̂ - θ, 2)
                    V_dθ̂ = norm(dθ̂, 2)

                    return ComponentArray(
                        r=r,
                        Δ=Δ,
                        e=e,
                        U_c=U_c,
                        u_c=u_c,
                        Uₘ=Uₘ,
                        u_base=u_base,
                        u_ad=u_ad,
                        u=u,
                        dx=dx,
                        dxₘ=dxₘ,
                        dθ̂=dθ̂,
                        V_e=V_e,
                        V_θ=V_θ,
                        V_dθ̂=V_dθ̂
                    )
                end
            end
        end

        function sys_CL!(du, u, p, t)
            output = save_output_CL(p, OPT)(u, t, [])
            du.x = output.dx
            du.xₘ = output.dxₘ
            du.θ̂ = output.dθ̂
        end

        u0 = ComponentArray(x=zeros(3), xₘ=zeros(3), θ̂=zeros(3))
        tspan = (0.0, 80.0)
        p = (; kₘ=kₘ, Γ=Γ, kₚ=kₚ)
        prob = ODEProblem(sys_CL!, u0, tspan, p)
        dt_save = 1e-2
        saved_values = SavedValues(Float64, typeof(save_output_CL(p, OPT)(u0, 0.0, [])))
        cb = SavingCallback(save_output_CL(p, OPT), saved_values, saveat=tspan[1]:dt_save:tspan[end])
        sol = solve(prob, Tsit5(), saveat=dt_save, reltol=1e-6, callback=cb)

        ## Plot Results
        state_names = propertynames(sol.u[1])
        output_names = propertynames(saved_values.saveval[1])
        states = Vector(undef, length(state_names))
        outputs = Vector(undef, length(output_names))

        for i in eachindex(state_names)
            states[i] = state_names[i] => hcat([sol.u[j][state_names[i]] for j in eachindex(sol.t)]...)'
        end
        for i in eachindex(output_names)
            outputs[i] = output_names[i] => hcat([saved_values.saveval[j][output_names[i]] for j in eachindex(saved_values.t)]...)'
        end
        sim = Dict([:t => sol.t; states...; outputs...])
        
        font_size    = 13
        title_string = "\$k_{P}=$(kₚ), \\mu=$(μ)\$"
        label_α      = ["\$\\alpha\$" "\$\\alpha_{m}\$" "\$\\alpha_{cmd}\$"]
        label_q      = ["\$q\$" "\$q_{m}\$"]
        label_u      = ["\$u\$" "\$u_{base}\$" "\$u_{ad}\$" "\$u_{c}\$"]
        label_Δ      = ["\$\\Delta\$" "\$-u_{ad}\$" "\$-u_{ad}-u_{c}\$"]
        label_Δ_aug  = ["\$\\Delta\$" "\$-u_{ad}\$" "\$-u_{c}\$" "\$-u_{ad}-u_{c}\$"]

        f_α = plot(sim[:t], rad2deg.([sim[:x][:, 1] sim[:xₘ][:, 1] sim[:r]]), xlabel="\$t\$ [s]", ylabel="\$\\alpha\$ [deg]", label=:false, ylims=(-5.4, 6.4), guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        if i_kₚ == 1 && i_μ == 1
            for j in eachindex(f_α.series_list)
                f_α.series_list[j].plotattributes[:label] = label_α[j]
            end
        end
        display(f_α)
        savefig(f_α, string(Fig_dir, "/Fig_alpha_$(i_kₚ)_$(i_μ).pdf"))

        f_q = plot(sim[:t], rad2deg.([sim[:x][:, 2] sim[:xₘ][:, 2]]), xlabel="\$t\$ [s]", ylabel="\$q\$ [deg/s]", ylims=(-6.5, 8.6), label=:false, guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        if i_kₚ == 1 && i_μ == 1
            for j in eachindex(f_q.series_list)
                f_q.series_list[j].plotattributes[:label] = label_q[j]
            end
        end
        display(f_q)
        savefig(f_q, string(Fig_dir, "/Fig_q_$(i_kₚ)_$(i_μ).pdf"))

        f_e_α = plot(sim[:t], [sim[:x][:, 3] sim[:xₘ][:, 3]], xlabel="\$t\$ [s]", ylabel="\$e_{\\alpha_{I}}\$", label=["\$e_{\\alpha_{I}}\$" "\$e_{\\alpha_{I}}_{m}\$"], guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        display(f_e_α)
        savefig(f_e_α, string(Fig_dir, "/Fig_e_alpha_$(i_kₚ)_$(i_μ).pdf"))

        f_e = plot(sim[:t], sim[:e], xlabel="\$t\$ [s]", ylabel=["\$e_{1}\$" "\$e_{2}\$" "\$e_{3}\$"], label=:false, layout=(3, 1), guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        display(f_e)
        savefig(f_e, string(Fig_dir, "/Fig_e_$(i_kₚ)_$(i_μ).pdf"))

        f_u = plot(sim[:t], [sim[:u] sim[:u_base] sim[:u_ad] sim[:u_c]], xlabel="\$t\$ [s]", ylabel="\$\\delta_{e}\$ [deg]", ylims=(-1.5, 2.0), label=:false, guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        if i_kₚ == 1 && i_μ == 1
            for j in eachindex(f_u.series_list)
                f_u.series_list[j].plotattributes[:label] = label_u[j]
            end
        end
        display(f_u)
        savefig(f_u, string(Fig_dir, "/Fig_u_$(i_kₚ)_$(i_μ).pdf"))

        f_U_c = plot(sim[:t], sim[:U_c], xlabel="\$t\$ [s]", ylabel=["\$U_{c_{1}}\$" "\$U_{c_{2}}\$" "\$U_{c_{3}}\$"], label=["\$U_{c_{1}}\$" "\$U_{c_{2}}\$" "\$U_{c_{3}}\$"], layout=(3, 1), guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        plot!(f_U_c, sim[:t], sim[:Uₘ], label=["\$U_{m_{1}}\$" "\$U_{m_{2}}\$" "\$U_{m_{3}}\$"])
        display(f_U_c)
        savefig(f_U_c, string(Fig_dir, "/Fig_U_c_$(i_kₚ)_$(i_μ).pdf"))

        f_θ = plot(sim[:t], [sim[:θ̂][:, 1] sim[:θ̂][:, 2] sim[:θ̂][:, 3]], xlabel="\$t\$ [s]", ylabel=["\$\\hat{\\theta}_{1}\$" "\$\\hat{\\theta}_{2}\$" "\$\\hat{\\theta}_{3}\$"], label=:false, layout=(3, 1), guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size) # label=["\$\\hat{\\theta}_{1}\$" "\$\\hat{\\theta}_{2}\$" "\$\\hat{\\theta}_{3}\$"]
        plot!(f_θ, sim[:t][[1, end]], [θ'; θ'], label=:false, linestyle=:dash) # label=["\$\\theta_{1}\$" "\$\\theta_{2}\$" "\$\\theta_{3}\$"]
        display(f_θ)
        savefig(f_θ, string(Fig_dir, "/Fig_theta_$(i_kₚ)_$(i_μ).pdf"))

        f_Δ = plot(sim[:t], [sim[:Δ] -sim[:u_ad] (-sim[:u_ad] - sim[:u_c])], xlabel="\$t\$ [s]", ylabel="\$\\Delta\$", ylims = (-1.5, 1.5), label=:false, legend = :bottomright, guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        if i_kₚ == 1 && i_μ == 1
            for j in eachindex(f_Δ.series_list)
                f_Δ.series_list[j].plotattributes[:label] = label_Δ[j]
            end
        end
        display(f_Δ)
        savefig(f_Δ, string(Fig_dir, "/Fig_Delta_$(i_kₚ)_$(i_μ).pdf"))

        f_Δ_aug = plot(sim[:t], [sim[:Δ] -sim[:u_ad] -sim[:u_c] (-sim[:u_ad] - sim[:u_c])], xlabel="\$t\$ [s]", ylabel="\$\\Delta\$", ylims = (-1.5, 1.5), label=:false, legend = :bottomright, guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        if i_kₚ == 1 && i_μ == 1
            for j in eachindex(f_Δ_aug.series_list)
                f_Δ_aug.series_list[j].plotattributes[:label] = label_Δ_aug[j]
            end
        end
        display(f_Δ_aug)
        savefig(f_Δ_aug, string(Fig_dir, "/Fig_Delta_aug_$(i_kₚ)_$(i_μ).pdf"))

        f_V_e = plot(sim[:t], sim[:V_e], xlabel="\$t\$ [s]", ylabel="\$||e|| \$", ylims = (0.0, 0.02), label=:false, guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        display(f_V_e)
        savefig(f_V_e, string(Fig_dir, "/Fig_V_e_$(i_kₚ)_$(i_μ).pdf"))

        f_V_θ = plot(sim[:t], sim[:V_θ], xlabel="\$t\$ [s]", ylabel="\$||\\tilde{\\theta}|| \$", ylims = (0.0, 11.0), label=:false, guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        display(f_V_θ)
        savefig(f_V_θ, string(Fig_dir, "/Fig_V_theta_$(i_kₚ)_$(i_μ).pdf"))

        f_V_dθ̂ = plot(sim[:t], sim[:V_dθ̂], xlabel="\$t\$ [s]", ylabel="\$|| \\dot{\\hat{\\theta}}|| \$", ylims = (0.0, 12.0), label=:false, guidefontsize = font_size, legendfontsize = font_size, title = title_string, titlefontsize = font_size)
        display(f_V_dθ̂)
        savefig(f_V_dθ̂, string(Fig_dir, "/Fig_V_dtheta_$(i_kₚ)_$(i_μ).pdf"))

        f_all = plot(f_α, f_q, f_u, layout=(3, 1), size=(1000, 1000), guidefontsize = font_size, legendfontsize = font_size)
        display(f_all)
        savefig(f_all, string(Fig_dir, "/Fig_all_$(i_kₚ)_$(i_μ).pdf"))

        f_list[i_kₚ, i_μ] = (; sim=sim, kₚ=kₚ, μ=μ, f_α=f_α, f_q=f_q, f_e_α=f_e_α, f_e=f_e, f_u=f_u, f_U_c=f_U_c, f_θ=f_θ, f_Δ=f_Δ, f_Δ_aug=f_Δ_aug, f_V_e=f_V_e, f_V_θ=f_V_θ, f_V_dθ̂=f_V_dθ̂, f_all=f_all)
    end
end

fig_size  = (1400,1400)
font_size = 13

f_α_coll = plot([f_list[i_kₚ, i_μ].f_α for i_kₚ in eachindex(kₚ_list) for i_μ in eachindex(μ_list)]..., guidefontsize = font_size, legendfontsize = font_size, left_margin=(3,:mm), size = fig_size, layout = (length(kₚ_list), length(μ_list)))
display(f_α_coll)
savefig(f_α_coll, string(Fig_dir, "/coll/Fig_alpha_coll.pdf"))

f_q_coll = plot([f_list[i_kₚ, i_μ].f_q for i_kₚ in eachindex(kₚ_list) for i_μ in eachindex(μ_list)]..., guidefontsize = font_size, legendfontsize = font_size, left_margin=(3,:mm), size = fig_size, layout = (length(kₚ_list), length(μ_list)))
display(f_q_coll)
savefig(f_q_coll, string(Fig_dir, "/coll/Fig_q_coll.pdf"))

f_u_coll = plot([f_list[i_kₚ, i_μ].f_u for i_kₚ in eachindex(kₚ_list) for i_μ in eachindex(μ_list)]..., guidefontsize = font_size, legendfontsize = font_size, left_margin=(3,:mm), size = fig_size, layout = (length(kₚ_list), length(μ_list)))
display(f_u_coll)
savefig(f_u_coll, string(Fig_dir, "/coll/Fig_u_coll.pdf"))

f_Δ_coll = plot([f_list[i_kₚ, i_μ].f_Δ for i_kₚ in eachindex(kₚ_list) for i_μ in eachindex(μ_list)]..., guidefontsize = font_size, legendfontsize = font_size, left_margin=(3,:mm), size = fig_size, layout = (length(kₚ_list), length(μ_list)))
display(f_Δ_coll)
savefig(f_Δ_coll, string(Fig_dir, "/coll/Fig_Delta_coll.pdf"))

f_Δ_aug_coll = plot([f_list[i_kₚ, i_μ].f_Δ_aug for i_kₚ in eachindex(kₚ_list) for i_μ in eachindex(μ_list)]..., guidefontsize = font_size, legendfontsize = font_size, left_margin=(3,:mm), size = fig_size, layout = (length(kₚ_list), length(μ_list)))
display(f_Δ_aug_coll)
savefig(f_Δ_aug_coll, string(Fig_dir, "/coll/Fig_Delta_aug_coll.pdf"))

f_V_e_coll = plot([f_list[i_kₚ, i_μ].f_V_e for i_kₚ in eachindex(kₚ_list) for i_μ in eachindex(μ_list)]..., guidefontsize = font_size, legendfontsize = font_size, left_margin=(3,:mm), size = fig_size, layout = (length(kₚ_list), length(μ_list)))
display(f_V_e_coll)
savefig(f_V_e_coll, string(Fig_dir, "/coll/Fig_V_e_coll.pdf"))

f_V_θ_coll = plot([f_list[i_kₚ, i_μ].f_V_θ for i_kₚ in eachindex(kₚ_list) for i_μ in eachindex(μ_list)]..., guidefontsize = font_size, legendfontsize = font_size, left_margin=(3,:mm), size = fig_size, layout = (length(kₚ_list), length(μ_list)))
display(f_V_θ_coll)
savefig(f_V_θ_coll, string(Fig_dir, "/coll/Fig_V_theta_coll.pdf"))

f_V_dθ̂_coll = plot([f_list[i_kₚ, i_μ].f_V_dθ̂ for i_kₚ in eachindex(kₚ_list) for i_μ in eachindex(μ_list)]..., guidefontsize = font_size, legendfontsize = font_size, left_margin=(4,:mm), size = fig_size, layout = (length(kₚ_list), length(μ_list)))
display(f_V_dθ̂_coll)
savefig(f_V_dθ̂_coll, string(Fig_dir, "/coll/Fig_V_dtheta_coll.pdf"))

## Test -- CVX
# using Convex, Mosek, MosekTools, LinearAlgebra
# Uc=0.1*ones(3)
# uc=Variable(1)
# Um=Variable(3)
# b=[-0.0022, -0.1756, 0.0]
# cons = [Uc==Um-b*uc]
# norm_p=2
# W=diagm([100.0,50.0,3.0])
# objective=norm(W*Um,norm_p)
# prob=minimize(objective,cons)
# Convex.solve!(prob, Mosek.Optimizer)

# Um_clf=W\(I(3) - W*b*pinv(W*b))*W*Uc
# Um_opt=evaluate(Um)

# uc_clf=-pinv(W*b)*W*Uc
# uc_opt=evaluate(uc)