## SARS-CoV-2 /COVID19
## Affan Shoukat, 2020

using DifferentialEquations, Plots, Parameters, DataFrames, CSV, LinearAlgebra
using StatsPlots, Query, Distributions, Statistics, Random, DelimitedFiles, ProgressMeter

∑(x) = sum(x) # synctactic sugar
heaviside(x) = x <= 0 ? 0 : 1

@with_kw mutable struct ModelParameters
    # default parameter values, (fixed: fixed parameters, sampled: sampled at time of run, input: given as input,i.e. scenarios)
    ## parameters for transmission dynamics.
    β::Float64 = 0.037 # (input) 0.037: R0 ~ 2.5, next generation method
    κ::Float64 = 0.5 # (fixed) reduction in transmission due to mild symptoms.
    σ::Float64 = 1/5.2 # (sampled)incubation period 5.2 days mean, LogNormal
    q::NTuple{4, Float64} = (0.05, 0.05, 0.05, 0.05) # (fixed) proportion of self-quarantine
    h::NTuple{4, Float64} = (0.02075, 0.02140, 0.025, 0.03885) ## (sampled), default values mean of distribution proporition of TOTAL infections going to hospital.
    fₐ::Float64 = 0.05 ##
    fᵢ::Float64 = 0.80 ##
    c::NTuple{4, Float64} = (0.0129, 0.03875, 0.0705, 0.15) ## (sampled), default values mean of distribution, mean is average of Chiense and US CDC.
    τₐ::Float64 = 0.5   ##
    τᵢ::Float64 = 1   ##
    γ::Float64 = 1/4.6 # (fixed) symptom onset to recovery, assumed fixed, based on serial interval... sampling creates a problem negative numbers
    δ::Float64 = 1/3.5 # (sampled), default value mean of distribution, symptom onset to hospitalization
    θ::NTuple{4, Float64} = (0.8, 0.8, 0.4, 0.2) ## (fixed)

    ## recovery and mortality
    mH::Float64 = 0.2296  ## prob of death in hospital
    μH::Float64 = 1/9.7   ## length of hospital stay before death (average of CDC reports)
    ψH::Float64 = 1/10    ## length of hospital stay before recovery (JAMA paper, send PR to CDC GitHub jama paper is median)
    mC::Float64 = 0.1396 ## prob of death in ICU
    μC::Float64 = 1/7     ## length of ICU stay before death Lancet Res "Clinical course and outcomes of critically ill Yang"
    ψC::Float64 = 1/13.25 ## length of ICU before recovery (Chad's ventilation calculation)

    ## internal parameters
    pop::NTuple{4, Float64} = (81982665,129596376,63157200,52431193)

end

function contact_matrix()
    ## contact matrix for general population and in household.
    M = ones(Float64, 4, 4)
    M[1, :] = [9.76190295449833 3.76760818611057 1.50823242972404 0.603972723406184]
    M[2, :] = [3.76760818611057 9.42964041341912 3.05467747979113 0.702942024329397]
    M[3, :] = [1.50823242972404 3.05467747979113 2.95716998311431 0.760587605942274]
    M[4, :] = [0.603972723406184 0.702942024329397 0.760587605942274 1.24948301635913]
    M̃ = ones(Float64, 4, 4)
    M̃[1, :] = [2.03651666746364 1.56405653862347 0.504076940943109 0.380678379808072]
    M̃[2, :] = [1.56405653862347 1.50876370897303 0.445015376096172 0.239252424523331]
    M̃[3, :] = [0.504076940943109 0.445015376096172 1.03746276893954 0.191098354935966]
    M̃[4, :] = [0.380678379808072 0.239252424523331 0.191098354935966 0.641792954732621]
    return M, M̃
end

function Model!(du, u, p, t)
    # model v4, with age structure, feb 20
    # 4 age groups: 0 - 18, 19 - 49, 50 - 65, 65+
    S₁, S₂, S₃, S₄,
    E₁, E₂, E₃, E₄,
    F₁, F₂, F₃, F₄,
    Iₙ₁, Iₙ₂, Iₙ₃, Iₙ₄,
    Qₙ₁, Qₙ₂, Qₙ₃, Qₙ₄,
    Iₕ₁, Iₕ₂, Iₕ₃, Iₕ₄,
    Qₕ₁, Qₕ₂, Qₕ₃, Qₕ₄,
    H₁, H₂, H₃, H₄,
    C₁, C₂, C₃, C₄,
    N₁, N₂, N₃, N₄,

    # internal incidence equations
    Z₁, Z₂, Z₃, Z₄,
    CX₁, CX₂, CX₃, CX₄,
    CY₁, CY₂, CY₃, CY₄,
    DX₁, DX₂, DX₃, DX₄,
    DY₁, DY₂, DY₃, DY₄,

    ## asymptomatic classes
    Aₙ₁, Aₙ₂, Aₙ₃, Aₙ₄,
    Aₛ₁, Aₛ₂, Aₛ₃, Aₛ₄,

    # set up the vectors for syntactic sugar
    S = (S₁, S₂, S₃, S₄)
    E = (E₁, E₂, E₃, E₄)
    F = (F₁, F₂, F₃, F₄)
    Iₙ = (Iₙ₁, Iₙ₂, Iₙ₃, Iₙ₄)
    Qₙ = (Qₙ₁, Qₙ₂, Qₙ₃, Qₙ₄)
    Iₕ = (Iₕ₁, Iₕ₂, Iₕ₃, Iₕ₄)
    Qₕ = (Qₕ₁, Qₕ₂, Qₕ₃, Qₕ₄)
    Aₙ = (Aₙ₁, Aₙ₂, Aₙ₃, Aₙ₄)
    Aₛ = (Aₛ₁, Aₛ₂, Aₛ₃, Aₛ₄)
    H = (H₁, H₂, H₃, H₄)
    C = (C₁, C₂, C₃, C₄)
    N = (N₁, N₂, N₃, N₄)

    # internal incidence euqations
    Z = (Z₁, Z₂, Z₃, Z₄)
    CX = (CX₁, CX₂, CX₃, CX₄)
    CY = (CY₁, CY₂, CY₃, CY₄)
    DX = (DX₁, DX₂, DX₃, DX₄)
    DY = (DY₁, DY₂, DY₃, DY₄)

    #get the contact matrix
    M, M̃ = contact_matrix()

    # constants
    pop = p.pop  ## fixed population
    Nᵥ = p.strat

    ## working in new model

    @unpack β, κ, ξ, ν, σ, q, h, fₐ, fᵢ, τₐ, τᵢ, γ, δ, θ, c, mH, μH, ψH, mC, μC, ψC = p
    for a = 1:4
        # sus S
        du[a+0] = -β*S[a]*(dot(M[a, :], Iₙ./pop) + dot(M[a, :], Iₕ./pop) + κ*dot(M[a, :], Aₙ./pop)  -
                   β*S[a]*(dot(M̃[a, :], Qₙ./pop) + dot(M̃[a, :], Qₕ./pop) + κ*dot(M̃[a, :], Aₛ./pop)
        # exposed E
        du[a+4]  = β*S[a]*(dot(M[a, :], Iₙ./pop) + dot(M[a, :], Iₕ./pop) + κ*dot(M[a, :], Aₙ./pop) +
                   β*S[a]*(dot(M̃[a, :], Qₙ./pop) + dot(M̃[a, :], Qₕ./pop) + κ*dot(M̃[a, :], Aₛ./pop) -
                   σ*E[a]
        # In class
        du[a+8] = (1 - θ[a])*(1 - q[a])*(1 - h[a])*σ*E[a]  - (1 - fᵢ)*γ*Iₙ[a] - fᵢ*τᵢ*Iₙ[a]
        # Qn class
        du[a+12] = (1 - θ[a])*q[a]*(1 - h[a])*σ*E[a] + fᵢ*τᵢ*Iₙ[a] - γ*Qₙ[a]
        # Ih class
        du[a+16] = (1 - θ[a])*(1 - q[a])*h[a]*σ*E[a] - (1 - fᵢ)*δ*Iₕ[a] - fᵢ*τᵢ*Iₕ[a]
        # Qh class
        du[a+20] = (1 - θ[a])*q[a]*h[a]*σ*E[a] + fᵢ*τᵢ*Iₕ[a] - δ*Qₕ[a]
        # Ha class
        du[a+24] = (1 - c[a])*(1 - fᵢ)*δ*Iₕ[a] + (1 - c[a])*δ*Qₕ[a] - (mH*μH + (1 - mH)*ψH)*H[a]
        # Ca class
        du[a+28] = c[a]*(1 - fᵢ)*δ*Iₕ[a] + c[a]*δ*Qₕ[a] - (mC*μC + (1 - mC)*ψC)*C[a]
        # Na class
        du[a+32] = -mC*μC*C[a] - mH*μH*H[a]

        # Z class to collect cumulative incidence (only from suspectibles)
        du[a+36] =  β*S[a]*(dot(M[a, :], Iₙ./pop) + dot(M[a, :], Iₕ./pop) + κ*dot(M[a, :], Aₙ./pop)  +
                    β*S[a]*(dot(M̃[a, :], Qₙ./pop) + dot(M̃[a, :], Qₕ./pop) + κ*dot(M̃[a, :], Aₛ./pop)
        ## collect cumulative incidence of hospitalization and ICU class 69 to 84
        du[a+40] = (1 - c[a])*(1 - fᵢ)*δ*Iₕ[a] + (1 - c[a])*δ*Qₕ[a]
        du[a+44] = c[a]*(1 - fᵢ)*δ*Iₕ[a] + c[a]*δ*Qₕ[a]
        du[a+48] = (mH*μH)*H[a]  ## DX
        du[a+52] = (mC*μC)*C[a]  ## DY

        ## Asymptomatic classes
        # Aₙ₁, Aₙ₂, Aₙ₃, Aₙ₄,
        du[a+56] = θ[a]*σ*(F[a] + E[a]) - (1 - fₐ)*γ*Aₙ[a] - fₐ*τₐ*Aₙ[a]
        # Aₛ₁, Aₛ₂, Aₛ₃, Aₛ₄,
        du[a+60] = fₐ*τₐ*Aₙ[a] - γ*Aₛ[a]
    end
end

function run_model(p::ModelParameters, nsims=1)
    ## set up ODE time and initial conditions
    tspan = (0.0, 1000.0)
    u0 = zeros(Float64, 64) ## total compartments
    sols = []
    for mc = 1:nsims
        ## reset the initial conditions
        u0[1] = u0[33] = p.pop[1]
        u0[2] = u0[34] = p.pop[2]
        u0[3] = u0[35] = p.pop[3]
        u0[4] = u0[36] = p.pop[4]
        # initial infected person
        u0[13] = 1

        # sample the parameters needed per simulation
        p.δ = 1/(rand(Uniform(2, 5)))
        p.σ = 1/(rand(LogNormal(log(5.2), 0.1)))
        p.h = (rand(Uniform(0.0085, 0.033)), rand(Uniform(0.0088, 0.034)), rand(Uniform(0.01, 0.042)), rand(Uniform(0.017, 0.066))) ./ (1 .- p.θ)
        p.c = (rand(Uniform(0.013, 0.015)), rand(Uniform(0.04, 0.044)), rand(Uniform(0.05, 0.1)), rand(Uniform(0.10, 0.20))) ## c dosnt change

        ## solve the ODE model
        #print("...sim: $mc, params: $(p.τ), $(p.f), $(p.nva), $(p.ν) \r")
        prob = ODEProblem(Model!, u0, tspan, p)
        sol = solve(prob, Rodas4(autodiff=false), dt=1.0, adaptive=false, callback=cb)  ## WORKS
        #sol = solve(prob, Rosenbrock23(autodiff=false),  dt=0.1, adaptive=false, callback=cbset)  ## WORKS
        push!(sols, sol)
    end
    #println("\n simulation scenario finished")
    return sols
end

function run_single()
    # A function that runs a single simulation for testing/calibrating purposes.
    ## setup parameters (overwrite some default ones if needed)
    p = ModelParameters()
    p.β = getβ("2.0")
    p.fₐ = 0.20
    p.τₐ = 1
    p.fᵢ = 0.80
    p.τᵢ = 1
    dump(p)
    sol = run_model(p, 1)[1] ## the [1] is required since runsims returns an array of solutions
    return sol
end

## hospital capacity: https://hcup-us.ahrq.gov/reports/statbriefs/sb185-Hospital-Intensive-Care-Units-2011.jsp
## https://www.aha.org/system/files/media/file/2020/01/2020-aha-hospital-fast-facts-new-Jan-2020.pdf
## https://www.sccm.org/Communications/Critical-Care-Statistics
