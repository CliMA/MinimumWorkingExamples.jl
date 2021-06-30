using PyPlot
mutable struct Single_Stack
    L::Float64
    N::Int64
    t::Float64

    ρ::Array{Float64,1}
    w::Array{Float64,1}
    ρθ::Array{Float64,1}

    _grav::Float64

    Π::Function


end

# compute tendency
function spatial_residual(ss::Single_Stack, ρ, w, ρθ; t = 0.0)


    N, L = ss.N, ss.L
    Δz = L/N
    _grav = ss._grav
    Π = ss.Π

    ρ_res, w_res, ρθ_res = copy(ρ), copy(w), copy(ρθ)

    for i = 1:N
        # ρ₊ = (i == N ? ρ[i] : (ρ[i] + ρ[i+1])/2.0)
        # ρ₋ = (i == 1 ? ρ[i] : (ρ[i] + ρ[i-1])/2.0)

        ρ₊ = (i == N ? 0.0 : (ρ[i] + ρ[i+1])/2.0)
        ρ₋ = (i == 1 ? 0.0 : (ρ[i] + ρ[i-1])/2.0)

        ρ_res[i] = -(w[i+1]ρ₊ - w[i]ρ₋)/Δz


        # ρθ₊ = (i == N ? ρθ[i] : (ρθ[i] + ρθ[i+1])/2.0)
        # ρθ₋ = (i == 1 ? ρθ[i] : (ρθ[i] + ρθ[i-1])/2.0)

        ρθ₊ = (i == N ? 0.0 : (ρθ[i] + ρθ[i+1])/2.0)
        ρθ₋ = (i == 1 ? 0.0 : (ρθ[i] + ρθ[i-1])/2.0)

        ρθ_res[i] = -(w[i+1]ρθ₊ - w[i]ρθ₋)/Δz
    end

    w_res[1] = w_res[N+1] = 0
    for i = 2:N
        w_res[i] = -1/2*(ρθ[i-1]/ρ[i-1] + ρθ[i]/ρ[i]) *(Π(ρθ[i]) - Π(ρθ[i-1]))/Δz - _grav
    end


    return ρ_res,w_res, ρθ_res
end

# update with RK4
function temporal_update!(ss::Single_Stack, Δt::Float64)
    ρ, w, ρθ = ss.ρ, ss.w, ss.ρθ


    ρ_res1, w_res1, ρθ_res1 = spatial_residual(ss::Single_Stack, ρ, w, ρθ; t = ss.t)
    ρ_res2, w_res2, ρθ_res2 = spatial_residual(ss::Single_Stack, ρ + Δt/2*ρ_res1, w + Δt/2*w_res1, ρθ + Δt/2*ρθ_res1; t = ss.t+Δt/2)
    ρ_res3, w_res3, ρθ_res3 = spatial_residual(ss::Single_Stack, ρ + Δt/2*ρ_res2, w + Δt/2*w_res2, ρθ + Δt/2*ρθ_res2; t = ss.t+Δt/2)
    ρ_res4, w_res4, ρθ_res4 = spatial_residual(ss::Single_Stack, ρ + Δt*ρ_res3, w + Δt*w_res3, ρθ + Δt*ρθ_res3; t = ss.t+Δt)


    ρ  .+=  Δt/6*(ρ_res1  + 2ρ_res2  + 2ρ_res3  + ρ_res4)
    w  .+=  Δt/6*(w_res1  + 2w_res2  + 2w_res3  + w_res4)
    ρθ .+=  Δt/6*(ρθ_res1 + 2ρθ_res2 + 2ρθ_res3 + ρθ_res4)

    ss.t += Δt
    @info "ρ ∈ ",   minimum(ρ), maximum(ρ)
    @info "w ∈ ",   minimum(w), maximum(w)
    @info "ρθ ∈ ",  minimum(ρθ), maximum(ρθ)



end


function compute_wave_speed(ρ, w, ρθ, _R_m, _C_p, γ, _MSLP)
    p =  ( ρθ * _R_m / _MSLP^(_R_m/_C_p) ).^(γ)
    c = sqrt.(γ*p./ρ)

    return maximum(w) + maximum(c)
end


# Decaying temperature profile
function DecayingTemperatureProfile(z::Float64; T_virt_surf, T_min_ref, _R_d, _grav, _MSLP)
    # Scale height for surface temperature
    H_sfc = _R_d * T_virt_surf / _grav
    H_t = H_sfc

    z′ = z / H_t
    tanh_z′ = tanh(z′)

    ΔTv =  T_virt_surf -  T_min_ref
    Tv =  T_virt_surf - ΔTv * tanh_z′

    ΔTv′ = ΔTv / T_virt_surf
    p = -H_t * (z′ + ΔTv′ * (log(1 - ΔTv′ * tanh_z′) - log(1 + tanh_z′) + z′))
    p /= H_sfc * (1 - ΔTv′^2)
    p = _MSLP * exp(p)
    ρ = p/(_R_d*Tv)
    return (Tv, p, ρ)
end


# update discrete_hydrostatic_balance!
function discrete_hydrostatic_balance!(ρ, w, ρθ, Δz::Float64, _grav::Float64, Π::Function)
    for i = 1:length(ρ)-1
        ρ[i+1] = ρθ[i+1]/(-2Δz*_grav/(Π(ρθ[i+1]) - Π(ρθ[i])) - ρθ[i]/ρ[i])
    end
end



###################################### Start
ndays = 10

# whether use discrete hydrostatic balance correction
DHB = true

N = 30
L = 30e3
Δz = L/N

_MSLP = 1e5
_grav = 9.8
_R_m = 287.058
γ = 1.4
_C_p = _R_m * γ / (γ - 1)
_C_v = _R_m / (γ - 1)
_R_d = _R_m


Π = (ρθ) -> _C_p*(_R_d*ρθ/_MSLP)^(_R_m/_C_v)

###################################### Initialization
ρ = zeros(Float64, N)
w = zeros(Float64, N+1)
ρθ = zeros(Float64, N)

# set the bottom velocity
# w[1] = 1.0

for i = 1:N
    z = (i-0.5)*L/N
    Tvᵢ, pᵢ, ρᵢ = DecayingTemperatureProfile(z; T_virt_surf = 280.0, T_min_ref = 230.0, _R_d = _R_d, _grav = _grav, _MSLP = _MSLP)
    ρ[i]  = ρᵢ
    ρθ[i] = ρᵢ*Tvᵢ*(_MSLP/pᵢ)^(_R_m/_C_p)
end


if DHB
    discrete_hydrostatic_balance!(ρ, w, ρθ, Δz, _grav, Π)
end

ss = Single_Stack(L, N, 0.0, copy(ρ), copy(w), copy(ρθ), _grav, Π)
max_wave_speed = compute_wave_speed(ρ, w, ρθ, _R_m, _C_p, γ, _MSLP)

@info "maximum Δt (cfl = 1) = ",  Δz / max_wave_speed

###################################### Advance
Δt = 3.0
Nt = Int64(round(86400  * ndays / Δt))
t_end
for i = 1:Nt
    temporal_update!(ss, Δt)
end
T = Nt * Δt


zz = Array(LinRange(Δz/2, L - Δz/2, N))
zz_h = Array(LinRange(0, L, N+1))

fig, (ax1, ax2, ax3) = PyPlot.subplots(ncols = 3, nrows=1, sharex=false, sharey=false, figsize=(12,6))
ax1.plot(ρ, zz,  "-o", fillstyle = "none", label = "T=0")
ax1.plot(ss.ρ, zz, "-", fillstyle = "none", label = "T=$(ndays) days")
ax1.legend()
ax1.set_ylabel("ρ")

ax2.plot(w, zz_h, "-o", fillstyle = "none", label = "T=0")
ax2.plot(ss.w, zz_h,  "-", fillstyle = "none", label = "T=$(ndays) days")
ax2.legend()
ax2.set_ylabel("w")

ax3.plot(ρθ, zz,  "-o", fillstyle = "none", label = "T=0")
ax3.plot(ss.ρθ, zz, "-", fillstyle = "none", label = "T=$(ndays) days")
ax3.legend()
ax3.set_ylabel("ρθ")

fig.tight_layout()
fig.savefig("HB-Theta-$(ndays)-HB-$(DHB).png")
