# Implicitly solving the heat equation
# at cell centers and cell faces
# Equation: ∂_t T = α * ∇²(T) + S, S = [β*sin(γ*z), β]
# Boundary conditions: ∇T(z=a) = ∇T_bottom, T(z=b) = T_top
import Plots
using LinearAlgebra

# Note that we can stably take huge timesteps even when α is large
# warning, analytic solution only valid for a=0, b=1
a,b, n = 0, 1, 10               # zmin, zmax, number of cells
n̂_min, n̂_max = -1, 1            # Outward facing unit vectors
α = 100;                        # thermal diffusivity, larger means more stiff
β, γ = 10000, π;                # source term coefficients
Δt = 100;                       # timestep
N_t = 10;                       # number of timesteps
FT = Float64;                   # float type
Δz = FT(b-a)/FT(n)
Δz² = Δz^2;
∇²_op = [1/Δz², -2/Δz², 1/Δz²]; # row or Laplacian operator
T_init = 1;                     # Initial temperature, shouldn't matter for steady solution
∇T_bottom = 10;                  # Temperature gradient at the top
T_top = 1;                      # Temperature at the bottom
use_sin_forcing = true
S(z) = use_sin_forcing ? β*sin(γ*z) : β # source term, (sin for easy integration)
zf = range(a, b, length=n+1);
zc = map(i-> zf[i]+Δz/2, 1:n);

# Analytic steady state solution:
# ∂∂T∂∂z = -S(z)/α = -β*sin(γ*z)/α
# ∂T∂z = β*cos(γ*z)/(γ*α)+c1
# T(z) = β*sin(γ*z)/(γ^2*α)+c1*z+c2 # generic solution
# Apply bc: ∂T∂z(a) = β*cos(γ*a)/(γ*α)+c1 = ∇T_bottom => c1 = ∇T_bottom-β*cos(γ*a)/(γ*α)
# Apply bc: T(b) =    β*sin(γ*b)/(γ^2*α)+c1*b+c2 = T_top =>
#         c2 = T_top-(β*sin(γ*b)/(γ^2*α)+c1*b)
function T_analytic_sin(z)
    c1 = ∇T_bottom-β*cos(γ*a)/(γ*α)
    c2 = T_top-(β*sin(γ*b)/(γ^2*α)+c1*b)
    return β*sin(γ*z)/(γ^2*α)+c1*z+c2
end

# Analytic steady state solution:
# ∂∂T∂∂z = -S(z)/α = -β/α
# ∂T∂z = -z*β/α+c1
# T(z) = -β*z^2/(2*α)+c1*z+c2 # generic solution
# Apply bc: ∂T∂z(a) = -a*β/α+c1 = ∇T_bottom => c1 = ∇T_bottom+a*β/α
# Apply bc: T(b) =    -β*b^2/(2*α)+c1*b+c2 = T_top =>
#         c2 = T_top-(-β*b^2/(2*α)+c1*b)
function T_analytic_uniform(z)
    c1 = ∇T_bottom+a*β/α
    c2 = T_top-(-β*b^2/(2*α)+c1*b)
    return -β*z^2/(2*α)+c1*z+c2
end

T_analytic(z) =
    use_sin_forcing ? T_analytic_sin(z) : T_analytic_uniform(z)

#####
##### T ∈ cell centers, ∇T ∈ cell faces
#####
println("Solve for T ∈ cell centers")
T = zeros(FT, n);

# Equations: derivation to matrix form
# ∂_t T = α * ∇²(T) + S
# (T^{n+1}-T^n) = Δt (α * ∇²(T^{n+1}) + S)
# (T^{n+1} - Δt α * ∇²(T^{n+1})) = T^n + Δt*S
# (I - Δt α * ∇²) T^{n+1} = T^n + Δt*S

# Derive Dirichlet boundary stencil & source:
# ∂_t T = α * (T[i-1]+T[g]-2*T[i])/Δz² + S
# ∂_t T = α * (T[i-1]+(2 T[b] - T[i])-2*T[i])/Δz² + S
# ∂_t T = α * (T[i-1]-3*T[i])/Δz² + S + α * (2 T[b])/Δz²

# Derive Neumann boundary stencil & source:
# ∇T_bottom*n̂ = (T[g] - T[i])/Δz,     n̂ = [-1,1] ∈ [zmin,zmax]
# ∇T_bottom*n̂ = -(T[i] - T[g])/Δz,    n̂ = [-1,1] ∈ [zmin,zmax]
# ∂_t T = α * ((T[ii] - T[i])/Δz - (T[i] - T[g])/Δz)/Δz + S
# ∂_t T = α * ((T[ii] - T[i])/Δz + ∇T_bottom*n̂)/Δz + S
# ∂_t T = α * (T[ii] - T[i])/Δz² + S + α*∇T_bottom*n̂/Δz

# Note that size(A) = (n,n) due to 1 Dirichlet and 1 Neumann.
∇² = Tridiagonal(
    ones(FT, n-1) .* ∇²_op[1],
    ones(FT, n)   .* ∇²_op[2],
    ones(FT, n-1) .* ∇²_op[3]
);
# Modify boundary stencil to account for BC
∇².d[1] = -1/Δz²
∇².d[end] = -3/Δz²

A = LinearAlgebra.I - Δt.* α .* ∇²

# Compute boundary sources: α * (2 T[b])/Δz², α * ∇T_bottom*n̂/Δz
AT_b = zeros(FT, n);
AT_b[1] = α*∇T_bottom*n̂_min/Δz;
AT_b[end] = α*2*T_top/Δz²;

# Set initial condition:
T .= T_init;
@inbounds for i_t in 1:N_t
    # change in temperature per iter:
    b_rhs = T .+ Δt .* (S.(zc) .+ AT_b)
    T_new = A \ b_rhs
    ΔT_norm = sum(T .- T_new)/length(T)
    @show ΔT_norm # watch ΔT norm
    T .= T_new
end

# Interpolate to, and plot on, cell faces
zf = range(a, b, length=n+1);
Tf = zeros(FT, n+1);
Tf[2:end-1] = [(T[i]+T[i+1])/2 for i in 1:length(T)-1]
# ∇T_bottom*n̂ = (T[g] - T[i])/Δz, T[g] = 2*T[b] - T[i]
# ∇T_bottom*n̂ = ((2*T[b] - T[i]) - T[i])/Δz
# Δz*∇T_bottom*n̂ = 2*T[b] - 2*T[i]
# (Δz*∇T_bottom*n̂ + 2*T[i])/2 = T[b]
Tf[1] = (Δz*∇T_bottom*n̂_min + 2*T[1])/2
Tf[end] = T_top
p1 = Plots.plot(zf, T_analytic.(zf), label="analytic", markershape=:circle, markersize=6)
p1 = Plots.plot!(p1, zf, Tf, label="numerical", markershape=:diamond)
p1 = Plots.plot!(p1, title="T ∈ cell centers")

p3 = Plots.plot(zf, abs.(Tf .- T_analytic.(zf)), label="error", markershape=:circle, markersize=6)
p3 = Plots.plot!(p3, title="T ∈ cell centers")

#####
##### T ∈ cell faces, ∇T ∈ cell centers
#####

println("Solve for T ∈ cell faces")
T = zeros(FT, n+1);

# Equations: derivation to matrix form
# ∂_t T = α * ∇²(T) + S
# (T^{n+1}-T^n) = Δt (α * ∇²(T^{n+1}) + S)
# (T^{n+1} - Δt α * ∇²(T^{n+1})) = T^n + Δt*S
# (I - Δt α * ∇²) T^{n+1} = T^n + Δt*S

# Derive Dirichlet boundary stencil & source:
# ∂_t T = α * (T[i-1]+T[b]-2*T[i])/Δz² + S
# ∂_t T = α * (T[i-1]-2*T[i])/Δz² + S + α * T[b] / Δz²

# Derive Neumann boundary stencil & source:
# ∇T_bottom*n̂= (T[g] - T[i])/(2Δz),     n̂ = [-1,1] ∈ [zmin,zmax]
# T[i] + 2*Δz*∇T_bottom*n̂ = T[g]
# ∂_t T = α * (((T[i] + 2*Δz*∇T_bottom*n̂) - T[b])/Δz - (T[b] - T[i])/Δz)/Δz + S
# ∂_t T = α * (((T[i]) - T[b])/Δz - (T[b] - T[i])/Δz)/Δz + S + α/Δz²*2*Δz*∇T_bottom*n̂
# ∂_t T = α * (2*T[i] - 2*T[b])/Δz² + S + 2α/Δz*∇T_bottom*n̂

# Note that size(A) = (n,n) due to 1 Dirichlet and 1 Neumann.
∇² = Tridiagonal(
    ones(FT, n-1) .* ∇²_op[1],
    ones(FT, n)   .* ∇²_op[2],
    ones(FT, n-1) .* ∇²_op[3]
);
# Modify boundary stencil to account for BCs
∇².d[1] = -2/Δz²
∇².du[1] = +2/Δz²

A = LinearAlgebra.I - Δt.* α .* ∇²

# Compute boundary source: α * T[b] / Δz²
AT_b = zeros(FT, n+1);
AT_b[1] = α*2/Δz*∇T_bottom*n̂_min;
AT_b[end-1] = α*T_top/Δz²;

# Set initial condition:
T .= T_init;
T[n+1] = T_top
@inbounds for i_t in 1:N_t
    interior = 1:n
    b_rhs = T[interior] .+ Δt .* (S.(zf[interior]) .+ AT_b[interior])
    T_new = A \ b_rhs
    ΔT_norm = sum(T[interior] .- T_new)/length(T)
    @show ΔT_norm # watch ΔT norm
    T[interior] .= T_new
end

p2 = Plots.plot(zf, T_analytic.(zf), label="analytic", markershape=:circle, markersize=6)
p2 = Plots.plot!(p2, zf, T, label="numerical", markershape=:diamond)
p2 = Plots.plot!(p2, title="T ∈ cell faces")

p4 = Plots.plot(zf, abs.(T .- T_analytic.(zf)), label="error", markershape=:circle, markersize=6)
p4 = Plots.plot!(p4, title="T ∈ cell faces")

Plots.plot(p3, p4, p1, p2)
