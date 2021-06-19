# Implicitly solving the heat equation
# at cell centers and cell faces
# Equation: ∂_t T = α * ∇²(T) + S
# Boundary conditions: T(z=a) = T_bottom, T(z=b) = T_top
import Plots
using LinearAlgebra

# Note that we can stably take huge timesteps even when α is large
# warning, analytic solution only valid for a=0, b=1
a,b, n = 0, 1, 10               # zmin, zmax, number of cells
α = 100;                        # thermal diffusivity, larger means more stiff
Δt = 1000;                      # timestep
N_t = 10;                       # number of timesteps
S = 1;                          # source term
FT = Float64;                   # float type
Δz² = (FT(b-a)/FT(n))^2;
∇²_op = [1/Δz², -2/Δz², 1/Δz²]; # row or Laplacian operator
T_init = 1;                     # Initial temperature
T_bottom = 1;                   # Temperature at the top
T_top = 1;                      # Temperature at the bottom

# Analytic steady state solution:
# ∂∂T∂∂z = -S/α
# ∂T∂z = -S/α*z+c1
# T(z) = -S/α*z^2/2+c1*z+c2
# Assuming a == 0, b == 1:
# Apply bc: T(0) = c2 = T_bottom
# Apply bc: T(1) = -S/α/2+c1 = 0 => c1 = S/(2α)
T_analytic(z) = 1 .+ -S/α*(z^2)/2 + S/(2α)*z

#####
##### T ∈ cell centers, ∇T ∈ cell faces
#####
println("Solve for T ∈ cell centers")

z = range(a, b, length=n);
T = zeros(FT, n);

# Equations: derivation to matrix form
# ∂_t T = α * ∇²(T) + S
# (T^{n+1}-T^n) = Δt (α * ∇²(T^{n+1}) + S)
# (T^{n+1} - Δt α * ∇²(T^{n+1})) = T^n + Δt*S
# (I - Δt α * ∇²) T^{n+1} = T^n + Δt*S

# At boundaries (Dirichlet):
# ∂_t T = α * (T[i±1]+T[g]-2*T[i])/Δz² + S
# ∂_t T = α * (T[i±1]+(2 T[b] - T[i])-2*T[i])/Δz² + S
# ∂_t T = α * (T[i±1]-3*T[i])/Δz² + S + α * (2 T[b])/Δz²

A = Tridiagonal(
    zeros(FT, n-1) .- Δt*α*∇²_op[1],
     ones(FT, n)   .- Δt*α*∇²_op[2],
    zeros(FT, n-1) .- Δt*α*∇²_op[3]
);

# Compute boundary source: (2 T[b])/Δz²
AT_b = zeros(FT, n);
AT_b[1] = α*2*T_bottom/Δz²;
AT_b[end] = α*2*T_top/Δz²;

# Modify matrix operator to account for BC
A[1,1] = 1 - Δt*α*(-3/Δz²);
A[end,end] = 1 - Δt*α*(-3/Δz²);

# Set initial condition:
T .= T_init;
@inbounds for i_t in 1:N_t
    # change in temperature per iter:
    T_new = inv(A) * (T .+ Δt .* (S .+ AT_b))
    ΔT_norm = sum(T .- T_new)/length(T)
    @show ΔT_norm # watch ΔT norm
    T .= T_new
end

# Interpolate to, and plot on, cell faces
zf = range(a, b, length=n+1);
Tf = zeros(FT, n+1);
Tf[2:end-1] = [(T[i]+T[i+1])/2 for i in 1:length(T)-1]
Tf[1] = T_bottom
Tf[end] = T_top
p1 = Plots.plot(zf, T_analytic.(zf), label="analytic", markershape=:circle, markersize=6)
p1 = Plots.plot!(p1, zf, Tf, label="numerical", markershape=:diamond)
p1 = Plots.plot!(p1, title="T ∈ cell centers")

#####
##### T ∈ cell faces, ∇T ∈ cell centers
#####

println("Solve for T ∈ cell faces")
z = range(a, b, length=n+1);
T = zeros(FT, n+1);

# Equations: derivation to matrix form
# ∂_t T = α * ∇²(T) + S
# (T^{n+1}-T^n) = Δt (α * ∇²(T^{n+1}) + S)
# (T^{n+1} - Δt α * ∇²(T^{n+1})) = T^n + Δt*S
# (I - Δt α * ∇²) T^{n+1} = T^n + Δt*S

# At first interior face (Dirichlet):
# ∂_t T = α * (T[i±1]+T[b]-2*T[i])/Δz² + S
# ∂_t T = α * (T[i±1]-2*T[i])/Δz² + S + α * T[b] / Δz²

A = Tridiagonal(
    zeros(FT, n-2) .- Δt*α*∇²_op[1],
     ones(FT, n-1) .- Δt*α*∇²_op[2],
    zeros(FT, n-2) .- Δt*α*∇²_op[3]
);

# Compute boundary source: α * T[b] / Δz²
AT_b = zeros(FT, n+1);
AT_b[2] = α*T_bottom/Δz²;
AT_b[end-1] = α*T_top/Δz²;

# Matrix operator need not be modified to account for Dirichlet BC

# Set initial condition:
T .= T_init;
@inbounds for i_t in 1:N_t
    interior = 2:n
    T_new = inv(A) * (T[interior] .+ Δt .* (S .+ AT_b[interior]))
    ΔT_norm = sum(T[interior] .- T_new)/length(T)
    @show ΔT_norm # watch ΔT norm
    T[interior] .= T_new
end

p2 = Plots.plot(z, T_analytic.(z), label="analytic", markershape=:circle, markersize=6)
p2 = Plots.plot!(p2, z, T, label="numerical", markershape=:diamond)
p2 = Plots.plot!(p2, title="T ∈ cell faces")

Plots.plot(p1, p2)
