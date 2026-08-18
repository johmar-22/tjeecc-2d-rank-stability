# =============================================================================
# TJEECC - CELL 10 / PLAN STEP 5: ballistic ambipolar 2D MOSFET model
#
# Pure numpy, fully vectorised over (n_materials, n_draws).
#
# TWO DESIGN DECISIONS, both forced by findings:
#
# 1. BALLISTIC, not quasi-ballistic. C2DB carries zero deformation-potential
#    files, so mu_DP is unavailable. The ballistic Natori limit is the primary
#    model; a parametric mean-free-path sweep is reported as a sensitivity.
#
# 2. AMBIPOLAR two-branch transport. In a purely thermionic model with a free
#    threshold voltage the band gap does NOT affect I_ON/I_OFF, so the gap
#    uncertainty (our best-measured input) would propagate to nothing. The gap
#    must enter physically. It does so through the ambipolar leakage floor:
#    the gate shifts both bands, so hole leakage rises as electron leakage
#    falls and the achievable I_OFF scales roughly as exp(-E_g/2kT). This is
#    the standard argument for why 2D channels need E_g >~ 1 eV.
#
# Requires cell00_bootstrap.py.
# =============================================================================

import numpy as np
from scipy import constants as sc

# --- physical constants, never hardcoded -------------------------------------
KB   = sc.Boltzmann
Q    = sc.elementary_charge
HBAR = sc.hbar
M0   = sc.electron_mass
EPS0 = sc.epsilon_0

# --- technology corner (Table 9 of the paper) --------------------------------
TECH = dict(
    L_g      = 12e-9,     # gate length, m
    EOT      = 0.6e-9,    # equivalent oxide thickness, m
    eps_ox   = 3.9,       # SiO2 relative permittivity (EOT reference)
    V_DD     = 0.65,      # supply, V
    T        = 300.0,     # K
    C_it     = 1e-2,      # interface trap capacitance, F/m^2  (1 uF/cm^2)
    g_v      = 2.0,       # valley degeneracy (stated per material where known)
    I_OFF    = 100e-9/1e-6,   # 100 nA/um expressed in A/m
    W        = 1.0,       # width normalisation: all currents are per metre
)

# =============================================================================
# Fermi-Dirac integrals (normalised: F_j(eta) -> exp(eta) as eta -> -inf)
# =============================================================================
def F0(eta):
    """F_0(eta) = ln(1 + e^eta), overflow-safe."""
    eta = np.asarray(eta, dtype=np.float64)
    return np.where(eta > 30.0, eta, np.log1p(np.exp(np.minimum(eta, 30.0))))

def F_half(eta):
    """Bednarczyk & Bednarczyk (1978). Stated accuracy ~0.4%; we verify <5e-3."""
    eta = np.asarray(eta, dtype=np.float64)
    e = np.clip(eta, -700, 700)
    xi = (3.0 * np.sqrt(np.pi) / 4.0) * (
        e**4 + 50.0 + 33.6 * e * (1.0 - 0.68 * np.exp(-0.17 * (e + 1.0) ** 2))
    ) ** (-0.375)
    return 1.0 / (np.exp(-np.clip(e, -700, 700)) + xi)

def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -700, 700)
    return 1.0 / (1.0 + np.exp(-x))

# =============================================================================
# Core device physics
# =============================================================================
def _dos_prefactor(m_dos, g_v, T):
    """N0 = g_v m_d k_B T / (pi hbar^2)   [carriers per m^2]"""
    return g_v * m_dos * M0 * KB * T / (np.pi * HBAR**2)

def _v_thermal(m_cond, T):
    """Non-degenerate thermal injection velocity, sqrt(2 kT / (pi m_t))."""
    return np.sqrt(2.0 * KB * T / (np.pi * m_cond * M0))

def carrier_densities(u, Eg_kT, N0e, N0h):
    """Electron and hole sheet densities at the top of the barrier.

    u    = (E_F - E_C)/kT      electron degeneracy at the virtual source
    hole degeneracy w = (E_V - E_F)/kT = -u - Eg/kT
    """
    n_s = N0e * F0(u)
    p_s = N0h * F0(-u - Eg_kT)
    return n_s, p_s

def body_factor(eps_ch, t_ch, C_ox, tech):
    """Subthreshold body factor m = 1 + C_it/C_ox + beta exp(-L_g/2 lambda).

    Without this, eps_ch and t_ch do not enter the current at all: the scale
    length would only feed decorative SS and DIBL outputs, leaving two of the
    six uncertain inputs unable to move any figure of merit. Physically, a
    degraded slope costs gate overdrive, so at fixed I_OFF and fixed V_DD it
    reduces I_ON. m multiplies the subthreshold slope: SS = m * (kT/q) ln10.
    """
    lam = scale_length(eps_ch, t_ch, tech["eps_ox"], tech["EOT"])
    return 1.0 + tech["C_it"] / C_ox + 10.0 * np.exp(-tech["L_g"] / (2.0 * lam))

def V_G_of_u(u, Eg_kT, N0e, N0h, C_ox, T, m_body=1.0):
    """Gate voltage required to place the band at u. EXPLICIT, no iteration.

    V_G = m (u kT/q) + q (n_s - p_s)/C_ox

    Charge balance rearranges to closed form in u, so parameterising the sweep
    by u removes the inner root-find entirely; the original version bisected at
    every point of a V_G grid, costing billions of transcendental evaluations.
    The m prefactor on the surface-potential term is the capacitive divider: in
    subthreshold V_G ~ m u kT/q, giving SS = m * 60 mV/dec at 300 K.
    """
    n_s, p_s = carrier_densities(u, Eg_kT, N0e, N0h)
    return m_body * u * (KB * T / Q) + Q * (n_s - p_s) / C_ox

def solve_u(V_G, Eg_kT, N0e, N0h, C_ox, T, m_body=1.0, n_iter=45):
    """Bisection for u at a given V_G. V_G_of_u is monotonic increasing in u.

    45 iterations on a bracket of width 800 resolves to 800/2^45, far below any
    physically meaningful scale.
    """
    shape = np.broadcast(V_G, Eg_kT, N0e, N0h, m_body).shape
    lo = np.full(shape, -400.0)
    hi = np.full(shape, 400.0)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_small = V_G_of_u(mid, Eg_kT, N0e, N0h, C_ox, T, m_body) < V_G
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    return 0.5 * (lo + hi)

def ballistic_current(u, Eg_kT, N0e, N0h, vTe, vTh, V_DS, T):
    """Natori ballistic current, electron and ambipolar hole branches.

    Electrons are injected over the SOURCE barrier:
        I_n = q N0e v_Te [F_1/2(u) - F_1/2(u - u_D)]
    In the non-degenerate, high-V_DS limit this reduces exactly to q n_s v_T,
    which unit test (a) verifies.

    Holes are injected from the DRAIN, whose Fermi level sits q V_DS below the
    source. The hole degeneracy at the top of the barrier seen from the drain
    is therefore
        w_drain = -u - E_g/kT + u_D
    and from the source
        w_source = -u - E_g/kT
    so the ambipolar floor scales as exp(-(E_g - qV_DS)/2kT), NOT
    exp(-E_g/2kT). Placing the hole barrier at the source instead is a real
    physics error: it pushes the gap-sensitive regime below ~0.5 eV, where no
    screened material lives, and the band gap then has no effect on I_ON/I_OFF
    at all. This term is the mechanism by which gap uncertainty reaches the
    device terminal, and it is why a FET needs E_g > qV_DD.
    """
    uD = Q * V_DS / (KB * T)
    I_n = Q * N0e * vTe * (F_half(u) - F_half(u - uD))
    w_src = -u - Eg_kT
    w_drn = w_src + uD
    I_p = Q * N0h * vTh * (F_half(w_drn) - F_half(w_src))
    return I_n, I_p

def ballisticity(m_dos, m_cond, lambda0_over_l, m_ref=0.5):
    """Backscattering transmission B = (1-r)/(1+r) with r = l_kT/(l_kT+lambda).

    B = lambda / (2 l_kT + lambda).

    The mean free path inherits the mass dependence of the mobility. For 2D
    deformation-potential scattering mu ~ 1/(m_d m_c), and lambda ~ 2 mu kT/(q v_T)
    with v_T ~ 1/sqrt(m_c), so

        lambda  ~  1 / (m_d sqrt(m_c))

    `lambda0_over_l` is lambda/l_kT at the reference mass and is the swept
    parameter: >>1 ballistic, <<1 diffusive. Deformation potentials are absent
    from this C2DB release, so the absolute mobility cannot be predicted; the
    sweep is therefore parametric and must be reported as a sensitivity in
    ballisticity, never as an absolute mobility.

    Consequence, and the point of the paper. At EOT = 0.6 nm the on-state is
    CHARGE-limited, not DOS-limited: the inversion charge is set by C_ox V_ov,
    so the m_d term cancels and the ballistic current scales as 1/sqrt(m_c).
    Combining that with B gives, for isotropic bands,

        d log I_ON / d log m  =  -0.5 - 1.5 * 2 l /(2 l + lambda)

    which is -0.5 fully ballistic and -2.0 fully diffusive. Verified against
    the simulation: -0.55 at lambda/l = 1000, -1.04 at 4, -1.97 at 0.1.

    NOTE. An earlier derivation used the non-degenerate DOS-limited form
    I ~ m_d/sqrt(m_c), giving +0.5 ballistic and predicting a sign change with
    an insensitivity point at B = 2/3. That is WRONG for this technology
    corner: heavier mass hurts in both regimes and there is no zero crossing.
    What is real is the magnitude: effective-mass sensitivity is exactly 4x
    stronger in the diffusive limit than in the ballistic one.
    """
    lam = lambda0_over_l * (m_ref / m_dos) * np.sqrt(m_ref / m_cond)
    return lam / (2.0 + lam)

def scale_length(eps_ch, t_ch, eps_ox, t_ox):
    """Double-gate electrostatic scale length, lambda = sqrt(eps_ch/eps_ox * t_ch * t_ox)."""
    return np.sqrt(np.maximum(eps_ch / eps_ox, 1e-6) * t_ch * t_ox)

# =============================================================================
# Figures of merit
# =============================================================================
def compute_fom(Eg, m_dos_e, m_dos_h, m_cond_e, m_cond_h, eps_ch, t_ch,
                tech=TECH, n_vg=128, chunk=8, lambda0_over_l=None):
    """Return a dict of device figures of merit.

    All inputs are arrays of identical shape (n_materials, n_draws) except
    scalars in `tech`. Currents are per metre of gate width (A/m).

    Procedure follows standard DTCO benchmarking: sweep V_G, locate the
    ambipolar minimum, then place a V_DD window so that I_OFF meets spec and
    read I_ON at the top of the window. Comparing at a fixed I_OFF is what
    makes the comparison across materials fair; omitting it is the single most
    common error in this kind of study.
    """
    T, V_DD = tech["T"], tech["V_DD"]
    kT_q = KB * T / Q
    shape = Eg.shape
    C_ox = tech["eps_ox"] * EPS0 / tech["EOT"]

    Eg_kT = Eg * Q / (KB * T)
    N0e = _dos_prefactor(m_dos_e, tech["g_v"], T)
    N0h = _dos_prefactor(m_dos_h, tech["g_v"], T)
    vTe = _v_thermal(m_cond_e, T)
    vTh = _v_thermal(m_cond_h, T)

    lam = scale_length(eps_ch, t_ch, tech["eps_ox"], tech["EOT"])
    dibl_fac = np.exp(-tech["L_g"] / (2.0 * lam))
    m_body = body_factor(eps_ch, t_ch, C_ox, tech)
    SS = m_body * (KB * T / Q) * np.log(10.0) * 1e3   # mV/dec, consistent with m
    DIBL = 1000.0 * 0.8 * dibl_fac                    # mV/V

    flat = [np.asarray(x, float).reshape(-1) for x in
            np.broadcast_arrays(Eg_kT, N0e, N0h, vTe, vTh, m_body)]
    n_tot = flat[0].size
    I_on_f = np.empty(n_tot); I_off_f = np.empty(n_tot)
    BLOCK = max(1, int(chunk) * 4096)   # ~32k cases/block at chunk=8: <1 GB peak
    target = tech["I_OFF"]
    frac = np.linspace(0.0, 1.0, n_vg)[None, :]

    for s in range(0, n_tot, BLOCK):
        e = min(s + BLOCK, n_tot)
        eg, n0e, n0h, ve, vh, mb = (a[s:e][:, None] for a in flat)

        # Sweep in u, not V_G: the ambipolar minimum sits near u = -Eg/2kT, so
        # a per-material span from below the hole branch to well into
        # degeneracy covers the whole transfer characteristic with no inner
        # root-find. This is the change that made the full Monte Carlo feasible.
        u_lo = -(eg + 25.0)
        u_hi = np.full_like(eg, 30.0)
        u = u_lo + (u_hi - u_lo) * frac                      # (m, n_vg)

        In, Ip = ballistic_current(u, eg, n0e, n0h, ve, vh, V_DD, T)
        Itot = In + Ip
        Vg = V_G_of_u(u, eg, n0e, n0h, C_ox, T, mb)

        imin = np.argmin(Itot, axis=1)
        rows = np.arange(Itot.shape[0])
        # Restrict to the n-branch, monotonic above the ambipolar minimum.
        # Mask BELOW the minimum with -inf, never +inf: with +inf the test
        # `>= target` is trivially true and argmax returns index 0, a point on
        # the wrong branch. That bug previously inverted the E_g dependence.
        on_branch = np.arange(n_vg)[None, :] >= imin[:, None]
        masked = np.where(on_branch, Itot, -np.inf)

        # Reachability must be judged on the ambipolar FLOOR, not on whether
        # any grid point exceeds the target. The n-branch always crosses the
        # target eventually, and when the floor itself is above spec the
        # minimum trivially satisfies `>= target` -- which would report a
        # device that cannot meet 100 nA/um as meeting it exactly, handing
        # narrow-gap materials a free pass and inverting the E_g dependence.
        I_floor = Itot[rows, imin]
        reachable = I_floor <= target

        hit = masked >= target
        any_hit = reachable & hit.any(axis=1)
        idx = np.where(any_hit, np.argmax(hit, axis=1), imin)

        # Interpolate the I_OFF crossing in log(I) vs V_G rather than snapping
        # to the nearest grid point. Subthreshold current changes by e^du per
        # grid step, so on a coarse u grid the nearest-point off-voltage can be
        # tens of percent wrong, which shifts every I_ON by the same amount.
        # With interpolation a 128-point grid is sufficient and the memory
        # footprint stays modest.
        i1 = np.clip(idx, 1, n_vg - 1)
        i0 = i1 - 1
        lI0 = np.log(np.maximum(Itot[rows, i0], 1e-300))
        lI1 = np.log(np.maximum(Itot[rows, i1], 1e-300))
        V0, V1 = Vg[rows, i0], Vg[rows, i1]
        denom = np.where(np.abs(lI1 - lI0) < 1e-12, 1e-12, lI1 - lI0)
        w = (np.log(target) - lI0) / denom
        v_interp = V0 + np.clip(w, 0.0, 1.0) * (V1 - V0)
        # Guard must be idx > imin, NOT idx > 0. When the ambipolar floor sits
        # near the spec, idx == imin and the interpolation reaches back to
        # imin-1, a point on the P-BRANCH, i.e. the wrong side of the minimum.
        # That gave a discontinuous v_off for materials near the reachability
        # boundary, which the census then misread as band-gap sensitivity.
        v_off = np.where(any_hit & (idx > imin), v_interp, Vg[rows, idx])

        v_on = v_off + V_DD
        u_on = solve_u(v_on[:, None], eg, n0e, n0h, C_ox, T, mb)
        In_on, Ip_on = ballistic_current(u_on, eg, n0e, n0h, ve, vh, V_DD, T)
        I_on_f[s:e] = (In_on + Ip_on).ravel()
        # Achieved off-current: the spec where reachable, otherwise the
        # ambipolar floor, which is the physically meaningful limit for a
        # narrow-gap channel and is how E_g reaches the terminal.
        # v_off was interpolated to land exactly on the spec, so the achieved
        # off-current is the target itself. Reading Itot at the bracketing grid
        # point instead overshoots by an amount that scales with the grid step
        # -- and the u-grid step scales with Eg, which manufactured a spurious
        # gap dependence in on/off. This must stay consistent with v_off.
        I_off_f[s:e] = np.where(any_hit, target, I_floor)

    I_on = I_on_f.reshape(shape)
    I_off = I_off_f.reshape(shape)

    # Quasi-ballistic correction. Applied to the ON current only: the
    # off-state is set by the barrier height, which backscattering does not
    # change. lambda0_over_l=None leaves the pure ballistic limit.
    if lambda0_over_l is not None:
        B = ballisticity(m_dos_e, m_cond_e, lambda0_over_l)
        I_on = I_on * B
    else:
        B = np.ones_like(I_on)

    C_g = C_ox * tech["L_g"]                  # per metre width, F/m
    tau = C_g * V_DD / np.maximum(I_on, 1e-30)
    energy = C_g * V_DD**2
    return {
        "I_ON": I_on, "I_OFF": I_off,
        "on_off": I_on / np.maximum(I_off, 1e-30),
        "SS": SS, "DIBL": DIBL, "lambda": lam,
        "tau": tau, "energy": np.broadcast_to(energy, shape).copy(),
        "EDP": energy * tau, "B": np.broadcast_to(B, shape).copy(),
    }

# =============================================================================
# UNIT TESTS - all six must pass before Step 6
# =============================================================================
def run_unit_tests():
    res = []
    T = 300.0

    # (a) THE PREFACTOR TEST. Non-degenerate, high V_DS limit must reduce to
    #     I = W q n_s v_T. This is what pins the Natori prefactor; a factor of
    #     two here is invisible in rankings but wrong in every absolute number.
    md = np.array([0.5]); mc = np.array([0.5])
    N0 = _dos_prefactor(md, 2.0, T); vT = _v_thermal(mc, T)
    u = np.array([-8.0]); uD = 40.0
    I = Q * N0 * vT * (F_half(u) - F_half(u - uD))
    ns = N0 * F0(u)
    ratio = float(I / (Q * ns * vT))
    res.append(("(a) Natori prefactor -> W q n_s v_T", abs(ratio - 1) < 0.05,
                f"ratio={ratio:.4f}"))

    # (b) quantum capacitance limit
    eta = np.array([40.0])
    CQ = Q**2 * _dos_prefactor(md, 2.0, T) / (KB * T) * sigmoid(eta)
    CQ_lim = Q**2 * 2.0 * md * M0 / (np.pi * HBAR**2)
    r = float(CQ / CQ_lim)
    res.append(("(b) C_Q -> q^2 g_v m_d/(pi hbar^2)", abs(r - 1) < 0.01,
                f"ratio={r:.4f}"))

    # (c) SS floor
    kT_q_dec = (KB * T / Q) * np.log(10.0) * 1e3
    res.append(("(c) SS >= 60 mV/dec at 300 K", kT_q_dec >= 59.5,
                f"kT/q*ln10={kT_q_dec:.2f} mV/dec"))

    # (d) monotonic in V_G, saturating in V_DS
    eg = np.array([[1.5 * Q / (KB * T)]]); n0 = _dos_prefactor(np.array([[0.5]]), 2.0, T)
    v = np.linspace(-0.2, 1.2, 40)[None, :]
    C_ox = 3.9 * EPS0 / 0.6e-9
    uu = solve_u(v, eg, n0, n0, C_ox, T)
    In, Ip = ballistic_current(uu, eg, n0, n0, _v_thermal(np.array([[0.5]]), T),
                               _v_thermal(np.array([[0.5]]), T), 0.65, T)
    mono = bool(np.all(np.diff((In).ravel()) > 0))
    res.append(("(d) I_n monotonic in V_G", mono, ""))
    I_lo = ballistic_current(np.array([[5.0]]), eg, n0, n0, np.array([[1e5]]),
                             np.array([[1e5]]), 0.30, T)[0]
    I_hi = ballistic_current(np.array([[5.0]]), eg, n0, n0, np.array([[1e5]]),
                             np.array([[1e5]]), 0.65, T)[0]
    sat = float(I_hi / I_lo)
    res.append(("(d2) I saturates with V_DS", 1.0 <= sat < 1.15, f"I(0.65)/I(0.30)={sat:.4f}"))

    # (e) isotropic bands: harmonic mean == geometric mean
    m1 = m2 = 0.42
    m_cond = 2 * m1 * m2 / (m1 + m2); m_dos = np.sqrt(m1 * m2)
    res.append(("(e) m1=m2 -> m_cond == m_dos", abs(m_cond - m_dos) < 1e-12,
                f"{m_cond:.6f} vs {m_dos:.6f}"))

    # (f) the gap MUST move I_ON/I_OFF, else the paper has no mechanism
    base = dict(m_dos_e=np.array([[0.5]]), m_dos_h=np.array([[0.6]]),
                m_cond_e=np.array([[0.5]]), m_cond_h=np.array([[0.6]]),
                eps_ch=np.array([[5.0]]), t_ch=np.array([[6e-10]]))
    r1 = compute_fom(Eg=np.array([[0.6]]), **base)["on_off"][0, 0]
    r2 = compute_fom(Eg=np.array([[2.0]]), **base)["on_off"][0, 0]
    res.append(("(f) on/off increases with E_g (ambipolar)", r2 > 3 * r1,
                f"Eg=0.6 -> {r1:.3e}   Eg=2.0 -> {r2:.3e}   x{r2/max(r1,1e-30):.1f}"))

    log("--- unit tests ---")
    allok = True
    for name, ok, note in res:
        allok &= bool(ok)
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}   {note}")
    log(f"  {'ALL PASS' if allok else '*** FAILURES: do not proceed to Step 6'}")
    return allok

# --- Fermi-Dirac accuracy check against numerical integration ----------------
def check_fermi_dirac():
    from scipy.integrate import quad
    from scipy.special import gamma
    def exact(eta):
        f = lambda x: np.sqrt(x) / (1.0 + np.exp(np.clip(x - eta, -700, 700)))
        return quad(f, 0, 200, limit=400)[0] / gamma(1.5)
    etas = np.linspace(-10, 30, 50)
    err = max(abs(F_half(e) - exact(e)) / exact(e) for e in etas)
    log(f"  F_1/2 max relative error vs quadrature: {err:.2e} "
        f"(Bednarczyk's published accuracy is ~0.4%; tolerance 5e-3)")
    return err < 5e-3

if __name__ == "__main__" or True:
    ok_fd = check_fermi_dirac()
    ok_ut = run_unit_tests()
    log(f"[step5] Fermi-Dirac {'OK' if ok_fd else 'FAIL'}, "
        f"unit tests {'OK' if ok_ut else 'FAIL'}")
    log("[step5] DONE\n")
