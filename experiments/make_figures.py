from common import *
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import gcd

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
})

COLORS = ["#1f4e79", "#c0392b", "#27ae60", "#8e44ad", "#d35400", "#16a085", "#7f8c8d", "#2c3e50", "#b7950b", "#e84393"]


def save(fig, name):
    eps = os.path.join(FIGURES, name + ".eps")
    fig.savefig(eps, format="eps", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    subprocess.run(["epstopdf", eps], check=True)


def fig_bloch_multi():
    data = load_json("waves_F2.json")
    waves = genuine_waves(data["waves"])
    multi = load_json("multistability.json")
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.7))
    ax = axs[0]
    chosen = []
    seen = set()
    for w in waves:
        if w["d"] in (4, 5, 6, 7, 9, 16, 24, 36, 48) and w["d"] not in seen:
            chosen.append(w)
            seen.add(w["d"])
    for c, w in zip(COLORS, chosen):
        g = np.array(w["grid"])
        eta = np.log(np.array(w["symbol"])) / w["T"]
        ax.plot(g / np.pi, eta, color=c, label=f"$d={w['d']}$")
        if 144 % w["d"] == 0:
            m = 144 // w["d"]
            r = np.arange(1, m // 2 + 1)
            th = 2 * np.pi * r / m
            ax.plot(th / np.pi, np.interp(th, g, eta), "o", ms=2.6, color=c)
    ax.axhline(0.0, color="black", lw=0.6, ls="--")
    ax.set_xlabel(r"Bloch phase $\theta/\pi$")
    ax.set_ylabel(r"$T^{-1}\log\rho_P(\theta)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.42, 0.2)
    ax.legend(ncol=3, loc="lower left", frameon=False, handlelength=1.4, columnspacing=0.8)
    ax.set_title("(a) Bloch symbols of waves at $F=2$")
    ax = axs[1]
    tab = multi["table"]
    Ns = np.array([r["N"] for r in tab])
    cnt = np.array([r["count"] for r in tab])
    colors = ["#c0392b" if n in (120, 144, 240) else "#1f4e79" for n in Ns]
    ax.bar(Ns, cnt, width=0.85, color=colors)
    run = np.cumsum(cnt) / (Ns - Ns[0] + 1)
    ax.plot(Ns, run, color="#27ae60", lw=1.2, label=r"running mean of $|\mathcal{I}(N)|$")
    lim = sum(1.0 / w["d"] for w in waves if w["universal_on_grid"])
    ax.axhline(lim, color="black", lw=0.7, ls="--", label=r"$\sum 1/d$ over universal waves")
    for n in (120, 144, 240):
        i = list(Ns).index(n)
        ax.text(n, cnt[i] + 0.25, f"{n}", ha="center", fontsize=7)
    ax.set_xlabel("lattice size $N$")
    ax.set_ylabel(r"inherited stable cycles $|\mathcal{I}(N)|$")
    ax.set_xlim(3, 242)
    ax.set_ylim(0, max(cnt) + 4.2)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="white", framealpha=1.0)
    ax.set_title("(b) Inherited multistability")
    fig.tight_layout()
    save(fig, "fig_bloch_multi")


def fig_chaos():
    ch = load_json("chaos.json")
    sad = load_json("saddle.json")
    fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.35))
    ax = axs[0]
    win = load_json("rossler_window.json")
    Nn = np.array(win["N"])
    ln = np.array(win["nat"])
    lu = np.array(win["upo"])
    for N, a, b in zip(Nn, ln, lu):
        if a >= 0:
            col = "#fadbd8"
        elif b > 0:
            col = "#fdebd0"
        else:
            col = "#d5f5e3"
        ax.axvspan(N - 0.5, N + 0.5, color=col, lw=0)
    ax.plot(Nn, ln, "o-", ms=3, color="#1f4e79", label="natural measure")
    ax.plot(Nn, lu, "s-", ms=3, color="#c0392b", label="period one orbit")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("ring size $N$")
    ax.set_ylabel("largest normal exponent")
    ax.set_xlim(1.5, 25.5)
    ax.set_ylim(-0.1, 0.14)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="white", framealpha=1.0)
    ax.set_title("(a) R\u00f6ssler ring, $\\varepsilon=1$")
    ax = axs[1]
    for name, c, nmax in [("Rossler", "#1f4e79", 16), ("HindmarshRose", "#27ae60", 6), ("Lorenz", "#c0392b", 3)]:
        rows = ch["rings"][name]["rows"]
        N = [r["N"] for r in rows if r["N"] <= nmax]
        D = [r["DKY_full"] for r in rows if r["N"] <= nmax]
        ax.plot(N, D, "o-", ms=3, color=c, label=name.replace("HindmarshRose", "Hindmarsh-Rose").replace("Rossler", "Rössler"))
        ax.axhline(rows[0]["DKY_base"], color=c, lw=0.6, ls=":")
    ax.set_xlabel("ring size $N$")
    ax.set_ylabel(r"$\dim_{\mathrm{KY}}$ of synchronized state")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("(b) Kaplan-Yorke inflation")
    ax = axs[2]
    ms = [r["m"] for r in sad["counts"]]
    npos = [r["n_positive"] for r in sad["counts"]]
    ax.plot(ms, npos, "o", ms=2.5, color="#1f4e79", label="lifted chaotic saddle")
    ax.plot([r["N"] / 5 for r in sad["actual"]], [r["n_positive"] for r in sad["actual"]], "s", ms=4,
            color="#c0392b", label="attractor of $N=5m$")
    mm = np.array(ms)
    ax.plot(mm, sad["slope"] * mm, color="black", lw=0.6, ls="--", label=f"slope {sad['slope']:.2f}")
    ax.set_xlabel("multiple $m$, $N=5m$")
    ax.set_ylabel("positive exponents")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("(c) Lorenz-96, $F=8$")
    fig.tight_layout()
    save(fig, "fig_chaos")


def fig_grid():
    g = load_json("grid.json")
    rows = [("Lorenz", r"Lorenz ring, $\varepsilon=3$"), ("Rossler", "Rössler ring, $\\varepsilon=1$"),
            ("HindmarshRose", r"Hindmarsh-Rose ring, $\varepsilon=1$"), ("Lorenz96", r"Lorenz-96, $d=5$, $F=4$")]
    fig = plt.figure(figsize=(7.0, 5.6))
    for i, (key, label) in enumerate(rows):
        panels = g[key]["panels"]
        for j, p in enumerate(panels):
            ax = fig.add_subplot(4, 6, 6 * i + j + 1, projection="3d")
            tr = np.array(p["traj"])
            col = "black" if j == 0 else "#1f5fa8"
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=col, lw=0.35)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.tick_params(length=0, pad=-2)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_alpha(0.0)
                axis._axinfo["grid"]["linewidth"] = 0.25
            ax.view_init(elev=22, azim=-58)
            N = p["N"]
            if key == "Lorenz96":
                ttl = "$d=5$" if j == 0 else f"$N={N}$"
            else:
                ttl = "single" if j == 0 else f"$N={N}$"
            ax.set_title(ttl, fontsize=7.5, pad=-2)
            if key == "Lorenz96":
                ax.set_xlabel(r"$x_0(t)$", fontsize=6.3, labelpad=-13)
                ax.set_ylabel(r"$x_0(t-\delta)$", fontsize=6.3, labelpad=-13)
                ax.set_zlabel(r"$x_0(t-2\delta)$", fontsize=6.3, labelpad=-15)
            else:
                ax.set_xlabel("$x_0$", fontsize=6.3, labelpad=-13)
                ax.set_ylabel("$y_0$", fontsize=6.3, labelpad=-13)
                ax.set_zlabel("$z_0$", fontsize=6.3, labelpad=-15)
            ax.set_box_aspect((1, 1, 0.85), zoom=0.86)
        fig.text(0.006, 1.0 - (i + 0.5) / 4.0 * 0.985, label, rotation=90, va="center", ha="left", fontsize=7.2)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.975, bottom=0.005, wspace=0.0, hspace=0.06)
    save(fig, "fig_grid")


def fig_surface():
    s = load_json("surface.json")
    F = np.array(s["F"])
    th = np.array(s["theta"])[1:]
    eta = np.array(s["eta"])[:, 1:]
    TH, FF = np.meshgrid(th / np.pi, F)
    fig = plt.figure(figsize=(5.6, 3.9))
    ax = fig.add_subplot(111, projection="3d")
    lim = np.max(np.abs(eta))
    surf = ax.plot_surface(TH, FF, eta, cmap="coolwarm", vmin=-lim, vmax=lim, linewidth=0.12, edgecolor="#333333",
                           rstride=1, cstride=2, antialiased=True)
    ax.contour(TH, FF, eta, levels=[0.0], colors="black", linewidths=1.6)
    zfloor = eta.min() - 0.05
    ax.contour(TH, FF, eta, levels=[0.0], colors="black", linewidths=0.9, linestyles="--", offset=zfloor)
    ax.contourf(TH, FF, eta, levels=[-10, 0, 10], colors=["#d6eaf8", "#fadbd8"], offset=zfloor)
    ax.set_zlim(zfloor, eta.max() + 0.02)
    ax.set_xlabel(r"Bloch phase $\theta/\pi$", labelpad=3)
    ax.set_ylabel(r"forcing $F$", labelpad=3)
    ax.set_zlabel(r"$T^{-1}\log\rho_P(\theta)$", labelpad=6)
    ax.view_init(elev=26, azim=-62)
    ax.set_box_aspect((1.25, 1.0, 0.75))
    cb = fig.colorbar(surf, shrink=0.52, pad=0.12)
    cb.set_label("exponent", fontsize=8)
    fig.subplots_adjust(left=0.0, right=0.92, bottom=0.04, top=0.98)
    save(fig, "fig_surface")


def fig_route():
    r = load_json("route_d5.json")["rows"]
    F = np.array([x["F"] for x in r])
    ms = list(range(2, 21))
    Z = np.array([[0 if x["stable"][str(m)] else 1 for x in r] for m in ms])
    fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.6))
    ax = axs[0]
    ax.imshow(Z, aspect="auto", origin="lower", extent=[F[0] - 0.01, F[-1] + 0.01, 1.5, 20.5],
              cmap=matplotlib.colors.ListedColormap(["#aed6f1", "#c0392b"]), interpolation="nearest")
    for Fp in [3.937853, 4.981884, 6.371496, 6.640968]:
        ax.axvline(Fp, color="black", lw=0.6, ls="--")
    ax.set_xlabel("forcing $F$")
    ax.set_ylabel("multiple $m$ ($N=5m$)")
    ax.set_title("(a) inheritance of the five site attractor")
    ax = axs[1]
    base = np.array([x["base_exps"][0] if x["kind"] == "chaotic" else
                     (sorted(x["base_exps"])[-2] if len(x["base_exps"]) > 1 else 0) for x in r])
    trans = {}
    for m in [2, 3, 4, 5, 10]:
        vals = []
        for x in r:
            fr = x["farey"]
            lf = x["lam_farey"]
            vals.append(max(lf[i] for i, (a, b) in enumerate(fr) if m % b == 0))
        trans[m] = np.array(vals)
    for c, m in zip(COLORS, trans):
        ax.plot(F, trans[m], color=c, label=f"$m={m}$")
    ax.plot(F, base, color="black", lw=0.8, label="in $V_5$")
    ax.axhline(0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("forcing $F$")
    ax.set_ylabel("largest exponent")
    ax.set_ylim(-0.6, 0.9)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    ax.set_title("(b) transverse Bloch exponents")
    fig.tight_layout()
    save(fig, "fig_route")


def fig_bench():
    wv = load_json("bench_wave.json")
    ly = load_json("bench_lyap.json")
    fig, axs = plt.subplots(1, 4, figsize=(5.9, 2.2))
    marker = {"SBR": "o", "FM": "s", "RM": "D", "IRR": "^", "SBD": "v", "NK": "P", "HILL": "X", "BQR": "<", "CQR": ">"}
    col = {"SBR": "#c0392b", "FM": "#1f4e79", "RM": "#27ae60", "IRR": "#8e44ad", "SBD": "#d35400", "NK": "#16a085",
           "HILL": "#7f8c8d", "BQR": "#b7950b", "CQR": "#e84393"}
    handles = {}
    for k, (data, key, ylab, ttl) in enumerate([(wv, "time", "time [s]", "(a) multipliers, time"),
                                                (wv, "memory", "peak memory [MB]", "(b) multipliers, memory"),
                                                (ly, "time", "time [s]", "(c) Lyapunov, time"),
                                                (ly, "memory", "peak memory [MB]", "(d) Lyapunov, memory")]):
        ax = axs[k]
        methods = []
        for r in data:
            if r["method"] not in methods:
                methods.append(r["method"])
        for mth in methods:
            rr = sorted([r for r in data if r["method"] == mth], key=lambda r: r["N"])
            x = [r["N"] for r in rr]
            y = [r[key] / (2 ** 20 if key == "memory" else 1) for r in rr]
            lab = "SBR (ours)" if mth == "SBR" else mth
            ln, = ax.loglog(x, y, marker=marker.get(mth, "o"), ms=3.2, color=col.get(mth, "k"), label=lab,
                            lw=1.5 if mth == "SBR" else 0.9)
            handles.setdefault(lab, ln)
        ax.set_xlabel("lattice size $N$", fontsize=7.5, labelpad=1)
        ax.set_ylabel(ylab, fontsize=7.5, labelpad=1)
        ax.set_title(ttl, fontsize=7.5)
        ax.tick_params(labelsize=6.5, pad=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=0.4)
    fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=9,
               frameon=False, fontsize=6.8, handlelength=1.4, columnspacing=0.8, handletextpad=0.4)
    save(fig, "fig_bench")


if __name__ == "__main__":
    which = sys.argv[1:] or ["bloch_multi", "chaos", "grid", "surface", "route", "bench"]
    for w in which:
        globals()["fig_" + w]()
        print("done", w, flush=True)
