from common import *
import math
from sublattice.hopf import first_hopf

F_PD = 3.937853


def hopf_table():
    h = load_json("hopf.json")
    lines = []
    for r in h["horizons"]:
        p, l = r["p"], r["l"]
        a, b = r["a"], r["b"]
        thp = l / p
        if abs(thp - a) < 1e-12:
            other, bound = b, 1.0 / (p * b - l)
        else:
            other, bound = a, 1.0 / (l - p * a)
        FH = 1.0 / (math.cos(2 * math.pi * l / p) - math.cos(4 * math.pi * l / p))
        mh = 0
        ties = []
        for m in range(1, 41):
            FHm, crit = first_hopf(p * m)
            if any(c["l"] == l * m for c in crit) and len(crit) > 1:
                ties.append(m)
                break
            if len(crit) == 1 and crit[0]["l"] == l * m:
                mh = m
            else:
                break
        lines.append(f"{p} & {l} & {FH:.6f} & {360 * thp:.3f} & {360 * other:.3f} & {bound:.4f} & {mh}"
                     + (f" ({ties[0]})" if ties else "") + r" \\")
    rows = [r for r in h["rows"] if not r["hopf_hopf"]]
    ok = sum(1 for r in rows if r["modes"][0]["spatial_period"] == r.get("observed_period"))
    return "\n".join(lines), ok, len(rows)


def t144():
    rows = load_json("table144.json")
    out = []
    for r in rows:
        out.append(f"{r['d']} & {r['m']} & {r['l']} & {r['T']:.4f} & {r['D']:.6f} & {r['rho_roots']:.6f} & "
                   f"{'stable' if r['stable'] else 'unstable'} & {r['final_dev']:.1e} & {r['seen_random']}" + r" \\")
    return "\n".join(out)


def route_table():
    rows = load_json("route_thresholds.json")
    out = []
    for r in rows:
        th = r["theta_first"]
        frac = "--" if th is None else f"{th / np.pi:.4f}"
        out.append(f"{r['m']} & {5 * r['m']} & {r['lo']:.4f} & {frac} & {r['m'] ** 2 * (r['lo'] - F_PD):.3f}" + r" \\")
    return "\n".join(out)


def pd_table():
    rows = load_json("phasediff.json")
    out = []
    for r in rows:
        out.append(f"{r['d']} & {r['l']} & {r['T']:.4f} & {r['c']:.6f} & {r['D_dyson']:.7f} & {r['D_green']:.7f} & {r['D_fd'][2]:.7f}" + r" \\")
    return "\n".join(out)


def bench_tables():
    wv = load_json("bench_wave.json")
    ly = load_json("bench_lyap.json")
    def fmt(data, methods, err=False):
        Ns = sorted(set(r["N"] for r in data))
        out = []
        for N in Ns:
            cells = [str(N)]
            for mth in methods:
                rr = [r for r in data if r["N"] == N and r["method"] == mth]
                if rr:
                    r = rr[0]
                    cells.append(f"{r['time']:.3g} / {r['memory'] / 2 ** 20:.3g}")
                else:
                    cells.append("--")
            out.append(" & ".join(cells) + r" \\")
        return "\n".join(out)
    return fmt(wv, ["SBR", "RM", "FM", "IRR", "SBD", "NK", "HILL"]), fmt(ly, ["SBR", "BQR", "CQR", "IRR", "SBD"])


if __name__ == "__main__":
    which = sys.argv[1:] or ["hopf", "t144", "route", "pd"]
    if "hopf" in which:
        s, ok, n = hopf_table()
        print(s)
        print("validated", ok, "of", n)
    if "t144" in which:
        print(t144())
    if "route" in which:
        print(route_table())
    if "pd" in which:
        print(pd_table())
    if "bench" in which:
        a, b = bench_tables()
        print(a)
        print(b)
