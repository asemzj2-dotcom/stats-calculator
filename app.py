
# -*- coding: utf-8 -*-
# =====================================================================
#  Advanced Statistics & Probability  -  Streamlit app  (15 formulas)
# =====================================================================
import re
from math import sqrt, ceil
from html import escape as esc

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import norm, t as tdist, chi2

st.set_page_config(page_title="Advanced Statistics & Probability", page_icon="📊", layout="wide")

plt.rcParams.update({
    "axes.grid": True, "grid.alpha": .25, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---------------------------------------------------------------- style
RED, GREEN, BLUE, ORANGE = "#C00000", "#00B050", "#0070C0", "#ED7D31"
XB = "X\u0304"      # X̄
PH = "P\u0302"      # P̂


# ------------------------------------------------------------- helpers
TABLE = False   # True -> round z to 2 decimals and use table values, like the lectures


def tr(x):
    """critical value (z, t, chi-square) as printed in the tables: 3 decimals"""
    return round(float(x), 3) if TABLE else float(x)


def zr(z):
    """z as the lecture uses it (2 decimals) when TABLE mode is on"""
    return round(float(z), 2) + 0.0 if TABLE else z


def zi(z):
    """z found backwards from a table (inverse): nearest 0.005, i.e. interpolating between table rows"""
    return round(round(float(z) / 0.005) * 0.005, 3) + 0.0 if TABLE else z


def prob_z(pt, za, zb):
    za, zb = zr(za), zr(zb)
    if pt == "lt":
        return float(norm.cdf(za))
    if pt == "gt":
        return float(norm.sf(za))
    need(za < zb, "a must be smaller than b")
    return float(norm.cdf(zb) - norm.cdf(za))


def g(x):
    """compact number"""
    return f"{x:.6g}"


def p_(x):
    """parenthesise negatives"""
    return f"({g(x)})" if x < 0 else g(x)


def f4(x):
    return f"{x:.4f}"


def tex(_template, **k):
    """tiny template:  @key@  ->  value"""
    for key, val in k.items():
        _template = _template.replace("@" + key + "@", str(val))
    return _template


def need(cond, msg):
    if not cond:
        raise ValueError(msg)


def sd_of(v):
    x = v["sv"]
    need(x > 0, "σ (or σ²) must be greater than 0")
    return sqrt(x) if v.get("stype", "sd") == "var" else x


def conf_of(v):
    c = v["cc"] if v["cp"] == "Other" else float(v["cp"].rstrip("%"))
    need(0 < c < 100, "Confidence level must be between 0 and 100")
    return c / 100


def alpha_step(conf):
    al = 1 - conf
    return tex(r"1-\alpha=@c@\;\Rightarrow\;\alpha=@a@\;\Rightarrow\;\frac{\alpha}{2}=@h@",
               c=g(round(conf, 8)), a=g(round(al, 8)), h=g(round(al / 2, 8)))


def prob_of(dist, pt, a, b):
    if pt == "lt":
        return float(dist.cdf(a))
    if pt == "gt":
        return float(dist.sf(a))
    need(a < b, "a must be smaller than b")
    return float(dist.cdf(b) - dist.cdf(a))


def pt_opts(sym):
    return [(f"P({sym} < a)", "lt"), (f"P({sym} > a)", "gt"), (f"P(a < {sym} < b)", "bt")]


def evt(sym, pt, a, b, sp=""):
    return {"lt": f"{sym}{sp}<{sp}{g(a)}", "gt": f"{sym}{sp}>{sp}{g(a)}",
            "bt": f"{g(a)}{sp}<{sp}{sym}{sp}<{sp}{g(b)}"}[pt]


def std_chain(sym, zs, pt, a, b, mu, se, prob, phi=True):
    """P(sym < a) = P(Z < (a-mu)/se) = ...   (lecture style)"""
    za, zb = (a - mu) / se, (b - mu) / se
    if phi:
        za, zb = zr(za), zr(zb)
    fr = lambda x: tex(r"\frac{@x@-@m@}{@s@}", x=g(x), m=p_(mu), s=g(se))
    Z = lambda z: f"{z:.2f}"
    if pt == "lt":
        s = rf"P({sym}<{g(a)})=P\left({zs}<{fr(a)}\right)=P({zs}<{Z(za)})"
        s += rf"=\varphi({Z(za)})={f4(prob)}" if phi else rf"={f4(prob)}"
    elif pt == "gt":
        s = rf"P({sym}>{g(a)})=P\left({zs}>{fr(a)}\right)=P({zs}>{Z(za)})"
        s += rf"=1-\varphi({Z(za)})=1-{f4(norm.cdf(za))}={f4(prob)}" if phi else rf"={f4(prob)}"
    else:
        s = (rf"P({g(a)}<{sym}<{g(b)})=P\left({fr(a)}<{zs}<{fr(b)}\right)"
             rf"=P({Z(za)}<{zs}<{Z(zb)})")
        s += (rf"=\varphi({Z(zb)})-\varphi({Z(za)})={f4(norm.cdf(zb))}-{f4(norm.cdf(za))}={f4(prob)}"
              if phi else rf"={f4(prob)}")
    return [s]


def plain_chain(sym, pt, a, b, dist, prob):
    if pt == "lt":
        return [rf"P({sym}<{g(a)})={f4(prob)}"]
    if pt == "gt":
        return [rf"P({sym}>{g(a)})=1-P({sym}<{g(a)})=1-{f4(dist.cdf(a))}={f4(prob)}"]
    return [rf"P({g(a)}<{sym}<{g(b)})=P({sym}>{g(a)})-P({sym}>{g(b)})"
            rf"={f4(dist.sf(a))}-{f4(dist.sf(b))}={f4(prob)}"]


# ------------------------------------------------------------- plotting
def _range(dist, pts):
    lo = min([dist.ppf(.003)] + list(pts))
    hi = max([dist.ppf(.997)] + list(pts))
    pad = (hi - lo) * .08
    lo, hi = lo - pad, hi + pad
    try:
        s_lo = dist.support()[0]
        if np.isfinite(s_lo):
            lo = max(lo, s_lo)
    except Exception:
        pass
    return lo, hi


def _grid(lo, hi, n=900):
    xs = np.linspace(lo, hi, n)
    if lo == 0:
        xs[0] = hi * 1e-3
    return xs


def prob_plot(dist, pt, a, b, prob, xlabel, title, marks=("a", "b"), ref=None, ref_label=None):
    pts = [a] if pt != "bt" else [a, b]
    lo, hi = _range(dist, pts)
    xs = _grid(lo, hi)
    ys = dist.pdf(xs)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, color=BLUE, lw=2.3, label="density")
    if ref is not None:
        ax.plot(xs, ref.pdf(xs), color="gray", ls="--", lw=1.5, label=ref_label)
    m = {"lt": xs <= a, "gt": xs >= a, "bt": (xs >= a) & (xs <= (b if pt == "bt" else a))}[pt]
    ax.fill_between(xs[m], ys[m], color=ORANGE, alpha=.55, label="required area")
    for x, l in zip(pts, marks):
        ax.axvline(x, color="k", ls="--", lw=1)
        l = f"{l} = {x:.4g}" if l in ("a", "b") else l
        ax.text(x, 1.01, l, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    ax.text(.98, .92, f"P = {prob:.4f}", transform=ax.transAxes, ha="right", va="top", fontsize=13,
            fontweight="bold", color=RED, bbox=dict(boxstyle="round", fc="white", ec=RED))
    ax.set_ylim(bottom=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("f(x)")
    ax.set_title(title, color=RED, fontweight="bold", pad=22)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def ci_figure(dist, c_lo, c_hi, lo, hi, point, point_lbl, top_x, bot_x, title, c_lbls, conf):
    a2 = (1 - conf) / 2
    lo_x, hi_x = _range(dist, [c_lo, c_hi])
    xs = _grid(lo_x, hi_x)
    ys = dist.pdf(xs)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), gridspec_kw={"height_ratios": [3, 1.25]})
    ax.plot(xs, ys, color=BLUE, lw=2.3)
    mid = (xs >= c_lo) & (xs <= c_hi)
    ax.fill_between(xs[mid], ys[mid], color=BLUE, alpha=.12, label=f"1 − α = {conf:.4g}")
    ax.fill_between(xs[xs <= c_lo], ys[xs <= c_lo], color=ORANGE, alpha=.65, label=f"α/2 = {a2:.4g}  (each tail)")
    ax.fill_between(xs[xs >= c_hi], ys[xs >= c_hi], color=ORANGE, alpha=.65)
    for x, l in zip([c_lo, c_hi], c_lbls):
        ax.axvline(x, color="k", ls="--", lw=1)
        ax.text(x, 1.01, f"{l} = {x:.3f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(top_x)
    ax.set_ylabel("f(x)")
    ax.set_title(title, color=RED, fontweight="bold", pad=22)
    ax.legend(loc="upper left")
    # --- interval on a number line
    span = hi - lo
    ax2.hlines(0, lo, hi, color=BLUE, lw=7)
    ax2.plot([lo, hi], [0, 0], "|", color=BLUE, ms=26, mew=3)
    ax2.plot(point, 0, "o", color=RED, ms=9, label=point_lbl)
    ax2.text(lo, .45, f"{lo:.4g}", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax2.text(hi, .45, f"{hi:.4g}", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax2.text(point, -.75, f"{point:.4g}", ha="center", fontsize=10, color=RED)
    ax2.set_xlim(lo - span * .6, hi + span * .6)
    ax2.set_ylim(-1, 1)
    ax2.set_yticks([])
    ax2.grid(False)
    ax2.set_xlabel(bot_x)
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# =====================================================================
#                              THE LAWS
# =====================================================================
# ---------- Lecture 1 : Normal ----------------------------------------
def law_normal(v, count=False):
    mu, s, a, b, pt = v["mu"], sd_of(v), v["a"], v["b"], v["pt"]
    d = norm(mu, s)
    prob = prob_of(d, pt, a, b)
    if TABLE:
        prob = prob_z(pt, (a - mu) / s, (b - mu) / s)
    steps = [r"Z=\frac{X-\mu}{\sigma}\sim N(0,1)",
             tex(r"\mu=@m@,\qquad \sigma=@s@", m=g(mu), s=g(s))]
    steps += std_chain("X", "Z", pt, a, b, mu, s, prob)
    res = f"P({evt('X', pt, a, b, ' ')}) = {prob:.4f}   ({prob * 100:.2f}%)"
    if count:
        N = int(v["N"])
        need(N > 0, "N must be positive")
        cnt = prob * N
        steps.append(tex(r"\text{Number of individuals}=P\times N=@p@\times @N@=@c@\approx @r@",
                         p=f4(prob), N=N, c=f"{cnt:.2f}", r=int(round(cnt))))
        res += f"   →   ≈ {int(round(cnt))} out of {N}"
    plot = lambda: prob_plot(d, pt, a, b, prob, "x", "Normal distribution  N(μ, σ²)")
    return steps, res, [], plot


def law_inverse(v):
    mu, s, side, p = v["mu"], sd_of(v), v["side"], v["p"]
    need(0 < p < 1, "p must be strictly between 0 and 1")
    phi_val = p if side == "lt" else 1 - p
    z = zi(float(norm.ppf(phi_val)))
    k = mu + z * s
    steps = [r"Z=\frac{X-\mu}{\sigma}", tex(r"\mu=@m@,\qquad \sigma=@s@", m=g(mu), s=g(s))]
    if side == "lt":
        steps.append(tex(r"P(X<k)=\varphi\left(\frac{k-@m@}{@s@}\right)=@p@", m=p_(mu), s=g(s), p=g(p)))
    else:
        steps.append(tex(r"P(X>k)=1-\varphi\left(\frac{k-@m@}{@s@}\right)=@p@\;\Rightarrow\;\varphi\left(\frac{k-@m@}{@s@}\right)=@q@",
                         m=p_(mu), s=g(s), p=g(p), q=g(round(1 - p, 8))))
    steps.append(tex(r"\frac{k-@m@}{@s@}=@z@\;\Rightarrow\;k-@m@=@zs@\;\Rightarrow\;k=@k@",
                     m=p_(mu), s=g(s), z=f"{z:.3f}", zs=f"{z * s:.4f}", k=f"{k:.4f}"))
    res = f"k = {k:.4f}"
    d = norm(mu, s)
    plot = lambda: prob_plot(d, side, k, k, p, "x", "Inverse normal:  find k", marks=(f"k = {k:.3f}", ""))
    return steps, res, [], plot


# ---------- Lecture 2 -------------------------------------------------
def law_t(v):
    n, pt, a, b = int(v["n"]), v["pt"], v["a"], v["b"]
    need(n >= 2, "n must be at least 2")
    df = n - 1
    d = tdist(df)
    prob = prob_of(d, pt, a, b)
    steps = [r"T=\frac{\bar{X}-\mu}{s/\sqrt{n}}\sim T_{(n-1)}",
             tex(r"n=@n@\;\Rightarrow\; df=n-1=@d@", n=n, d=df)]
    steps += plain_chain("T", pt, a, b, d, prob)
    res = f"P({evt('T', pt, a, b, ' ')}) = {prob:.4f}   (df = {df})"
    plot = lambda: prob_plot(d, pt, a, b, prob, "t", f"Student T-distribution  (df = {df})",
                             ref=norm(0, 1), ref_label="N(0,1)  for comparison")
    return steps, res, [], plot


def law_chi(v):
    k, pt, a, b = int(v["k"]), v["pt"], v["a"], v["b"]
    need(k >= 1, "Degrees of freedom k must be at least 1")
    d = chi2(k)
    prob = prob_of(d, pt, a, b)
    steps = [r"\chi^2=Z_1^2+Z_2^2+\cdots+Z_k^2",
             tex(r"E(\chi^2)=k=@k@,\qquad V(\chi^2)=2k=@v@", k=k, v=2 * k)]
    steps += plain_chain(r"\chi^2", pt, a, b, d, prob)
    res = f"P({evt('χ²', pt, a, b, ' ')}) = {prob:.4f}   (k = {k})"
    plot = lambda: prob_plot(d, pt, a, b, prob, "χ²", f"χ² distribution  (k = {k})")
    return steps, res, [], plot


def law_xbar(v):
    mu, sv, n, pt, a, b = v["mu"], v["sv"], int(v["n"]), v["pt"], v["a"], v["b"]
    need(sv > 0, "σ (or s) must be greater than 0")
    need(n >= 2, "n must be at least 2")
    se = sv / sqrt(n)
    warns = []
    if v["sk"] == "known":
        dist, zs, phi, S = norm(mu, se), "Z", True, r"\sigma"
        stat = r"Z=\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\sim N(0,1)"
        if n < 30:
            warns.append("n < 30 : this is valid only if the population is normally distributed.")
    elif n >= 30:
        dist, zs, phi, S = norm(mu, se), "Z", True, "s"
        stat = r"Z=\frac{\bar{X}-\mu}{s/\sqrt{n}}\sim N(0,1)\quad (n\geq 30)"
    else:
        dist, zs, phi, S = tdist(n - 1, loc=mu, scale=se), "T", False, "s"
        stat = r"T=\frac{\bar{X}-\mu}{s/\sqrt{n}}\sim T_{(n-1)}\quad (n<30)"
        warns.append("n < 30 and σ unknown : T-distribution is used (population must be approximately normal).")
    prob = prob_of(dist, pt, a, b)
    if TABLE and zs == "Z":
        prob = prob_z(pt, (a - mu) / se, (b - mu) / se)
    steps = [stat, tex(r"\mu=@m@,\quad n=@n@,\quad \frac{@S@}{\sqrt{n}}=\frac{@sv@}{\sqrt{@n@}}=@se@",
                       m=g(mu), n=n, S=S, sv=g(sv), se=f"{se:.4f}")]
    steps += std_chain(r"\bar{X}", zs, pt, a, b, mu, se, prob, phi=phi)
    res = f"P({evt(XB, pt, a, b, ' ')}) = {prob:.4f}   ({prob * 100:.2f}%)"
    plot = lambda: prob_plot(dist, pt, a, b, prob, XB, f"Sampling distribution of {XB}  ({'Z' if zs == 'Z' else 'T'})",
                             marks=(f"a = {g(a)}", f"b = {g(b)}"))
    return steps, res, warns, plot


# ---------- Lecture 3 -------------------------------------------------
def law_s2(v):
    n, var, pt, a, b = int(v["n"]), v["var"], v["pt"], v["a"], v["b"]
    need(n >= 2, "n must be at least 2")
    need(var > 0, "σ² must be greater than 0")
    df = n - 1
    d = chi2(df)
    ca, cb = a * df / var, b * df / var
    prob = prob_of(d, pt, ca, cb)
    steps = [r"\chi^2=\frac{(n-1)s^2}{\sigma^2}\sim\chi^2_{(n-1)}",
             tex(r"n=@n@,\quad \sigma^2=@v@,\quad df=@d@", n=n, v=g(var), d=df),
             tex(r"\chi^2_a=\frac{@a@(@n@-1)}{@v@}=@c@", a=g(a), n=n, v=g(var), c=f"{ca:.4f}")]
    if pt == "bt":
        steps.append(tex(r"\chi^2_b=\frac{@b@(@n@-1)}{@v@}=@c@", b=g(b), n=n, v=g(var), c=f"{cb:.4f}"))
    steps += plain_chain(r"\chi^2", pt, ca, cb, d, prob)
    res = f"P({evt('s²', pt, a, b, ' ')}) = {prob:.4f}"
    plot = lambda: prob_plot(d, pt, ca, cb, prob, "χ² = (n−1)s²/σ²", f"Sampling distribution of s²  (χ², df = {df})",
                             marks=(f"s² = {g(a)}", f"s² = {g(b)}"))
    return steps, res, [], plot


def law_phat(v):
    P, n, pt, a, b = v["P"], int(v["n"]), v["pt"], v["a"], v["b"]
    need(0 < P < 1, "P must be strictly between 0 and 1")
    need(n >= 2, "n must be at least 2")
    se0 = sqrt(P * (1 - P) / n)
    se = round(se0, 3) if TABLE else se0
    warns = []
    if n * P < 5 or n * (1 - P) < 5:
        warns.append(f"Normal approximation condition fails:  nP = {n * P:.2f},  n(1−P) = {n * (1 - P):.2f}  (both must be ≥ 5).")
    d = norm(P, se)
    prob = prob_of(d, pt, a, b)
    if TABLE:
        prob = prob_z(pt, (a - P) / se, (b - P) / se)
    steps = [r"E(\hat{P})=P,\qquad V(\hat{P})=\frac{P(1-P)}{n}",
             r"Z=\frac{\hat{P}-P}{\sqrt{P(1-P)/n}}\sim N(0,1)",
             tex(r"V(\hat{P})=\frac{@P@(1-@P@)}{@n@}=@V@,\qquad \text{SE}=\sqrt{V(\hat{P})}=@se@",
                 P=g(P), n=n, V=f"{se0 ** 2:.6f}", se=(f"{se:.3f}" if TABLE else f"{se:.4f}"))]
    steps += std_chain(r"\hat{P}", "Z", pt, a, b, P, se, prob)
    res = f"P({evt(PH, pt, a, b, ' ')}) = {prob:.4f}   ({prob * 100:.2f}%)"
    plot = lambda: prob_plot(d, pt, a, b, prob, PH, f"Sampling distribution of {PH}  (mean = P, SE = {se:.4f})",
                             marks=(f"a = {g(a)}", f"b = {g(b)}"))
    return steps, res, warns, plot


# ---------- Lecture 4 : CI for the mean ------------------------------
def mean_ci_core(xbar, sd, n, conf, kind, pre, warns):
    al = 1 - conf
    a2 = al / 2
    S = r"\sigma" if kind == "z_known" else "s"
    if kind == "t":
        crit, dist = tr(tdist.ppf(1 - a2, n - 1)), tdist(n - 1)
        form = tex(r"\bar{X}-t_{(\frac{\alpha}{2},\,n-1)}\frac{s}{\sqrt{n}}<\mu<\bar{X}+t_{(\frac{\alpha}{2},\,n-1)}\frac{s}{\sqrt{n}}")
        cr = tex(r"t_{(\frac{\alpha}{2},\,n-1)}=t_{(@a@,\,@d@)}=@c@", a=g(round(a2, 8)), d=n - 1, c=f"{crit:.4f}")
        cl, cname = "t", "t"
    else:
        crit, dist = tr(norm.ppf(1 - a2)), norm(0, 1)
        form = tex(r"\bar{X}-z_{\frac{\alpha}{2}}\frac{@S@}{\sqrt{n}}<\mu<\bar{X}+z_{\frac{\alpha}{2}}\frac{@S@}{\sqrt{n}}", S=S)
        cr = tex(r"z_{\frac{\alpha}{2}}=z_{@a@}=@c@", a=g(round(a2, 8)), c=f"{crit:.4f}")
        cl, cname = "z", "z"
    se = sd / sqrt(n)
    me = crit * se
    lo, hi = xbar - me, xbar + me
    steps = [form] + pre + [
        tex(r"n=@n@,\quad \bar{X}=@x@,\quad @S@=@s@", n=n, x=g(xbar), S=S, s=f"{sd:.4f}"),
        alpha_step(conf), cr,
        tex(r"@x@\pm @c@\cdot\frac{@s@}{\sqrt{@n@}}=@x@\pm @c@\cdot @se@=@x@\pm @me@",
            x=g(xbar), c=f"{crit:.4f}", s=f"{sd:.4f}", n=n, se=f"{se:.4f}", me=f"{me:.4f}"),
        tex(r"@lo@<\mu<@hi@", lo=f"{lo:.4f}", hi=f"{hi:.4f}")]
    res = f"{lo:.4f}  <  μ  <  {hi:.4f}      ( {conf * 100:g}% confidence )"
    lbl = (f"−{cname}(α/2)", f"+{cname}(α/2)")
    plot = lambda: ci_figure(dist, -crit, crit, lo, hi, xbar, f"{XB} = {xbar:g}", f"{cl}", "μ  (population mean)",
                             f"{conf * 100:g}% confidence interval for μ", lbl, conf)
    return steps, res, warns, plot


def law_ci_mean(kind):
    def fn(v):
        n = int(v["n"])
        need(n >= 2, "n must be at least 2")
        conf = conf_of(v)
        sd = sd_of(v) if kind == "z_known" else v["sv"]
        need(sd > 0, "σ / s must be greater than 0")
        warns = []
        if kind == "z_known" and n < 30:
            warns.append("n < 30 : valid only if the population is normally distributed.")
        if kind == "z_s" and n < 30:
            warns.append("n < 30 : the lecture rule uses the T-distribution here (use formula 13).")
        if kind == "t" and n >= 30:
            warns.append("n ≥ 30 : Z is usually used (formula 12), but T is still valid and gives a slightly wider interval.")
        return mean_ci_core(v["xbar"], sd, n, conf, kind, [], warns)
    return fn


def law_ci_sums(v):
    n, sx, sxx = int(v["n"]), v["sx"], v["sxx"]
    need(n >= 2, "n must be at least 2")
    conf = conf_of(v)
    xbar = sx / n
    s2 = (sxx - sx ** 2 / n) / (n - 1)
    need(s2 > 0, "Computed s² is not positive — check Σx and Σx².")
    s = sqrt(s2)
    pre = [tex(r"\bar{X}=\frac{\sum x_i}{n}=\frac{@sx@}{@n@}=@x@", sx=g(sx), n=n, x=g(round(xbar, 6))),
           tex(r"s^2=\frac{1}{n-1}\left[\sum x_i^2-\frac{(\sum x_i)^2}{n}\right]=\frac{1}{@m@}\left[@sxx@-\frac{@sx@^2}{@n@}\right]=@s2@\;\Rightarrow\; s=@s@",
               m=n - 1, sxx=g(sxx), sx=g(sx), n=n, s2=f"{s2:.4f}", s=f"{s:.4f}")]
    kind = "z_s" if n >= 30 else "t"
    warns = [f"n = {n}  →  {'Z (n ≥ 30)' if n >= 30 else 'T (n < 30, population approximately normal)'} is used automatically."]
    return mean_ci_core(xbar, s, n, conf, kind, pre, warns)


def law_sample_size(v):
    sd, e = sd_of(v), v["e"]
    need(e > 0, "e (margin of error) must be greater than 0")
    conf = conf_of(v)
    z = tr(norm.ppf(1 - (1 - conf) / 2))
    n_exact = (z * sd / e) ** 2
    n = ceil(n_exact)
    steps = [r"\text{SE}=z_{\frac{\alpha}{2}}\frac{\sigma}{\sqrt{n}}=e\;\Rightarrow\; n=\left(\frac{z_{\alpha/2}\,\sigma}{e}\right)^2",
             alpha_step(conf),
             tex(r"z_{\frac{\alpha}{2}}=@z@,\qquad \sigma=@s@,\qquad e=@e@", z=f"{z:.4f}", s=g(sd), e=g(e)),
             tex(r"@e@=@z@\cdot\frac{@s@}{\sqrt{n}}\;\Rightarrow\;\sqrt{n}=@r@\;\Rightarrow\;n=@n@",
                 e=g(e), z=f"{z:.4f}", s=g(sd), r=f"{z * sd / e:.4f}", n=f"{n_exact:.2f}"),
             tex(r"n=\lceil @n@\rceil=@c@", n=f"{n_exact:.2f}", c=n)]
    warns = []
    if n < 30:
        warns.append("n < 30 : Z-based sample size is only approximate (T would need iteration).")
    res = f"Required sample size  n = {n}"

    def plot():
        es = np.linspace(e * .4, e * 2.5, 300)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(es, (z * sd / es) ** 2, color=BLUE, lw=2.3, label="n = (zσ / e)²")
        ax.plot(e, n_exact, "o", color=RED, ms=9)
        ax.annotate(f"e = {g(e)}\nn = {n}", (e, n_exact), textcoords="offset points", xytext=(15, 15), color=RED)
        ax.set_xlabel("margin of error  e")
        ax.set_ylabel("required n")
        ax.set_title(f"Sample size vs margin of error  ({conf * 100:g}% confidence)", color=RED, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.show()
    return steps, res, warns, plot


# ---------- Lecture 5 -------------------------------------------------
def law_ci_prop(v):
    n = int(v["n"])
    need(n >= 2, "n must be at least 2")
    conf = conf_of(v)
    pre = []
    if v["mode"] == "x":
        x = int(v["x"])
        need(0 <= x <= n, "x must be between 0 and n")
        ph = x / n
        pre.append(tex(r"\hat{P}=\frac{x}{n}=\frac{@x@}{@n@}=@p@", x=x, n=n, p=g(round(ph, 6))))
    else:
        ph = v["ph"]
        need(0 <= ph <= 1, "P̂ must be between 0 and 1")
    al = 1 - conf
    z = tr(norm.ppf(1 - al / 2))
    se = sqrt(ph * (1 - ph) / n)
    me = z * se
    lo, hi = ph - me, ph + me
    warns = []
    if n * ph < 5 or n * (1 - ph) < 5:
        warns.append(f"Normal approximation condition fails:  nP̂ = {n * ph:.2f},  n(1−P̂) = {n * (1 - ph):.2f}  (should be ≥ 5).")
    if lo < 0 or hi > 1:
        warns.append("The interval extends outside [0, 1].")
    steps = [r"\hat{P}-z_{\frac{\alpha}{2}}\sqrt{\frac{\hat{P}(1-\hat{P})}{n}}<P<\hat{P}+z_{\frac{\alpha}{2}}\sqrt{\frac{\hat{P}(1-\hat{P})}{n}}"] + pre + [
        tex(r"n=@n@,\qquad \hat{P}=@p@", n=n, p=g(round(ph, 6))),
        alpha_step(conf),
        tex(r"z_{\frac{\alpha}{2}}=z_{@a@}=@z@", a=g(round(al / 2, 8)), z=f"{z:.4f}"),
        tex(r"@p@\pm @z@\sqrt{\frac{@p@(1-@p@)}{@n@}}=@p@\pm @z@\cdot @se@=@p@\pm @me@",
            p=g(round(ph, 6)), z=f"{z:.4f}", n=n, se=f"{se:.4f}", me=f"{me:.4f}"),
        tex(r"@lo@<P<@hi@", lo=f"{lo:.4f}", hi=f"{hi:.4f}")]
    res = f"{lo:.4f}  <  P  <  {hi:.4f}      ( {conf * 100:g}% confidence )"
    plot = lambda: ci_figure(norm(0, 1), -z, z, lo, hi, ph, f"{PH} = {ph:.4g}", "z", "P  (population proportion)",
                             f"{conf * 100:g}% confidence interval for P", ("−z(α/2)", "+z(α/2)"), conf)
    return steps, res, warns, plot


def parse_nums(txt):
    items = [x for x in re.split(r"[,\s;]+", txt.strip()) if x]
    try:
        return [float(x) for x in items]
    except ValueError:
        raise ValueError("Raw data must be numbers separated by commas, spaces or new lines.")


def law_ci_var(v):
    conf = conf_of(v)
    mode = v["mode"]
    pre = []
    if mode == "raw":
        data = parse_nums(v["data"])
        n = len(data)
        need(n >= 2, "Enter at least 2 numbers")
        sx, sxx = sum(data), sum(x * x for x in data)
        s2 = float(np.var(data, ddof=1))
        need(s2 > 0, "All values are identical (s² = 0).")
        pre += [tex(r"\sum x_i=@a@,\qquad \sum x_i^2=@b@,\qquad n=@n@", a=g(round(sx, 6)), b=g(round(sxx, 6)), n=n),
                tex(r"s^2=\frac{1}{n-1}\left[\sum x_i^2-\frac{(\sum x_i)^2}{n}\right]=@s@", s=f"{s2:.5f}")]
    else:
        n = int(v["n"])
        need(n >= 2, "n must be at least 2")
        sv = v["sv"]
        need(sv > 0, "s² / s must be greater than 0")
        s2 = sv if mode == "s2" else sv ** 2
        if mode == "s":
            pre.append(tex(r"s=@s@\;\Rightarrow\; s^2=@s2@", s=g(sv), s2=g(round(s2, 8))))
    df = n - 1
    al = 1 - conf
    up = tr(chi2.ppf(1 - al / 2, df))     # chi2_(alpha/2, n-1)
    low = tr(chi2.ppf(al / 2, df))        # chi2_(1-alpha/2, n-1)
    lo, hi = df * s2 / up, df * s2 / low
    steps = [r"\frac{(n-1)s^2}{\chi^2_{(\frac{\alpha}{2},\,n-1)}}<\sigma^2<\frac{(n-1)s^2}{\chi^2_{(1-\frac{\alpha}{2},\,n-1)}}"] + pre + [
        tex(r"n=@n@,\quad df=n-1=@d@,\quad s^2=@s2@", n=n, d=df, s2=f"{s2:.5f}"),
        alpha_step(conf),
        tex(r"\chi^2_{(\frac{\alpha}{2},\,n-1)}=\chi^2_{(@a@,\,@d@)}=@u@,\qquad \chi^2_{(1-\frac{\alpha}{2},\,n-1)}=\chi^2_{(@b@,\,@d@)}=@l@",
            a=g(round(al / 2, 8)), b=g(round(1 - al / 2, 8)), d=df, u=f"{up:.4f}", l=f"{low:.4f}"),
        tex(r"\frac{@d@\cdot @s2@}{@u@}<\sigma^2<\frac{@d@\cdot @s2@}{@l@}", d=df, s2=f"{s2:.5f}", u=f"{up:.4f}", l=f"{low:.4f}"),
        tex(r"@lo@<\sigma^2<@hi@", lo=f"{lo:.4f}", hi=f"{hi:.4f}"),
        tex(r"@a@<\sigma<@b@", a=f"{sqrt(lo):.4f}", b=f"{sqrt(hi):.4f}")]
    res = f"{lo:.4f}  <  σ²  <  {hi:.4f}      ( {conf * 100:g}% confidence )"
    warns = ["Valid only when the population is normally distributed."]
    plot = lambda: ci_figure(chi2(df), low, up, lo, hi, s2, f"s² = {s2:.4g}", "χ²", "σ²  (population variance)",
                             f"{conf * 100:g}% confidence interval for σ²  (df = {df})",
                             ("χ²(1−α/2)", "χ²(α/2)"), conf)
    return steps, res, warns, plot


# =====================================================================
#                       FIELD DEFINITIONS & REGISTRY
# =====================================================================
def FL(k, label, default=0.0, show=None): return dict(k=k, label=label, kind="float", default=default, show=show)
def IN(k, label, default=10, show=None): return dict(k=k, label=label, kind="int", default=default, show=show)
def CH(k, label, opts, default, show=None): return dict(k=k, label=label, kind="choice", opts=opts, default=default, show=show)
def TX(k, label, default="", show=None): return dict(k=k, label=label, kind="text", default=default, show=show)


CONF = [CH("cp", "Confidence level  (1−α)", ["90%", "95%", "98%", "99%", "99.9%", "Other"], "95%"),
        FL("cc", "Custom level (%)", 95.0, show=("cp", {"Other"}))]


def SIG(label="Value of σ or σ²", default=1.0):
    return [CH("stype", "Given as", [("σ  (standard deviation)", "sd"), ("σ²  (variance)", "var")], "sd"),
            FL("sv", label, default)]


def PROB(sym, a=0.0, b=1.0):
    return [CH("pt", "Probability type", pt_opts(sym), "lt"),
            FL("a", "a", a),
            FL("b", "b  (only for  a < … < b)", b, show=("pt", {"bt"}))]


L = lambda s: s  # readability

LAWS = [
    # ------------------------------------------------- Lecture 1
    dict(lec=0, name="1 · Normal probabilities   P(X<a), P(X>a), P(a<X<b)",
         formula=r"Z=\frac{X-\mu}{\sigma}\sim N(0,1),\quad P(X<a)=\varphi\!\left(\tfrac{a-\mu}{\sigma}\right),\quad P(X>a)=1-\varphi\!\left(\tfrac{a-\mu}{\sigma}\right)",
         req="X ~ N(μ, σ²).  Needs μ and σ (or σ²).",
         fields=[FL("mu", "μ  (mean)", 16)] + SIG(default=0.9) + PROB("X", 14, 18), fn=law_normal),
    dict(lec=0, name="2 · Find k   (inverse normal)   P(X<k)=p  or  P(X>k)=p",
         formula=r"\varphi\!\left(\frac{k-\mu}{\sigma}\right)=p\;\Rightarrow\;k=\mu+z\,\sigma",
         req="X ~ N(μ, σ²).  Needs μ, σ and the probability p.",
         fields=[FL("mu", "μ  (mean)", 12)] + SIG(default=3.0)
         + [CH("side", "Condition", [("P(X < k) = p", "lt"), ("P(X > k) = p", "gt")], "lt"),
            FL("p", "p  (probability)", 0.9875)], fn=law_inverse),
    dict(lec=0, name="3 · Expected number of individuals   ( P × N )",
         formula=r"\text{Number}=P(\cdot)\times N,\qquad P(\cdot)\ \text{from}\ Z=\frac{X-\mu}{\sigma}",
         req="X ~ N(μ, σ²) and a population of N individuals.",
         fields=[FL("mu", "μ  (mean)", 68)] + SIG(default=3.0) + [IN("N", "N  (population size)", 1000)] + PROB("X", 64, 71), fn=lambda v: law_normal(v, True)),
    # ------------------------------------------------- Lecture 2
    dict(lec=1, name="4 · Student T-distribution   P(T>t)",
         formula=r"T=\frac{\bar{X}-\mu}{s/\sqrt{n}}\sim T_{(n-1)},\qquad df=n-1",
         req="Small sample (n < 30), σ unknown, population approximately normal.  Symmetric, mean 0, heavier tails than Z.",
         fields=[IN("n", "n  (sample size)", 19)] + PROB("T", 2.101, 3.0), fn=law_t),
    dict(lec=1, name="5 · Chi-square distribution   χ²",
         formula=r"\chi^2=Z_1^2+\cdots+Z_k^2,\qquad E(\chi^2)=k,\quad V(\chi^2)=2k",
         req="k = degrees of freedom.  Non-negative and right-skewed.",
         fields=[IN("k", "k  (degrees of freedom)", 9)] + PROB("χ²", 16.919, 20.0), fn=law_chi),
    dict(lec=1, name="7 · Sampling distribution of the mean   X̄",
         formula=r"Z=\frac{\bar{X}-\mu}{\sigma/\sqrt{n}}\ (\sigma\ \text{known}),\quad Z=\frac{\bar{X}-\mu}{s/\sqrt{n}}\ (n\geq30),\quad T=\frac{\bar{X}-\mu}{s/\sqrt{n}}\ (n<30)",
         req="σ known: population normal or n ≥ 30.  σ unknown: n ≥ 30 → Z with s;  n < 30 and normal population → T(n−1).",
         fields=[FL("mu", "μ  (population mean)", 200),
                 CH("sk", "Population σ is", [("known  (use σ)", "known"), ("unknown  (use s)", "unknown")], "known"),
                 FL("sv", "σ  or  s", 6), IN("n", "n  (sample size)", 16)] + PROB(XB, 199, 202), fn=law_xbar),
    # ------------------------------------------------- Lecture 3
    dict(lec=2, name="8 · Sampling distribution of the variance   s²",
         formula=r"\chi^2=\frac{(n-1)s^2}{\sigma^2}\sim\chi^2_{(n-1)}",
         req="Random sample from a NORMAL population with known σ².",
         fields=[IN("n", "n  (sample size)", 25), FL("var", "σ²  (population variance)", 6)] + PROB("s²", 9.1, 12.0), fn=law_s2),
    dict(lec=2, name="10 · Sampling distribution of the proportion   P̂",
         formula=r"E(\hat{P})=P,\quad V(\hat{P})=\frac{P(1-P)}{n},\quad Z=\frac{\hat{P}-P}{\sqrt{P(1-P)/n}}\sim N(0,1)",
         req="Normal approximation needs  nP ≥ 5  and  n(1−P) ≥ 5.   (Enter P̂ values as proportions, e.g. 0.35)",
         fields=[FL("P", "P  (population proportion)", 0.3), IN("n", "n  (sample size)", 120)] + PROB(PH, 0.35, 0.4), fn=law_phat),
    # ------------------------------------------------- Lecture 4
    dict(lec=3, name="11 · CI for the mean   —   σ² known",
         formula=r"\bar{X}-z_{\frac{\alpha}{2}}\frac{\sigma}{\sqrt{n}}<\mu<\bar{X}+z_{\frac{\alpha}{2}}\frac{\sigma}{\sqrt{n}}",
         req="σ² known  and  (population normal  or  n ≥ 30).",
         fields=[FL("xbar", "X̄  (sample mean)", 15)] + SIG(default=4.0) + [IN("n", "n  (sample size)", 25)] + CONF, fn=law_ci_mean("z_known")),
    dict(lec=3, name="12 · CI for the mean   —   σ² unknown,  n ≥ 30",
         formula=r"\bar{X}-z_{\frac{\alpha}{2}}\frac{s}{\sqrt{n}}<\mu<\bar{X}+z_{\frac{\alpha}{2}}\frac{s}{\sqrt{n}}",
         req="σ² unknown  and  n ≥ 30.",
         fields=[FL("xbar", "X̄  (sample mean)", 2.6), FL("sv", "s  (sample std deviation)", 4.0), IN("n", "n  (sample size)", 36)] + CONF, fn=law_ci_mean("z_s")),
    dict(lec=3, name="13 · CI for the mean   —   σ² unknown,  n < 30",
         formula=r"\bar{X}-t_{(\frac{\alpha}{2},\,n-1)}\frac{s}{\sqrt{n}}<\mu<\bar{X}+t_{(\frac{\alpha}{2},\,n-1)}\frac{s}{\sqrt{n}}",
         req="σ² unknown,  population approximately normal,  n < 30.",
         fields=[FL("xbar", "X̄  (sample mean)", 7.38), FL("sv", "s  (sample std deviation)", 1.24), IN("n", "n  (sample size)", 12)] + CONF, fn=law_ci_mean("t")),
    dict(lec=3, name="14 · CI for the mean   —   from  Σxᵢ  and  Σxᵢ²",
         formula=r"\bar{X}=\frac{\sum x_i}{n},\quad s^2=\frac{1}{n-1}\left[\sum x_i^2-\frac{(\sum x_i)^2}{n}\right]\ \Rightarrow\ \text{then Z (n≥30) or T (n<30)}",
         req="σ² unknown.  Z is chosen automatically if n ≥ 30, otherwise T (population approx. normal).",
         fields=[FL("sx", "Σxᵢ", 280), FL("sxx", "Σxᵢ²", 2100), IN("n", "n  (sample size)", 40)] + CONF, fn=law_ci_sums),
    dict(lec=3, name="15 · Required sample size   (from the standard error)",
         formula=r"\text{SE}=z_{\frac{\alpha}{2}}\frac{\sigma}{\sqrt{n}}=e\;\Rightarrow\; n=\left(\frac{z_{\alpha/2}\,\sigma}{e}\right)^2",
         req="Needs σ (or s from a pilot sample), the allowed margin of error e, and the confidence level.",
         fields=SIG("Value of σ (or s)  /  variance", 0.3) + [FL("e", "e  (margin of error)", 0.05)] + CONF, fn=law_sample_size),
    # ------------------------------------------------- Lecture 5
    dict(lec=4, name="16 · CI for the population proportion   P",
         formula=r"\hat{P}-z_{\frac{\alpha}{2}}\sqrt{\frac{\hat{P}(1-\hat{P})}{n}}<P<\hat{P}+z_{\frac{\alpha}{2}}\sqrt{\frac{\hat{P}(1-\hat{P})}{n}}",
         req="Large sample:  nP̂ ≥ 5  and  n(1−P̂) ≥ 5.   Give either P̂ or the count x.",
         fields=[CH("mode", "Input", [("P̂  (sample proportion)", "ph"), ("x  (number with the characteristic)", "x")], "ph"),
                 FL("ph", "P̂", 0.24, show=("mode", {"ph"})), IN("x", "x", 340, show=("mode", {"x"})),
                 IN("n", "n  (sample size)", 100)] + CONF, fn=law_ci_prop),
    dict(lec=4, name="17 · CI for the population variance   σ²",
         formula=r"\frac{(n-1)s^2}{\chi^2_{(\frac{\alpha}{2},\,n-1)}}<\sigma^2<\frac{(n-1)s^2}{\chi^2_{(1-\frac{\alpha}{2},\,n-1)}}",
         req="Population must be NORMAL.  Give s², or s, or the raw data.",
         fields=[CH("mode", "Input", [("s²  (sample variance)", "s2"), ("s  (sample std deviation)", "s"), ("Raw data", "raw")], "s2"),
                 FL("sv", "Value of s² or s", 5.76, show=("mode", {"s2", "s"})),
                 IN("n", "n  (sample size)", 30, show=("mode", {"s2", "s"})),
                 TX("data", "Raw data  (comma / space / new-line separated)", "46.4, 46.1, 45.8, 47, 46.1, 45.9, 45.8, 46.9, 45.2, 46",
                    show=("mode", {"raw"}))] + CONF, fn=law_ci_var),
]

for _i, _law in enumerate(LAWS, 1):          # keep numbering continuous
    _law["name"] = re.sub(r"^\d+ · ", f"{_i} · ", _law["name"])

LECTURES = ["Lecture 1  ·  Normal Distribution",
            "Lecture 2  ·  T, χ², F distributions & Sampling of X̄",
            "Lecture 3  ·  Sampling of s² and P̂",
            "Lecture 4  ·  Confidence Interval for the Mean",
            "Lecture 5  ·  Confidence Interval for Proportion & Variance"]



# =====================================================================
#                                  UI  (Streamlit)
# =====================================================================
def _show():
    """matplotlib's plt.show() -> draw inside the page"""
    st.pyplot(plt.gcf(), clear_figure=True)
    plt.close("all")


plt.show = _show


def box(color, bg, text):
    st.markdown(
        f"<div style='background:{bg};border-left:5px solid {color};padding:10px 14px;margin:6px 0;"
        f"color:#222;font-size:15px;border-radius:4px'>{text}</div>", unsafe_allow_html=True)


def input_widget(f, key):
    k = f["kind"]
    if k == "float":
        return st.number_input(f["label"], value=float(f["default"]), format="%g", key=key)
    if k == "int":
        return st.number_input(f["label"], value=int(f["default"]), step=1, key=key)
    if k == "choice":
        opts = [o if isinstance(o, tuple) else (o, o) for o in f["opts"]]
        labels = [o[0] for o in opts]
        default_idx = next(i for i, o in enumerate(opts) if o[1] == f["default"])
        chosen = st.selectbox(f["label"], labels, index=default_idx, key=key)
        return dict(opts)[chosen]
    return st.text_area(f["label"], value=f["default"], height=90, key=key)


# ------------------------------------------------------------ header
st.markdown(f"""
<div style="background:linear-gradient(90deg,#FFE5D0,#FFF3EA);border-radius:12px;padding:16px 22px;margin-bottom:14px;
            border:1px solid #f1c7a8;text-align:center">
  <div style="font-size:30px;font-weight:bold;color:{RED}">Advanced Statistics &amp; Probability</div>
  <div style="font-size:15px;color:{BLUE};margin-top:4px">Interactive formula calculator &nbsp;·&nbsp; Lectures 1 – 5 &nbsp;·&nbsp; {len(LAWS)} formulas</div>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"<div style='font-size:20px;font-weight:bold;color:{RED}'>Choose formula</div>", unsafe_allow_html=True)
    lec = st.radio("Lecture", range(len(LECTURES)), format_func=lambda i: LECTURES[i], label_visibility="collapsed")
    st.divider()
    here = [i for i, law in enumerate(LAWS) if law["lec"] == lec]
    idx = st.radio("Formula", here, format_func=lambda i: LAWS[i]["name"], label_visibility="collapsed")
    st.divider()
    TABLE = st.checkbox("Match lecture tables (round z to 2 decimals)", value=True,
                        help="ON: z is rounded to 2 decimals and φ is read like the printed table, "
                             "as in the lecture solutions.  OFF: exact values from scipy.")
    st.divider()
    st.caption("Values are computed exactly with scipy, so they may differ slightly from the printed tables "
               "(e.g. z = 1.96 vs 1.95996). Enter P̂ as a proportion between 0 and 1.")

law = LAWS[idx]

# ------------------------------------------------------------ main
st.markdown(f"<div style='font-size:19px;font-weight:bold;color:{RED}'>{esc(LECTURES[lec])}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size:22px;font-weight:bold;color:{GREEN};margin:4px 0'>{esc(law['name'])}</div>", unsafe_allow_html=True)
st.markdown(f"<div style='font-size:14px;color:{BLUE};margin-bottom:6px'><b>Requirements:</b> {esc(law['req'])}</div>", unsafe_allow_html=True)
st.latex(law["formula"])

vals = {}
with st.container(border=True):
    cols = st.columns(2)
    n_shown = 0
    for f in law["fields"]:
        if f.get("show"):
            ctrl, allowed = f["show"]
            if vals.get(ctrl) not in allowed:
                vals[f["k"]] = f["default"]   # hidden field: keep its default
                continue
        with cols[n_shown % 2]:
            vals[f["k"]] = input_widget(f, f"{idx}_{f['k']}")
        n_shown += 1
    go = st.button("Calculate", type="primary", use_container_width=True, key=f"go_{idx}")

# results stay on screen (and update live) once Calculate was pressed for this formula
if go:
    st.session_state[f"done_{idx}"] = True
if st.session_state.get(f"done_{idx}"):
    try:
        steps, res, warns, plot = law["fn"](vals)
    except ValueError as e:
        box(RED, "#fdecea", f"⚠️ <b>Input error:</b> {esc(str(e))}")
    except Exception as e:  # noqa
        box(RED, "#fdecea", f"⚠️ <b>Unexpected error:</b> {type(e).__name__}: {esc(str(e))}")
    else:
        st.markdown(f"<h3 style='color:{GREEN};margin:14px 0 2px'>Solution</h3>", unsafe_allow_html=True)
        for s in steps:
            st.latex(s)
        for w in warns:
            box("#ED7D31", "#fff4e5", f"⚠️ {esc(w)}")
        st.markdown(
            f"<div style='background:#eaf3fb;border:2px solid {BLUE};border-radius:10px;padding:14px 18px;margin:12px 0;"
            f"text-align:center;color:{BLUE};font-size:23px;font-weight:bold'>{esc(res)}</div>", unsafe_allow_html=True)
        try:
            plot()
        except Exception as e:  # noqa
            box(RED, "#fdecea", f"⚠️ Could not draw the figure: {esc(str(e))}")
