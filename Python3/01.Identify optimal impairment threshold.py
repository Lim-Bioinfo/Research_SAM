import math
import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, binom, betabinom


## Function lists
def is_region_like(s):
    """
    Return True if a ClinVar 'GeneSymbol' entry looks like a region / locus
    (e.g. '17p13.1:17p13.3' or 'subset of ...') rather than a single gene.
    These should be excluded when building gene sets.
    """
    if pd.isna(s):
        return True
    text = str(s)
    if "subset of" in text.lower():
        return True
    return bool(re.search(r"[A-Za-z0-9-]+:[A-Za-z0-9-]+$", text))


def split_symbols(s):
    """
    Split a ClinVar GeneSymbol field that may contain multiple genes
    into a list of clean, uppercase gene symbols.
    """
    if pd.isna(s):
        return []
    parts = re.split(r"[;,|/]\s*|\s+and\s+", str(s))
    symbols = []
    for part in parts:
        sym = part.strip().upper()
        if re.fullmatch(r"[A-Z0-9-]+", sym):
            symbols.append(sym)
    return symbols

# EV–controlled minimum patient count (min_pat / k*)
def choose_min_pat(gvb_df, T, genes=None, method="betabinom", target_EV=1.0):
    """
    For a given GVB threshold T, choose the minimum patient count k* such that
    the expected number of *null* genes passing (EV) is <= target_EV.

    Model:
      - Each gene's count of impaired patients X_g ~ Binomial(n, mu) or
        Beta–Binomial(n, alpha, beta), depending on over-dispersion rho.

    Parameters
    ----------
    gvb_df : DataFrame
        GVB matrix, genes x patients; lower values = more impaired.
    T : float
        GVB threshold.
    genes : iterable of str or None
        Subset of genes to use for parameter estimation; default all rows.
    method : {"binom", "betabinom"}
        Distribution used for EV computation; over-dispersion estimated
        from data if "betabinom".
    target_EV : float
        Upper bound on expected number of null genes passing the (T, k) filter.

    Returns
    -------
    dict with keys:
      - k_star, n, G, mu, var, rho, alpha, beta, EV_at_k, model, EV_curve
    """
    if genes is None:
        genes = gvb_df.index

    M = gvb_df.loc[list(genes)]
    n = M.shape[1]     # number of patients
    G = M.shape[0]     # number of genes

    # counts of impaired patients per gene at threshold T
    x = (M.lt(T)).sum(axis=1).to_numpy()

    # empirical mean/variance of per-gene frequency
    mu = x.mean() / float(n)
    var = x.var(ddof=1)

    # clamp mu into (0,1) for numerical stability
    mu = min(max(mu, 1e-9), 1 - 1e-9)
    denom = n * mu * (1.0 - mu)

    if denom <= 0:
        # completely degenerate: fall back to a simple rule-of-thumb k*
        k_star = int(math.ceil(0.1 * n))
        ks = np.arange(1, n + 1)
        return dict(
            k_star=k_star,
            n=n,
            G=G,
            mu=mu,
            var=var,
            rho=np.nan,
            alpha=np.nan,
            beta=np.nan,
            EV_at_k=np.nan,
            model="degenerate",
            EV_curve=pd.DataFrame({"k": ks, "EV": np.nan}),
        )

    # over-dispersion estimate rho (method-of-moments for beta-binomial)
    rho = (var / denom - 1.0) / (n - 1.0)
    rho = max(0.0, float(rho))
    use_bb = (method == "betabinom") and (rho > 1e-9)

    if use_bb:
        inv = (1.0 / rho) - 1.0
        alpha = max(mu * inv, 1e-8)
        beta = max((1.0 - mu) * inv, 1e-8)
        dist = betabinom
        model = "beta-binomial"
        tail = dist.sf
    else:
        alpha = beta = np.nan
        dist = binom
        model = "binomial"
        tail = dist.sf

    ks = np.arange(1, n + 1, dtype=int)

    if use_bb:
        # survival function P[X >= k] = sf(k-1)
        tail_probs = tail(ks - 1, n, alpha, beta)
    else:
        tail_probs = tail(ks - 1, n, mu)

    EV = G * tail_probs  # expected number of null genes passing threshold
    idx = np.where(EV <= target_EV)[0]

    if len(idx) > 0:
        k_star = int(ks[idx[0]])
        EV_at_k = float(EV[idx[0]])
    else:
        # never reaches target_EV; default to 10% of patients as a coarse rule
        k_star = int(math.ceil(0.1 * n))
        if np.isfinite(EV).any():
            EV_at_k = float(EV[np.searchsorted(ks, k_star)])
        else:
            EV_at_k = np.nan

    return dict(
        k_star=k_star,
        n=n,
        G=G,
        mu=mu,
        var=var,
        rho=(rho if rho > 0 else 0.0),
        alpha=alpha,
        beta=beta,
        EV_at_k=EV_at_k,
        model=model,
        EV_curve=pd.DataFrame({"k": ks, "EV": EV}),
    )

# Fisher enrichment with Haldane–Anscombe OR
def fisher_enrich(Aset, Bset, U, alternative="greater"):
    """
    Perform Fisher's exact test for enrichment of A in B within universe U,
    and compute both the raw Odds Ratio and the Haldane–Anscombe corrected OR.

        U : universe gene set
        A : "impaired" genes (e.g. I(T, k*))
        B : reference set (e.g. LOEUF-constrained genes)

    Returns
    -------
    dict with:
      OR_raw, OR_HA, p, ci95_lo, ci95_hi,
      n11, n12, n21, n22, nU, nA, nB
    """
    U = set(U)
    A = set(Aset) & U
    B = set(Bset) & U

    n11 = len(A & B)            # in A and in B
    n12 = len(B - A)            # not in A, in B
    n21 = len(A - B)            # in A, not in B
    n22 = len(U - (A | B))      # in neither

    OR_raw, p = fisher_exact([[n11, n12], [n21, n22]], alternative=alternative)
    OR_ha = ((n11 + 0.5) * (n22 + 0.5)) / ((n12 + 0.5) * (n21 + 0.5))

    ci_lo = ci_hi = None
    try:
        from statsmodels.stats.api import Table2x2
        ci_lo, ci_hi = Table2x2([[n11, n12], [n21, n22]]).oddsratio_confint(method="exact")
    except Exception:
        pass

    return dict(
        OR_raw=OR_raw,
        OR_HA=OR_ha,
        p=p,
        ci95_lo=ci_lo,
        ci95_hi=ci_hi,
        n11=n11,
        n12=n12,
        n21=n21,
        n22=n22,
        nU=len(U),
        nA=len(A),
        nB=len(B),
    )

# ClinVar gene-set builders: Get genes both related to pathogenic and oncogenic(melanoma)
def build_clinvar_sets(variant_summary_path, restrict_small_variants=True, keep_likely_oncogenic=True, melanoma_regex=r"melanoma"):
    """
    Build two gene sets from ClinVar variant_summary.txt:
    (1) High-confidence germline pathogenic(P)/likely pathogenic(LP) genes.
    (2) Melanoma-related somatic oncogenic (± likely oncogenic) genes.
    """
    cv = pd.read_csv(variant_summary_path, sep="\t", compression="infer", low_memory=False)

    # Germline P/LP
    germ = cv.copy()
    sig = germ["ClinicalSignificance"].astype(str).str.lower()

    keep_plp = sig.str.contains(r"\bpathogenic\b")
    drop_terms = (
        "benign|uncertain|risk allele|risk factor|drug response|protective|"
        "conflicting|association|affects|no classifications|not provided|other"
)
    germ = germ[keep_plp & ~sig.str.contains(drop_terms)].copy()

    if "OriginSimple" in germ.columns:
        germ = germ[germ["OriginSimple"].str.contains(r"\bgermline\b", case=False, na=False)]

    if "ReviewStatus" in germ.columns:
        good_status = {"criteria provided, multiple submitters, no conflicts", "reviewed by expert panel", "practice guideline"}
        germ = germ[germ["ReviewStatus"].str.lower().isin(good_status)]

    if restrict_small_variants:
        type_col = next((c for c in germ.columns if c.lower() in {"type", "varianttype"}), None)
        if type_col is not None:
            allowed = {"single nucleotide variant", "snv", "insertion", "deletion", "indel"}
            germ = germ[germ[type_col].astype(str).str.lower().isin(allowed)]

    germ = germ[germ["GeneSymbol"].notna()]
    germ = germ[~germ["GeneSymbol"].apply(is_region_like)]
    germ["symbol_list"] = germ["GeneSymbol"].apply(split_symbols)
    germ = germ.explode("symbol_list").dropna(subset=["symbol_list"])
    germline_plp_genes = set(germ["symbol_list"])

    # Melanoma somatic oncogenic
    som = cv.copy()
    if "PhenotypeList" in som.columns and melanoma_regex:
        som = som[som["PhenotypeList"].astype(str).str.contains(melanoma_regex, case=False, na=False)]

    if "Oncogenicity" in som.columns:
        ok = ["Oncogenic"]
        if keep_likely_oncogenic:
            ok.append("Likely oncogenic")
        som = som[som["Oncogenicity"].isin(ok)]

    if "OriginSimple" in som.columns:
        som = som[som["OriginSimple"].str.contains("somatic", case=False, na=False)]

    som = som[som["GeneSymbol"].notna()]
    som = som[~som["GeneSymbol"].apply(is_region_like)]
    som["symbol_list"] = som["GeneSymbol"].apply(split_symbols)
    som = som.explode("symbol_list").dropna(subset=["symbol_list"])
    melanoma_somatic_oncogenic_genes = set(som["symbol_list"])

    return germline_plp_genes, melanoma_somatic_oncogenic_genes

# LOEUF gene-set builders: Get LOEUF gene sets
def build_loeuf_sets(constraint_path, protein_coding_only=True):
    """
    Build LOEUF-constrained and LoF-tolerant gene sets from gnomAD constraint metrics.
    LOEUF version v2: Uses column 'oe_lof_upper', constrained < 0.35 (historical cutpoint).
    """
    df = pd.read_csv(constraint_path, sep="\t", low_memory=False, compression="gzip")

    # Prioritize mane/canonical transcripts when collapsing to gene-level
    sort_cols = []
    for col in ["mane_select", "canonical"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
            sort_cols.append(col)
          
    if sort_cols:
        df = (df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols)).drop_duplicates(subset=["gene"], keep="first"))

    # optionally restrict to protein-coding genes
    if protein_coding_only:
        for c in ["transcript_type", "gene_type"]:
            if c in df.columns:
                df = df[df[c].astype(str).str.lower() == "protein_coding"]

    ser = df.set_index("gene")["oe_lof_upper"].dropna()
    constrained = set(ser.index[ser < 0.35])
    tolerant = set(ser.index[ser >= 1.0])

    return constrained, tolerant

# Stability based on split-half Jaccard similarity
def jaccard_stability(gvb_df, T, k_star, B=50, seed=1):
    rng = np.random.default_rng(seed)
    genes = gvb_df.index
    n = gvb_df.shape[1]
    jaccards = []

    for _ in range(B):
        idx = rng.permutation(n)
        left_idx = idx[: n // 2]
        right_idx = idx[n // 2 :]

        k_left = max(1, int(math.floor(k_star * (len(left_idx) / float(n)))))
        k_right = max(1, int(math.ceil(k_star * (len(right_idx) / float(n)))))

        left_flag = gvb_df.iloc[:, left_idx].lt(T)
        right_flag = gvb_df.iloc[:, right_idx].lt(T)

        A1 = set(genes[left_flag.sum(axis=1) >= k_left])
        A2 = set(genes[right_flag.sum(axis=1) >= k_right])

        union_size = len(A1 | A2)
        if union_size == 0:
            jaccards.append(0.0)
        else:
            jaccards.append(len(A1 & A2) / float(union_size))

    return float(np.median(jaccards)) # Median Jaccard index over B random half-splits

# Threshold scan across T grid
# "EV" or "fixed"
def scan_T_grid(gvb_df, T_list, U, germline_PLP, loeuf_constrained, tolerant=None, oncogenic=None, target_EV=1.0, stability_B=50, stability_seed=452456, min_pat_mode="EV", min_pat_fixed=None, min_pat_ref_T=None):
    """
    Scan a grid of GVB thresholds T and evaluate:
    (1) EV–controlled min_pat k* (or fixed k)
    (2) Size of impaired set A(T, k)
    (3) Jaccard stability
    (4) Enrichment vs Anchor gene sets (from ClinVar, LOEUF)
    """
    # Prepare fixed k if requested
    k_fixed = None
    if min_pat_mode == "fixed":
        if min_pat_fixed is not None:
            k_fixed = int(min_pat_fixed)
        else:
            T0 = T_list[0] if min_pat_ref_T is None else float(min_pat_ref_T)
            ch0 = choose_min_pat(gvb_df, T0, method="betabinom", target_EV=target_EV)
            k_fixed = int(ch0["k_star"])

    U = set(U)
    rows = []

    for T in T_list:
        # choose min_pat for this T
        if min_pat_mode == "EV":
            ch = choose_min_pat(gvb_df, T, method="betabinom", target_EV=target_EV)
            k = ch["k_star"]
        else:
            ch = choose_min_pat(gvb_df, T, method="betabinom", target_EV=target_EV)
            k = k_fixed

        EV_at_k = ch["EV_at_k"]
        mu = ch["mu"]
        rho = ch["rho"]
        alpha = ch["alpha"]
        beta = ch["beta"]
        model = ch["model"]

        # Candidate impaired genes at (T, k)
        low_flag = gvb_df.lt(T)
        impaired_genes = set(low_flag.index[low_flag.sum(axis=1) >= k]) & U
        A_size = len(impaired_genes)

        # Split-half stability
        stability = jaccard_stability(gvb_df, T, k, B=stability_B, seed=stability_seed)

        # Do enrichment test against anchor sets
        anchor_specs = [("ClinVar_germline_PLP", germline_PLP, "greater"),("LOEUF_constrained", loeuf_constrained, "greater"),]
        if tolerant is not None:
            anchor_specs.append(("LoF_tolerant", tolerant, "less"))
        if oncogenic is not None:
            anchor_specs.append(("Somatic_oncogenic", oncogenic, "two-sided"))

        for set_name, ref_set, alt in anchor_specs:
            stats = fisher_enrich(impaired_genes, ref_set, U, alternative=alt)
            stats.update(dict(T=T, min_pat=k, A_size=A_size, stability=stability, EV_at_k=EV_at_k, mu=mu, rho=rho, alpha=alpha, beta=beta, model=model, set=set_name))
            rows.append(stats)

    out = pd.DataFrame(rows)

    # Do multiple-testing corrections: global, within T, and by gene set
    try:
        from statsmodels.stats.multitest import multipletests

        out["q_global"] = multipletests(out["p"].values, method="fdr_bh")[1]

        out["q_withinT"] = np.nan
      
        for T, idx in out.groupby("T").groups.items():
            out.loc[idx, "q_withinT"] = multipletests(out.loc[idx, "p"].values, method="fdr_bh")[1]

        out["q_by_set"] = np.nan
        for s, idx in out.groupby("set").groups.items():
            out.loc[idx, "q_by_set"] = multipletests(out.loc[idx, "p"].values, method="fdr_bh")[1]
          
    except Exception:
        out["q_global"] = np.nan
        out["q_withinT"] = np.nan
        out["q_by_set"] = np.nan

    return out.sort_values(["set", "T"]).reset_index(drop=True) # DataFrame with one row per (T, anchor_set) combination

# Select optimal threshold T of GVB from a result of scan_T_grid()
def pick_T(res_df, core_sets=("ClinVar_germline_PLP", "LOEUF_constrained"), stab_cut=0.6, minA=50, maxA_frac=0.5, use_pvalues=False, q_col="q_by_set", q_cut=0.05):
    """
    Always require:
    (1) stability >= stab_cut
    (2) minA <= A_size <= maxA_frac * |U|
    
    If use_pvalues is True, additionally require for each core set:
    (1) at least one row with q_col <= q_cut at that T.

    If multiple T satisfy criteria, choose the center of the longest contiguous block (assuming step size ~0.1). 
    If no T qualifies, fall back to the T with the highest stability (then largest A_size).
    """
    Ts = sorted(res_df["T"].unique())
    U_size = int(res_df["nU"].iloc[0])

    ok_Ts = []
    for T in Ts:
        sub = res_df[res_df["T"] == T]
        A_size = int(sub["A_size"].iloc[0])
        stab = float(sub["stability"].iloc[0])

        size_ok = (A_size >= minA) and (A_size <= int(maxA_frac * U_size))
        if (not size_ok) or (stab < stab_cut):
            continue

        if use_pvalues:
            good = True
            for s in core_sets:
                ss = sub[sub["set"] == s]
                if ss.empty or not (ss[q_col] <= q_cut).any():
                    good = False
                    break
            if not good:
                continue

        ok_Ts.append(T)

    if ok_Ts:
        # Group into contiguous blocks, step size of 0.1
        blocks = []
        cur_block = [ok_Ts[0]]
        for t in ok_Ts[1:]:
            if abs(t - cur_block[-1] - 0.1) < 1e-9:
                cur_block.append(t)
            else:
                blocks.append(cur_block)
                cur_block = [t]
        blocks.append(cur_block)

        best_block = max(blocks, key=len)
        T_star = best_block[len(best_block) // 2]
        why = "plateau center using " + ("stability+size" if not use_pvalues else "stability+size+%s≤%g" % (q_col, q_cut))
        return float(T_star), why

    # Fallback: Maximize stability then A_size
    tmp = (res_df.groupby("T", as_index=False).agg(stab=("stability", "first"), A_size=("A_size", "first")).sort_values(["stab", "A_size"], ascending=[False, False]))
    return float(tmp.iloc[0]["T"]), "fallback: max stability then size"


## Identify ideal GVB threshold to quantify gene impairment
if __name__ == "__main__":
    # gvb_df : genes x patients GVB matrix (lower means more impaired)
    # T_list : list of thresholds, e.g. [0.1, 0.2, ..., 0.9]
    # U : universe of genes, usually from gvb_df.index
  
    gvb_df = pd.read_csv("TCGA_SKCM_GVB.txt", sep = "\t") # Gene symbol for index, column for sample
    gvb_df.set_index('Hugo_Symbol', inplace = True)
    germ_plp, mel_onc = build_clinvar_sets("variant_summary.txt.gz", restrict_small_variants=True, keep_likely_oncogenic=True, melanoma_regex=r"melanoma")
    loeuf_constrained, loeuf_tolerant = build_loeuf_sets("gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz")
    U = set(gvb_df.index)
    T_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    summary_rows = []

    for target_ev_value in [0.5, 1.0]:
        res = scan_T_grid(gvb_df, T_list, U, germline_PLP=germ_plp, loeuf_constrained=loeuf_constrained, tolerant=loeuf_tolerant, oncogenic=mel_onc, target_EV=target_ev_value, stability_B=1000, stability_seed=452456, min_pat_mode="EV")

        for stab_cut_value in [0.6]:
            for minA_frac_value in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
                for maxA_frac_value in [0.3, 0.5]:
                    T_star, rationale = pick_T(res, stab_cut=stab_cut_value, minA=int(len(U) * minA_frac_value), maxA_frac=maxA_frac_value, use_pvalues=False)
                    
                    summary_rows.append(dict(T_star=T_star, rationale=rationale, target_EV=target_ev_value, stab_cut=stab_cut_value, minA_frac=minA_frac_value, minA_abs=int(len(U) * minA_frac_value), maxA_frac=maxA_frac_value, U_size=len(U)))
                    
                    print("Chosen T: {T_star} | rationale: {why} | EV: {ev} | " "minA_frac: {minA_frac} | stab_cut: {stab_cut} | ""maxA_frac: {maxA_frac}".format( T_star=T_star, why=rationale, ev=target_ev_value, minA_frac=minA_frac_value, stab_cut=stab_cut_value, maxA_frac=maxA_frac_value,))

    threshold_scan_summary = pd.DataFrame(summary_rows)
    # threshold_scan_summary.to_csv("Identify_optimal_GVB_threshold_impairment.tsv", sep="\t", index=False)
