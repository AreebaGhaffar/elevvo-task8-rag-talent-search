"""Bias Check: does the retrieved top-k favour one group over the pool?

Only attributes explicitly labelled in the dataset are used:
graduation-year cohort, college, location, and years-of-experience band.
The data has NO gender, ethnicity or age field, and protected attributes
are NEVER inferred from names, photos or anything else — the check
reports skew on the labelled attributes only, and says so.

A group is flagged when its share in the retrieved top-k differs from
its share in the whole pool by more than FLAG_THRESHOLD percentage
points, or when one group makes up >= 80% of the top-k while it is
< 50% of the pool. A flag is a prompt to review the query/ranking, not
proof of discrimination: small pools (220 resumes) and genuinely
skill-linked attributes (e.g. a niche college for a niche skill) can
produce skew for legitimate reasons.
"""
import re

import pandas as pd

FLAG_THRESHOLD = 25.0  # percentage points


def _cohort(grad_year: str) -> str:
    try:  # values arrive as "2017.0" (pandas wrote the year as a float)
        y = int(float(str(grad_year).strip()))
    except (TypeError, ValueError):
        return "unknown"
    if y <= 2005:
        return "2005 or earlier"
    if y <= 2012:
        return "2006-2012"
    return "2013 or later"


_CITY_VARIANTS = {
    "bangalore": "bengaluru",
    "bangalore urban": "bengaluru",
    "bengaluru urban": "bengaluru",
    "bombay": "mumbai",
    "madras": "chennai",
}


def _city(location: str) -> str:
    loc = str(location).strip().lower()
    loc = re.sub(r",.*$", "", loc).strip()  # drop ", Karnataka" etc.
    return _CITY_VARIANTS.get(loc, loc) or "unknown"


def _college_group(college: str, top5: set[str]) -> str:
    """Colleges are nearly unique in this pool (~190 distinct for 194
    labelled resumes), so per-college shares are meaningless; compare the
    combined share of the 5 most common colleges instead."""
    c = str(college).strip()
    if not c:
        return "unknown"
    return "top-5 colleges" if c in top5 else "other colleges"


def _exp_band(years) -> str:
    try:
        y = float(years)
    except (TypeError, ValueError):
        return "unknown"
    if y < 0:
        return "unknown"
    if y < 3:
        return "junior (<3 yrs)"
    if y <= 7:
        return "mid (3-7 yrs)"
    return "senior (>7 yrs)"


def bias_check(retrieved_ids: list[str], pool: pd.DataFrame) -> dict:
    """Compare the retrieved set against the whole pool on labelled attributes."""
    pool = pool.fillna("")
    retrieved = pool[pool["id"].isin(retrieved_ids)]
    n_pool, n_ret = len(pool), len(retrieved)
    top5_colleges = set(
        pool["college"].replace("", pd.NA).dropna().value_counts().head(5).index)
    attrs = {
        "graduation cohort": pool["grad_year"].map(_cohort),
        "college": pool["college"].map(lambda c: _college_group(c, top5_colleges)),
        "location": pool["location"].map(_city),
        "experience band": pool["years_experience"].map(_exp_band),
    }
    ret_attrs = {
        "graduation cohort": retrieved["grad_year"].map(_cohort),
        "college": retrieved["college"].map(
            lambda c: _college_group(c, top5_colleges)),
        "location": retrieved["location"].map(_city),
        "experience band": retrieved["years_experience"].map(_exp_band),
    }
    report = {"n_pool": n_pool, "n_retrieved": n_ret, "attributes": {}, "flags": []}
    for name in attrs:
        pool_share = attrs[name].value_counts(normalize=True) * 100
        ret_share = ret_attrs[name].value_counts(normalize=True) * 100
        groups = {}
        for group in sorted(set(pool_share.index) | set(ret_share.index)):
            p, r = float(pool_share.get(group, 0.0)), float(ret_share.get(group, 0.0))
            groups[group] = {
                "pool_pct": round(p, 1), "retrieved_pct": round(r, 1),
                "diff_pp": round(r - p, 1)}
            if abs(r - p) > FLAG_THRESHOLD or (r >= 80 and p < 50):
                report["flags"].append(
                    f"{name}: '{group}' is {r:.0f}% of the retrieved top-{n_ret} "
                    f"but {p:.0f}% of the pool (diff {r - p:+.0f} pp)")
        report["attributes"][name] = groups
    return report


def format_report(report: dict) -> str:
    lines = [f"Bias check — top-{report['n_retrieved']} vs pool of "
             f"{report['n_pool']} resumes"]
    for name, groups in report["attributes"].items():
        top = sorted(groups.items(), key=lambda kv: -kv[1]["retrieved_pct"])[:3]
        lines.append(f"  {name}: " + "; ".join(
            f"{g} pool {v['pool_pct']:.0f}% -> retrieved {v['retrieved_pct']:.0f}%"
            for g, v in top))
    if report["flags"]:
        lines.append("  FLAGS:")
        lines += [f"    ! {f}" for f in report["flags"]]
    else:
        lines.append("  No group is flagged (all differences within "
                     f"{FLAG_THRESHOLD:.0f} pp).")
    lines.append("  Note: only explicitly labelled attributes are checked; "
                 "no protected attribute is inferred from names.")
    return "\n".join(lines)
