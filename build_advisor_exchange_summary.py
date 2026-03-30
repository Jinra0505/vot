import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


CASE_PATHS = {
    "simple_case": {
        "yaml": "simple_case_semantic_fix_v13_constant_background_with_grid.yaml",
        "results": "advisor_exchange_outputs/simple/results.json",
        "report": "advisor_exchange_outputs/simple/report.json",
    },
    "complex_case": {
        "yaml": "complex_case_larger_reframed_v11_network_heterogeneous_refined_with_grid.yaml",
        "results": "advisor_exchange_outputs/complex/results.json",
        "report": "advisor_exchange_outputs/complex/report.json",
    },
}

MODE_TO_SUPERMODE = {
    "EV": "EV",
    "eVTOL_fast": "eVTOL",
    "eVTOL_slow": "eVTOL",
    "EV_to_eVTOL_fast": "EV_to_eVTOL",
    "EV_to_eVTOL_slow": "EV_to_eVTOL",
}

DIAG_MODE_TO_SUMMARY_MODE = {
    "pure_EV": "EV",
    "pure_eVTOL_fast": "eVTOL_fast",
    "pure_eVTOL_slow": "eVTOL_slow",
    "multimodal_EV_to_eVTOL_fast": "EV_to_eVTOL_fast",
    "multimodal_EV_to_eVTOL_slow": "EV_to_eVTOL_slow",
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_times_from_flow_map(flow_map: Dict[str, Any]) -> Iterable[str]:
    for it_map in flow_map.values():
        if isinstance(it_map, dict):
            for grp_map in it_map.values():
                if isinstance(grp_map, dict):
                    for t in grp_map:
                        yield str(t)


def _select_peak_t(report: Dict[str, Any], case_block: Dict[str, Any], flow_map: Dict[str, Any]) -> str:
    diag_peak = report.get("diagnostics", {}).get("peak_t")
    if diag_peak is not None:
        return str(diag_peak)

    times = sorted(set(_iter_times_from_flow_map(flow_map)), key=lambda x: int(x) if str(x).isdigit() else str(x))
    if not times:
        raise RuntimeError("Cannot select peak_t: no times in final flow map")

    rep_od = report.get("summary", {}).get("representative_od") or report.get("config_used", {}).get("representative_od")
    demand_by_od = case_block.get("demand", {}).get("by_od_group_time", {})
    if rep_od and rep_od in demand_by_od:
        best_t = None
        best_val = -1.0
        for t in times:
            total = 0.0
            for grp, t_map in demand_by_od[rep_od].items():
                if isinstance(t_map, dict):
                    total += float(t_map.get(str(t), 0.0))
            if total > best_val:
                best_val = total
                best_t = t
        if best_t is not None:
            return str(best_t)

    return str(times[0])


def _ensure_rep_od(report: Dict[str, Any], rep_od: str) -> None:
    cfg_rep = report.get("config_used", {}).get("representative_od")
    if cfg_rep is not None and str(cfg_rep) != str(rep_od):
        raise RuntimeError(f"representative_od mismatch: summary={rep_od} config={cfg_rep}")


def _safe_get_3(m: Dict[str, Any], a: str, b: str) -> Any:
    if not isinstance(m, dict):
        return None
    if a not in m or not isinstance(m[a], dict):
        return None
    return m[a].get(str(b))


def _build_itinerary_and_peak(
    case_block: Dict[str, Any],
    results: Dict[str, Any],
    report: Dict[str, Any],
    peak_t: str,
    rep_od: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    flow_map = results.get("f", {})
    genc = results.get("generalized_costs", {})
    rawc = results.get("generalized_costs_raw", {})
    util = results.get("utilities", {})

    defs = case_block.get("itinerary_results", {}).get("definitions", {})
    groups = case_block.get("structure", {}).get("groups", [])

    for it_id, it_def in defs.items():
        total = 0.0
        peak_by_group = {g: 0.0 for g in groups}
        it_flow = flow_map.get(it_id, {})
        for g in groups:
            g_map = it_flow.get(g, {}) if isinstance(it_flow, dict) else {}
            for t, val in (g_map or {}).items():
                total += float(val)
            peak_by_group[g] = float((g_map or {}).get(str(peak_t), 0.0))

        it_def["final_flow_total"] = float(total)
        it_def["final_flow_peak_t_by_group"] = peak_by_group

    peak_details = case_block.get("itinerary_results", {}).get("peak_od_details", {})
    peak_details["peak_t"] = str(peak_t)
    peak_details["representative_od"] = rep_od

    demand_by_group = case_block.get("demand", {}).get("by_od_group_time", {}).get(rep_od, {})
    travelers = {g: float((demand_by_group.get(g, {}) or {}).get(str(peak_t), 0.0)) for g in groups}
    peak_details["traveler_counts_by_group"] = travelers

    mode_shares_by_group: Dict[str, Dict[str, float]] = {}
    key_itins = peak_details.get("key_itineraries", {})
    rep_token = str(rep_od).replace("-", "").replace("_", "").lower()
    rep_itinerary_ids = [it_id for it_id in defs if str(it_id).replace("_", "").lower().startswith(rep_token)]

    for g in groups:
        denom = travelers[g]
        mode_totals = {
            "EV": 0.0,
            "eVTOL_fast": 0.0,
            "eVTOL_slow": 0.0,
            "EV_to_eVTOL_fast": 0.0,
            "EV_to_eVTOL_slow": 0.0,
        }
        for it_id, it_def in defs.items():
            if rep_itinerary_ids and it_id not in rep_itinerary_ids:
                continue
            mode_label = it_def.get("mode")
            val = float(((flow_map.get(it_id, {}) or {}).get(g, {}) or {}).get(str(peak_t), 0.0))
            if mode_label in mode_totals:
                mode_totals[mode_label] += val

        if denom > 0.0:
            mode_shares_by_group[g] = {k: float(v / denom) for k, v in mode_totals.items()}
        else:
            mode_shares_by_group[g] = {k: 0.0 for k in mode_totals}

        super_sum = 0.0
        super_agg = {"EV": 0.0, "eVTOL": 0.0, "EV_to_eVTOL": 0.0}
        for mode_name, share in mode_shares_by_group[g].items():
            super_agg[MODE_TO_SUPERMODE[mode_name]] += share
        super_sum = sum(super_agg.values())
        if denom > 0.0 and abs(super_sum - 1.0) > 5.0e-3:
            raise RuntimeError(
                f"Supermode shares do not sum to 1 for group={g} at peak_t={peak_t}: {super_sum}"
            )

    peak_details["mode_shares_by_group"] = mode_shares_by_group

    for it_id, payload in key_itins.items():
        payload["peak_flow_by_group"] = {
            g: float((((flow_map.get(it_id, {}) or {}).get(g, {}) or {}).get(str(peak_t), 0.0)))
            for g in groups
        }
        payload["peak_generalized_cost_by_group"] = {
            g: _safe_get_3(genc.get(it_id, {}), g, str(peak_t)) for g in groups
        }
        payload["peak_raw_cost_by_group"] = {
            g: _safe_get_3(rawc.get(it_id, {}), g, str(peak_t)) for g in groups
        }
        payload["peak_utility_by_group"] = {
            g: _safe_get_3(util.get(it_id, {}), g, str(peak_t)) for g in groups
        }

    return defs, peak_details


def _fill_station_table(case_block: Dict[str, Any], results: Dict[str, Any], report: Dict[str, Any]) -> None:
    by_st = case_block.get("station_results", {}).get("by_station_time", {})

    eff = results.get("effective_price", {})
    eff_comp = results.get("effective_price_components", {})
    station_loads = report.get("diagnostics_detail", {}).get("station_loads", {})
    p_ev_kw = station_loads.get("P_ev_req_kw", {})
    p_vt_kw_grid = station_loads.get("P_vt_req_kw_grid", {})
    p_total = station_loads.get("P_total_req", {})
    mu_local = results.get("shadow_price_power", {})
    surcharge = results.get("surcharge_power", {})
    ev_wait = results.get("w", {})
    vt_waits = report.get("diagnostics_detail", {}).get("vt_departure_waits", {})

    for s, t_map in by_st.items():
        for t, row in t_map.items():
            t = str(t)
            row["effective_price"] = _safe_get_3(eff, s, t)
            row["effective_price_components"] = _safe_get_3(eff_comp, s, t)
            row["P_ev_req_kw"] = _safe_get_3(p_ev_kw, s, t)
            row["P_vt_req_kw_grid"] = _safe_get_3(p_vt_kw_grid, s, t)
            row["total_requested_power"] = _safe_get_3(p_total, s, t)
            row["local_shadow_price_mu_kw"] = _safe_get_3(mu_local, s, t)
            row["local_surcharge"] = _safe_get_3(surcharge, s, t)
            row["ev_wait"] = _safe_get_3(ev_wait, s, t)
            row["vt_fast_wait"] = ((vt_waits.get(s, {}) or {}).get("fast", {}) or {}).get(str(t))
            row["vt_slow_wait"] = (
                ((vt_waits.get(s, {}) or {}).get("slow", {}) or {}).get(str(t))
            )

            grounded_maps = {
                "effective_price": _safe_get_3(eff, s, t),
                "effective_price_components": _safe_get_3(eff_comp, s, t),
                "P_ev_req_kw": _safe_get_3(p_ev_kw, s, t),
                "P_vt_req_kw_grid": _safe_get_3(p_vt_kw_grid, s, t),
                "total_requested_power": _safe_get_3(p_total, s, t),
                "local_shadow_price_mu_kw": _safe_get_3(mu_local, s, t),
                "local_surcharge": _safe_get_3(surcharge, s, t),
                "ev_wait": _safe_get_3(ev_wait, s, t),
                "vt_fast_wait": ((vt_waits.get(s, {}) or {}).get("fast", {}) or {}).get(str(t)),
                "vt_slow_wait": ((vt_waits.get(s, {}) or {}).get("slow", {}) or {}).get(str(t)),
            }
            for fld, grounded in grounded_maps.items():
                if grounded is not None and row.get(fld) is None:
                    raise RuntimeError(
                        f"Missing export for grounded station field {fld} at station={s}, t={t}"
                    )


def _validate_mode_share_with_diagnostics(case_name: str, peak_details: Dict[str, Any], report: Dict[str, Any]) -> None:
    peak_t = str(peak_details.get("peak_t"))
    rep_od = peak_details.get("representative_od")
    by_mode = report.get("diagnostics_detail", {}).get("mode_share_by_group_and_mode", [])

    target = None
    for entry in by_mode:
        if str(entry.get("t")) == peak_t and entry.get("od") == rep_od:
            target = entry
    if target is None:
        raise RuntimeError(f"No diagnostics mode-share entry for case={case_name} rep_od={rep_od} peak_t={peak_t}")

    diag_shares = target.get("shares", {})
    for g, shares in peak_details.get("mode_shares_by_group", {}).items():
        for d_mode, s_mode in DIAG_MODE_TO_SUMMARY_MODE.items():
            expected = float((diag_shares.get(g, {}) or {}).get(d_mode, 0.0))
            got = float((shares or {}).get(s_mode, 0.0))
            if abs(expected - got) > 1.0e-6:
                raise RuntimeError(
                    f"Mode share mismatch case={case_name}, group={g}, mode={s_mode}, got={got}, expected={expected}"
                )


def _validate_itinerary_totals(case_name: str, defs: Dict[str, Any], flow_map: Dict[str, Any]) -> None:
    for it_id, it_def in defs.items():
        expected = 0.0
        it_flow = flow_map.get(it_id, {})
        for g_map in (it_flow or {}).values():
            for v in (g_map or {}).values():
                expected += float(v)
        got = float(it_def.get("final_flow_total", 0.0))
        if abs(got - expected) > 1.0e-8:
            raise RuntimeError(
                f"final_flow_total mismatch case={case_name}, itinerary={it_id}, got={got}, expected={expected}"
            )


def refresh_summary(summary_path: Path) -> None:
    summary = _read_json(summary_path)

    for case_name, paths in CASE_PATHS.items():
        results = _read_json(Path(paths["results"]))
        report = _read_json(Path(paths["report"]))
        case_block = summary.get(case_name, {})

        flow_map = results.get("f", {})
        peak_t = _select_peak_t(report, case_block, flow_map)
        rep_od = report.get("summary", {}).get("representative_od") or report.get("config_used", {}).get("representative_od")
        if rep_od is None:
            raise RuntimeError(f"No representative_od in report for {case_name}")

        _ensure_rep_od(report, rep_od)

        defs, peak_details = _build_itinerary_and_peak(case_block, results, report, peak_t, rep_od)
        _fill_station_table(case_block, results, report)

        if peak_details.get("peak_t") is None:
            raise RuntimeError(f"peak_t is null in {case_name} despite non-empty times")

        _validate_mode_share_with_diagnostics(case_name, peak_details, report)
        _validate_itinerary_totals(case_name, defs, flow_map)

        case_block["itinerary_results"]["definitions"] = defs
        case_block["itinerary_results"]["peak_od_details"] = peak_details

        case_block.setdefault("run_status", {})["ran_successfully"] = True
        case_block["run_status"]["stop_reason"] = report.get("summary", {}).get("convergence", {}).get("stop_reason")
        case_block["run_status"]["converged_loose"] = report.get("summary", {}).get("convergence", {}).get("converged_loose")
        case_block["run_status"]["converged_strictly"] = report.get("summary", {}).get("convergence", {}).get("converged_strictly")

        summary[case_name] = case_block

    summary.setdefault("meta", {})["generated_at"] = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    summary["meta"]["notes"] = (
        "Summary refreshed from final case results/report exports with peak/itinerary/station reporting guards enabled."
    )

    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="advisor_exchange_cases_latest.json")
    args = parser.parse_args()
    refresh_summary(Path(args.summary))


if __name__ == "__main__":
    main()
