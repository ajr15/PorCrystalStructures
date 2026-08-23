# Compare full geometry optimization at triplet/quintet spin states (data/benchmark/orca)
# against the original electronic structure calculations (data/electron_structure), which
# only has a full optimization for the singlet. Compares both final energies and geometries.
import argparse
import os
import re
import tempfile
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from openbabel import openbabel as ob

from torinax.io.OrcaOut import OrcaOut

from src.parsers.molecular_geometry import (
	compute_bond_length,
	compute_bond_angle,
	compute_dihedral,
	get_neighbors_dict,
	get_position,
	read_xyz_file,
)


HARTREE_TO_KCAL_MOL = 627.5094740631
DIR_PATTERN = re.compile(r"^(?P<sid>.+)_0_S(?P<spin>\d+)_out$")
SKIP_DIRS = {"inputs", "reinputs"}
SPIN_COLORS = {3: "tab:orange", 5: "tab:green"}


def find_out_file(dir_path: str) -> Optional[str]:
	for fname in os.listdir(dir_path):
		if fname.endswith(".out"):
			return os.path.join(dir_path, fname)
	return None


def scan_out_dirs(root: str) -> Dict[Tuple[str, int], str]:
	"""Scan a directory of ORCA output dirs (name pattern {sid}_0_S{spin}_out), return {(structure_id, spin): out_file_path}."""
	result = {}
	for name in sorted(os.listdir(root)):
		full = os.path.join(root, name)
		if not os.path.isdir(full) or name in SKIP_DIRS:
			continue
		match = DIR_PATTERN.match(name)
		if not match:
			continue
		out_file = find_out_file(full)
		if out_file is None:
			continue
		result[(match.group("sid"), int(match.group("spin")))] = out_file
	return result


def lookup_out_dirs(root: str, targets: set) -> Dict[Tuple[str, int], str]:
	"""Directly look up {(structure_id, spin): out_file_path} for the given (structure_id, spin) targets,
	without listing the whole (potentially large) directory."""
	result = {}
	for sid, spin in sorted(targets):
		dir_name = f"{sid}_0_S{spin}_out"
		full = os.path.join(root, dir_name)
		if not os.path.isdir(full):
			continue
		out_file = find_out_file(full)
		if out_file is None:
			continue
		result[(sid, spin)] = out_file
	return result


def lookup_dft_singlet_outs(root: str, structure_ids: set) -> Dict[str, str]:
	"""Directly look up {structure_id: out_file_path} in the DFT dir, whose dirs (all singlet) follow the
	pattern {structure_id}_0_out, i.e. the benchmark's own-basis-set singlet reference."""
	result = {}
	for sid in sorted(structure_ids):
		full = os.path.join(root, f"{sid}_0_out")
		if not os.path.isdir(full):
			continue
		out_file = find_out_file(full)
		if out_file is None:
			continue
		result[sid] = out_file
	return result


def build_energy_table(
	benchmark_outs: Dict[Tuple[str, int], str],
	es_outs: Dict[Tuple[str, int], str],
	dft_singlet_outs: Dict[str, str],
) -> pd.DataFrame:
	"""Benchmark (triplet/quintet) and electron_structure calculations use different basis sets, so their raw
	energies are not directly comparable. Instead, each source's spin energies are expressed relative to its own
	basis-set-consistent singlet: the benchmark singlet comes from data/dft ({structure_id}_0_out), while the
	electron_structure singlet comes from its own spin==1 entry."""
	structure_ids = sorted({sid for sid, _ in benchmark_outs} | {sid for sid, _ in es_outs} | set(dft_singlet_outs))
	rows = []
	for sid in structure_ids:
		for spin in (1, 3, 5):
			bench_path = dft_singlet_outs.get(sid) if spin == 1 else benchmark_outs.get((sid, spin))
			es_path = es_outs.get((sid, spin))
			if bench_path is None and es_path is None:
				continue
			bench_data = OrcaOut(bench_path).read_scalar_data() if bench_path else None
			es_data = OrcaOut(es_path).read_scalar_data() if es_path else None
			bench_e = bench_data["final_energy"] if bench_data else np.nan
			es_e = es_data["final_energy"] if es_data else np.nan
			rows.append({
				"structure_id": sid,
				"spin": spin,
				"benchmark_energy_hartree": bench_e,
				"benchmark_finished_normally": bench_data["finished_normally"] if bench_data else None,
				"electron_structure_energy_hartree": es_e,
				"electron_structure_finished_normally": es_data["finished_normally"] if es_data else None,
			})
	energy_df = pd.DataFrame(rows)

	# relative energies vs each source's own (basis-set-consistent) singlet
	energy_df["benchmark_rel_singlet_kcal_mol"] = np.nan
	energy_df["electron_structure_rel_singlet_kcal_mol"] = np.nan
	for sid, group in energy_df.groupby("structure_id"):
		idx = group.index
		singlet_row = group[group["spin"] == 1]
		bench_singlet_e = singlet_row.iloc[0]["benchmark_energy_hartree"] if not singlet_row.empty else np.nan
		es_singlet_e = singlet_row.iloc[0]["electron_structure_energy_hartree"] if not singlet_row.empty else np.nan
		if not pd.isna(bench_singlet_e):
			energy_df.loc[idx, "benchmark_rel_singlet_kcal_mol"] = (
				energy_df.loc[idx, "benchmark_energy_hartree"] - bench_singlet_e
			) * HARTREE_TO_KCAL_MOL
		if not pd.isna(es_singlet_e):
			energy_df.loc[idx, "electron_structure_rel_singlet_kcal_mol"] = (
				energy_df.loc[idx, "electron_structure_energy_hartree"] - es_singlet_e
			) * HARTREE_TO_KCAL_MOL

	# spin-gap comparison: both sides are now relative energies, so this is meaningful despite the basis set mismatch
	energy_df["rel_singlet_diff_kcal_mol"] = (
		energy_df["benchmark_rel_singlet_kcal_mol"] - energy_df["electron_structure_rel_singlet_kcal_mol"]
	)

	return energy_df.sort_values(["structure_id", "spin"]).reset_index(drop=True)


def load_obmol_from_out(out_path: str) -> ob.OBMol:
	"""Read the final (optimized) geometry from an ORCA out file and perceive its bonds via OpenBabel."""
	molecule = OrcaOut(out_path).read_specie()
	fd, tmp_path = tempfile.mkstemp(suffix=".xyz")
	os.close(fd)
	try:
		molecule.save_to_file(tmp_path)
		return read_xyz_file(tmp_path)
	finally:
		os.remove(tmp_path)


def bond_lengths_dict(obmol: ob.OBMol) -> Dict[str, float]:
	positions = {atom.GetIdx(): get_position(atom) for atom in ob.OBMolAtomIter(obmol)}
	result = {}
	for bond in ob.OBMolBondIter(obmol):
		i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
		label = "bond({},{})".format(*sorted((i, j)))
		result[label] = compute_bond_length(positions[i], positions[j])
	return result


def angles_dict(obmol: ob.OBMol) -> Dict[str, float]:
	positions = {atom.GetIdx(): get_position(atom) for atom in ob.OBMolAtomIter(obmol)}
	neighbors = get_neighbors_dict(obmol)
	result = {}
	for j in neighbors:
		for i in neighbors[j]:
			for k in neighbors[j]:
				if i < k:
					result[f"angle({i},{j},{k})"] = compute_bond_angle(positions[i], positions[j], positions[k])
	return result


def dihedrals_dict(obmol: ob.OBMol) -> Dict[str, float]:
	positions = {atom.GetIdx(): get_position(atom) for atom in ob.OBMolAtomIter(obmol)}
	neighbors = get_neighbors_dict(obmol)
	result = {}
	for j in neighbors:
		for k in neighbors[j]:
			for i in neighbors[j] - {k}:
				for l in neighbors[k] - {j}:
					result[f"dihedral({i},{j},{k},{l})"] = compute_dihedral(
						positions[i], positions[j], positions[k], positions[l]
					)
	return result


GEOMETRY_FEATURES = (
	("bond_length", bond_lengths_dict),
	("angle", angles_dict),
	("dihedral", dihedrals_dict),
)


def summarize_pair(bench_vals: np.ndarray, es_vals: np.ndarray) -> Dict[str, float]:
	if len(bench_vals) < 2:
		return {"n": len(bench_vals), "pearson_r": np.nan, "r2": np.nan, "mae": np.nan, "rmse": np.nan}
	r, _ = stats.pearsonr(es_vals, bench_vals)
	diffs = bench_vals - es_vals
	return {
		"n": len(bench_vals),
		"pearson_r": r,
		"r2": r ** 2,
		"mae": float(np.mean(np.abs(diffs))),
		"rmse": float(np.sqrt(np.mean(diffs ** 2))),
	}


def compare_geometries(benchmark_outs: Dict[Tuple[str, int], str], es_outs: Dict[Tuple[str, int], str], outdir: str):
	"""For every (structure, spin) present in both sources, compare bond/angle/dihedral geometries.
	Returns a summary DataFrame and writes one correlation plot per structure."""
	pairs = sorted(set(benchmark_outs) & set(es_outs))
	summary_rows = []
	feature_data = {}  # structure_id -> feature_name -> spin -> DataFrame(label, benchmark, electron_structure)

	for sid, spin in pairs:
		bench_obmol = load_obmol_from_out(benchmark_outs[(sid, spin)])
		es_obmol = load_obmol_from_out(es_outs[(sid, spin)])
		for feature_name, extractor in GEOMETRY_FEATURES:
			bench_d = extractor(bench_obmol)
			es_d = extractor(es_obmol)
			common_labels = sorted(set(bench_d) & set(es_d))
			bench_vals = np.array([bench_d[l] for l in common_labels])
			es_vals = np.array([es_d[l] for l in common_labels])

			stats_dict = summarize_pair(bench_vals, es_vals)
			summary_rows.append({"structure_id": sid, "spin": spin, "feature": feature_name, **stats_dict})

			feature_data.setdefault(sid, {}).setdefault(feature_name, {})[spin] = (bench_vals, es_vals)

	for sid, features in feature_data.items():
		fig, axes = plt.subplots(1, len(GEOMETRY_FEATURES), figsize=(5 * len(GEOMETRY_FEATURES), 5))
		for ax, (feature_name, _) in zip(axes, GEOMETRY_FEATURES):
			spin_values = features.get(feature_name, {})
			all_vals = []
			for spin, (bench_vals, es_vals) in sorted(spin_values.items()):
				if len(bench_vals) == 0:
					continue
				ax.scatter(es_vals, bench_vals, alpha=0.6, s=25, color=SPIN_COLORS.get(spin, "gray"), label=f"S{spin}")
				all_vals.extend(es_vals.tolist())
				all_vals.extend(bench_vals.tolist())
			if all_vals:
				lo, hi = min(all_vals), max(all_vals)
				ax.plot([lo, hi], [lo, hi], "k--", lw=1)
			ax.set_xlabel(f"electron_structure {feature_name}")
			ax.set_ylabel(f"benchmark {feature_name}")
			ax.set_title(feature_name)
			ax.grid(True, alpha=0.3)
			ax.legend()
		fig.suptitle(f"Geometry comparison - {sid}")
		plt.tight_layout()
		plot_path = os.path.join(outdir, f"{sid}_geometry_correlation.png")
		plt.savefig(plot_path, dpi=150)
		plt.close(fig)
		print(f"Wrote: {plot_path}")

	return pd.DataFrame(summary_rows).sort_values(["structure_id", "spin", "feature"]).reset_index(drop=True)


def main():
	parser = argparse.ArgumentParser(
		description="Compare energies and geometries of the triplet/quintet benchmark (full re-optimization) to the original electron structure calculations."
	)
	parser.add_argument("--benchmark_dir", default="../data/benchmark/orca", help="Directory with the benchmark ORCA output dirs.")
	parser.add_argument("--electron_structure_dir", default="../data/electron_structure", help="Directory with the electron structure ORCA output dirs.")
	parser.add_argument("--dft_dir", default="../data/dft", help="Directory with the benchmark's own-basis-set singlet ORCA output dirs ({structure_id}_0_out).")
	parser.add_argument("--outdir", default="../results/benchmark/geometry_optimization", help="Directory for output CSV/plot files.")
	args = parser.parse_args()

	script_dir = os.path.dirname(os.path.abspath(__file__))
	benchmark_dir = os.path.abspath(os.path.join(script_dir, args.benchmark_dir))
	es_dir = os.path.abspath(os.path.join(script_dir, args.electron_structure_dir))
	dft_dir = os.path.abspath(os.path.join(script_dir, args.dft_dir))
	outdir = os.path.abspath(os.path.join(script_dir, args.outdir))
	os.makedirs(outdir, exist_ok=True)

	benchmark_outs = scan_out_dirs(benchmark_dir)
	# only need the electron_structure counterparts of what's in the benchmark, plus each structure's singlet
	structure_ids = {sid for sid, _ in benchmark_outs}
	es_targets = set(benchmark_outs) | {(sid, 1) for sid in structure_ids}
	es_outs = lookup_out_dirs(es_dir, es_targets)
	dft_singlet_outs = lookup_dft_singlet_outs(dft_dir, structure_ids)
	print(
		f"Found {len(benchmark_outs)} benchmark outputs, {len(es_outs)} electron structure outputs "
		f"and {len(dft_singlet_outs)} dft singlet outputs."
	)

	energy_df = build_energy_table(benchmark_outs, es_outs, dft_singlet_outs)
	energy_path = os.path.join(outdir, "energy_comparison.csv")
	energy_df.to_csv(energy_path, index=False)
	print(f"Wrote: {energy_path}")
	spin_gaps = energy_df.dropna(subset=["rel_singlet_diff_kcal_mol"])
	if not spin_gaps.empty:
		print(
			"Mean |benchmark - electron_structure| spin-gap (rel. to own singlet) diff: "
			f"{spin_gaps['rel_singlet_diff_kcal_mol'].abs().mean():.3f} kcal/mol"
		)

	geometry_summary = compare_geometries(benchmark_outs, es_outs, outdir)
	geometry_summary_path = os.path.join(outdir, "geometry_correlation_summary.csv")
	geometry_summary.to_csv(geometry_summary_path, index=False)
	print(f"Wrote: {geometry_summary_path}")
	if not geometry_summary.empty:
		for feature_name, group in geometry_summary.groupby("feature"):
			print(f"Mean R^2 for {feature_name}: {group['r2'].dropna().mean():.3f}")


if __name__ == "__main__":
	main()
