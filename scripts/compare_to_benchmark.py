import argparse
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


EV_NM_CONSTANT = 1239.841984
NMR_PROPERTY_PATTERN = "H%/isotropic"
ORBITAL_PROPERTIES = ["HOMO-1_energy", "HOMO-0_energy", "LUMO+0_energy", "LUMO+1_energy"]


def _normalized(text: str) -> str:
	return re.sub(r"[^a-z0-9]+", "", str(text).strip().lower())


def _find_column(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> Optional[str]:
	normalized_to_column = {_normalized(col): col for col in df.columns}
	for alias in aliases:
		key = _normalized(alias)
		if key in normalized_to_column:
			return normalized_to_column[key]
	if required:
		raise ValueError(f"Could not find required column. Expected one of: {list(aliases)}")
	return None


def _parse_first_float(value) -> Optional[float]:
	if pd.isna(value):
		return None
	if isinstance(value, (int, float, np.number)):
		return float(value)
	text = str(value).replace(",", ".")
	match = re.search(r"-?\d+(?:\.\d+)?", text)
	if match is None:
		return None
	return float(match.group(0))


def _parse_integral(value) -> int:
	if pd.isna(value):
		return 1
	text = str(value).strip()
	match = re.search(r"(\d+(?:\.\d+)?)\s*H", text, flags=re.IGNORECASE)
	if match is None:
		match = re.search(r"(\d+(?:\.\d+)?)", text)
	if match is None:
		return 1
	parsed = int(round(float(match.group(1))))
	return max(parsed, 1)


def load_literature_data(xlsx_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	try:
		all_structures = pd.read_excel(xlsx_path, sheet_name="all_structures")
		nmr_raw = pd.read_excel(xlsx_path, sheet_name="nmr")
		uvvis_raw = pd.read_excel(xlsx_path, sheet_name="uvvis")
	except ImportError as exc:
		raise RuntimeError(
			"Reading .xlsx requires openpyxl. Install it with: pip install openpyxl"
		) from exc

	sid_col = _find_column(all_structures, ["structure", "structure_id", "id"])
	paper_col = _find_column(all_structures, ["paper_notation", "paper notation", "structure name", "name"])
	all_structures = all_structures[[sid_col, paper_col]].rename(columns={sid_col: "structure", paper_col: "paper_notation"})
	all_structures = all_structures.dropna(subset=["structure", "paper_notation"])
	all_structures["structure"] = all_structures["structure"].astype(str).str.strip()
	all_structures["paper_notation"] = all_structures["paper_notation"].astype(str).str.strip()

	nmr_name_col = _find_column(nmr_raw, ["Structure name", "structure_name", "paper_notation", "name"])
	nmr_shift_col = _find_column(nmr_raw, ["shift", "ppm"])
	nmr_integral_col = _find_column(nmr_raw, ["integral"], required=False)
	nmr_df = nmr_raw[[nmr_name_col, nmr_shift_col] + ([nmr_integral_col] if nmr_integral_col else [])].rename(
		columns={nmr_name_col: "paper_notation", nmr_shift_col: "shift", nmr_integral_col: "integral"}
	)
	nmr_df["paper_notation"] = nmr_df["paper_notation"].astype(str).str.strip()
	nmr_df["shift"] = nmr_df["shift"].apply(_parse_first_float)
	if "integral" not in nmr_df.columns:
		nmr_df["integral"] = 1
	nmr_df["integral"] = nmr_df["integral"].apply(_parse_integral)
	nmr_df = nmr_df.dropna(subset=["paper_notation", "shift"])

	uv_name_col = _find_column(uvvis_raw, ["structure_name", "structure name", "paper_notation", "name"])
	uv_lambda_col = _find_column(uvvis_raw, ["lambdamax", "lambda_max", "lambda max", "wavelength", "nm"])
	uv_notes_col = _find_column(uvvis_raw, ["notes", "note"], required=False)
	uvvis_df = uvvis_raw[[uv_name_col, uv_lambda_col] + ([uv_notes_col] if uv_notes_col else [])].rename(
		columns={uv_name_col: "paper_notation", uv_lambda_col: "lambdamax", uv_notes_col: "notes"}
	)
	uvvis_df["paper_notation"] = uvvis_df["paper_notation"].astype(str).str.strip()
	uvvis_df["lambdamax"] = uvvis_df["lambdamax"].apply(_parse_first_float)
	if "notes" not in uvvis_df.columns:
		uvvis_df["notes"] = ""
	uvvis_df["notes"] = uvvis_df["notes"].fillna("").astype(str)
	uvvis_df = uvvis_df.dropna(subset=["paper_notation", "lambdamax"])

	return all_structures, nmr_df, uvvis_df


def connect_session(db_path: str):
	engine = create_engine(f"sqlite:///{db_path}")
	session = sessionmaker(bind=engine)()
	return engine, session


def fetch_h_shieldings(engine, structure_ids: List[str]) -> pd.DataFrame:
	if not structure_ids:
		return pd.DataFrame(columns=["structure", "property", "value"])
	placeholders = ",".join(["?"] * len(structure_ids))
	query = (
		"SELECT structure, property, value "
		"FROM structure_properties "
		"WHERE source='orca_nmr/shielding' "
		"AND property LIKE ? "
		f"AND structure IN ({placeholders})"
	)
	params = [NMR_PROPERTY_PATTERN] + structure_ids
	return pd.read_sql_query(query, engine, params=tuple(params))


def fetch_orbital_levels(engine, structure_ids: List[str]) -> pd.DataFrame:
	if not structure_ids:
		return pd.DataFrame(columns=["structure"] + ORBITAL_PROPERTIES)
	placeholders = ",".join(["?"] * len(structure_ids))
	prop_placeholders = ",".join(["?"] * len(ORBITAL_PROPERTIES))
	query = (
		"SELECT structure, property, value "
		"FROM structure_properties "
		"WHERE source='parser/base_calculation' "
		f"AND property IN ({prop_placeholders}) "
		f"AND structure IN ({placeholders})"
	)
	params = ORBITAL_PROPERTIES + structure_ids
	raw = pd.read_sql_query(query, engine, params=tuple(params))
	if raw.empty:
		return pd.DataFrame(columns=["structure"] + ORBITAL_PROPERTIES)
	pivoted = raw.pivot_table(index="structure", columns="property", values="value", aggfunc="first").reset_index()
	for column in ORBITAL_PROPERTIES:
		if column not in pivoted.columns:
			pivoted[column] = np.nan
	return pivoted[["structure"] + ORBITAL_PROPERTIES]


def _expanded_literature_shifts(group: pd.DataFrame) -> List[float]:
	expanded = []
	for _, row in group.iterrows():
		expanded.extend([float(row["shift"])] * int(row["integral"]))
	return sorted(expanded)


def build_nmr_comparison(mapping_df: pd.DataFrame, nmr_df: pd.DataFrame, calc_shieldings: pd.DataFrame) -> pd.DataFrame:
	nmr_mapped = nmr_df.merge(mapping_df, on="paper_notation", how="inner")
	calc_grouped = calc_shieldings.groupby("structure")["value"].apply(lambda s: sorted([- float(v) for v in s])).to_dict()

	records = []
	for (structure, paper_notation), lit_group in nmr_mapped.groupby(["structure", "paper_notation"]):
		lit_values = _expanded_literature_shifts(lit_group)
		calc_values = calc_grouped.get(structure, [])

		n_pair = min(len(lit_values), len(calc_values))
		if n_pair > 0:
			lit_arr = np.array(lit_values[:n_pair], dtype=float)
			calc_arr = np.array(calc_values[:n_pair], dtype=float)
			diffs = calc_arr - lit_arr
			mae = float(np.mean(np.abs(diffs)))
			rmse = float(np.sqrt(np.mean(diffs**2)))
			mse = float(np.mean(diffs))
		else:
			mae = np.nan
			rmse = np.nan
			mse = np.nan

		records.append(
			{
				"structure": structure,
				"paper_notation": paper_notation,
				"n_literature": len(lit_values),
				"n_calculated": len(calc_values),
				"n_paired": n_pair,
				"mean_signed_error_calc_minus_lit": mse,
				"mae": mae,
				"rmse": rmse,
				"lit_sorted_ppm": ";".join(f"{x:.3f}" for x in lit_values),
				"calc_sorted_ppm": ";".join(f"{x:.3f}" for x in calc_values),
			}
		)

	return pd.DataFrame(records).sort_values(["structure", "paper_notation"]).reset_index(drop=True)


def classify_uvvis_band(notes: str) -> str:
	text = str(notes).lower()
	if "soret" in text:
		return "soret"
	if re.search(r"\bq\b", text):
		return "q"
	return "other"


def build_soret_comparison(mapping_df: pd.DataFrame, uvvis_df: pd.DataFrame, orbitals_df: pd.DataFrame) -> pd.DataFrame:
	uv = uvvis_df.copy()
	uv["band"] = uv["notes"].apply(classify_uvvis_band)
	soret = uv[uv["band"] == "soret"].copy()
	if soret.empty:
		return pd.DataFrame(
			columns=[
				"structure",
				"paper_notation",
				"soret_nm",
				"soret_eV",
				"HOMO-1_energy",
				"HOMO-0_energy",
				"LUMO+0_energy",
				"LUMO+1_energy",
				"d_homo_lumo",
				"d_homominus1_lumoplus1",
				"d_homo_lumoplus1",
				"d_homominus1_lumo",
				"q_model_eV",
				"soret_model_eV",
				"soret_model_error_eV",
				"soret_model_abs_error_eV",
			]
		)

	soret = soret.merge(mapping_df, on="paper_notation", how="inner")
	soret_summary = soret.groupby(["structure", "paper_notation"], as_index=False)["lambdamax"].mean()
	soret_summary = soret_summary.rename(columns={"lambdamax": "soret_nm"})

	merged = soret_summary.merge(orbitals_df, on="structure", how="left")
	merged["soret_eV"] = EV_NM_CONSTANT / merged["soret_nm"]

	merged["d_homo_lumo"] = merged["LUMO+0_energy"] - merged["HOMO-0_energy"]
	merged["d_homominus1_lumoplus1"] = merged["LUMO+1_energy"] - merged["HOMO-1_energy"]
	merged["d_homo_lumoplus1"] = merged["LUMO+1_energy"] - merged["HOMO-0_energy"]
	merged["d_homominus1_lumo"] = merged["LUMO+0_energy"] - merged["HOMO-1_energy"]

	# Gouterman-style 4-orbital decomposition: Q-like and Soret-like paired transitions.
	merged["q_model_eV"] = (merged["d_homo_lumo"] + merged["d_homominus1_lumoplus1"]) / 2.0
	merged["soret_model_eV"] = (merged["d_homo_lumoplus1"] + merged["d_homominus1_lumo"]) / 2.0
	merged["soret_model_error_eV"] = merged["soret_model_eV"] - merged["soret_eV"]
	merged["soret_model_abs_error_eV"] = merged["soret_model_error_eV"].abs()

	columns = [
		"structure",
		"paper_notation",
		"soret_nm",
		"soret_eV",
		"HOMO-1_energy",
		"HOMO-0_energy",
		"LUMO+0_energy",
		"LUMO+1_energy",
		"d_homo_lumo",
		"d_homominus1_lumoplus1",
		"d_homo_lumoplus1",
		"d_homominus1_lumo",
		"q_model_eV",
		"soret_model_eV",
		"soret_model_error_eV",
		"soret_model_abs_error_eV",
	]
	return merged[columns].sort_values(["structure", "paper_notation"]).reset_index(drop=True)


def main():
	parser = argparse.ArgumentParser(description="Compare literature benchmark values to calculated DB values.")
	parser.add_argument("--db", default="../main.db", help="Path to SQLite database.")
	parser.add_argument("--xlsx", default="../structure_details.xlsx", help="Path to literature xlsx file.")
	parser.add_argument("--outdir", default="../results/benchmark", help="Directory for output CSV files.")
	args = parser.parse_args()

	script_dir = os.path.dirname(os.path.abspath(__file__))
	db_path = os.path.abspath(os.path.join(script_dir, args.db))
	xlsx_path = os.path.abspath(os.path.join(script_dir, args.xlsx))
	outdir = os.path.abspath(os.path.join(script_dir, args.outdir))
	os.makedirs(outdir, exist_ok=True)

	mapping_df, nmr_df, uvvis_df = load_literature_data(xlsx_path)
	literature_names = set(nmr_df["paper_notation"]).union(set(uvvis_df["paper_notation"]))
	mapping_df = mapping_df[mapping_df["paper_notation"].isin(literature_names)].copy()
	structure_ids = sorted(mapping_df["structure"].unique().tolist())

	engine, _ = connect_session(db_path)
	calc_h_shieldings = fetch_h_shieldings(engine, structure_ids)
	orbitals_df = fetch_orbital_levels(engine, structure_ids)

	nmr_comparison = build_nmr_comparison(mapping_df, nmr_df, calc_h_shieldings)
	soret_comparison = build_soret_comparison(mapping_df, uvvis_df, orbitals_df)

	nmr_path = os.path.join(outdir, "nmr_comparison.csv")
	soret_path = os.path.join(outdir, "soret_4orbital_comparison.csv")
	nmr_comparison.to_csv(nmr_path, index=False)
	soret_comparison.to_csv(soret_path, index=False)

	print(f"Mapped literature structures: {len(structure_ids)}")
	print(f"NMR rows with mapped structures: {len(nmr_comparison)}")
	print(f"Soret rows with mapped structures: {len(soret_comparison)}")
	if not nmr_comparison.empty:
		print(f"NMR paired MAE (mean over structures): {nmr_comparison['mae'].dropna().mean():.3f} ppm")
	if not soret_comparison.empty:
		print(
			"Soret 4-orbital MAE: "
			f"{soret_comparison['soret_model_abs_error_eV'].dropna().mean():.3f} eV"
		)
	print(f"Wrote: {nmr_path}")
	print(f"Wrote: {soret_path}")

	# Create NMR scatter plots per structure
	nmr_structures = nmr_comparison["structure"].unique()
	for structure in nmr_structures:
		struct_data = nmr_comparison[nmr_comparison["structure"] == structure]
		lit_values = []
		calc_values = []
		for _, row in struct_data.iterrows():
			if pd.notna(row["lit_sorted_ppm"]):
				lit = [float(x) for x in row["lit_sorted_ppm"].split(";") if len(x) > 0]
				calc = [float(x) for x in row["calc_sorted_ppm"].split(";") if len(x) > 0]
				n = min(len(lit), len(calc))
				lit_values.extend(lit[:n])
				calc_values.extend(calc[:n])
		
		if lit_values and calc_values:
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.scatter(lit_values, calc_values, alpha=0.6, s=50)
			# Fit best fit line
			coeffs = np.polyfit(lit_values, calc_values, 1)
			poly = np.poly1d(coeffs)
			x_line = np.array([min(lit_values), max(lit_values)])
			ax.plot(x_line, poly(x_line), "r-", lw=2, label=f"Best fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}")
			ax.set_xlabel("Literature NMR Shift (ppm)")
			ax.set_ylabel("Calculated NMR Shift (ppm)")
			ax.set_title(f"NMR Shifts - {structure}")
			ax.grid(True, alpha=0.3)
			ax.legend()
			plt.tight_layout()
			nmr_plot_path = os.path.join(outdir, f"nmr_scatter_{structure}.png")
			plt.savefig(nmr_plot_path, dpi=150)
			plt.close()
			print(f"Wrote: {nmr_plot_path}")

	# Create Soret scatter plot
	if not soret_comparison.empty:
		paired_indices = soret_comparison["soret_eV"].notna() & soret_comparison["d_homo_lumo"].notna()
		
		if paired_indices.sum() > 0:
			lit_soret = soret_comparison.loc[paired_indices, "soret_eV"].values
			calc_soret = soret_comparison.loc[paired_indices, "d_homo_lumo"].values
			
			fig, ax = plt.subplots(figsize=(6, 6))
			ax.scatter(lit_soret, calc_soret, alpha=0.6, s=50)
			min_val = min(min(lit_soret), min(calc_soret))
			max_val = max(max(lit_soret), max(calc_soret))
			ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)
			ax.set_xlabel("Literature Soret Band (eV)")
			ax.set_ylabel("Calculated Soret Band (eV)")
			ax.set_title("Soret Bands - All Structures")
			ax.grid(True, alpha=0.3)
			plt.tight_layout()
			soret_plot_path = os.path.join(outdir, "soret_scatter.png")
			plt.savefig(soret_plot_path, dpi=150)
			plt.close()
			print(f"Wrote: {soret_plot_path}")


if __name__ == "__main__":
	main()
