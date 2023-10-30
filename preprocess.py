import pandas as pd
from ase.io.vasp import read_vasp
import numpy as np
import math


def preprocessData(poscar_file_path: str) -> pd.DataFrame:
    df_list = []
    try:
        atoms = read_vasp(poscar_file_path)
    except Exception as e:
        data = {
            "晶格常數a": "error",
            "晶格常數b": "error",
            "晶格常數c": "error",
            "夾角(alpha)": "error",
            "夾角(beta)": "error",
            "夾角(gamma)": "error",
            "吸附氣體的分子式": "error",
            "氣體原子的數量": "error",
            "基板原子的數量": "error",
            "有無氧鈍化表面": "error",
            "有無過渡金屬": "error",
            "過渡金屬": "error",
            "吸附氣體的中心X坐標": "error",
            "吸附氣體的中心Y坐標": "error",
            "吸附氣體的中心Z坐標": "error",
            "放置點位": "error",
            "最接近該點位之氣體原子": "error",
            "氣體與基板的距離 (Å)": "error",
        }

        df = pd.DataFrame(data, index=[poscar_file_path])
        df_list.append(df)

        print(f"Error in folder {poscar_file_path}: {str(e)}")

    else:
        cell_matrix = atoms.get_cell()

        alpha = round(atoms.get_cell_lengths_and_angles()[3], 2)
        beta = round(atoms.get_cell_lengths_and_angles()[4], 2)
        gamma = round(atoms.get_cell_lengths_and_angles()[5], 2)

        cell_matrix = atoms.get_cell()

        LengthA = round(np.linalg.norm(cell_matrix[0]), 4)
        LengthB = round(np.linalg.norm(cell_matrix[1]), 4)
        LengthC = round(np.linalg.norm(cell_matrix[2]), 4)

        atoms = read_vasp(poscar_file_path)

        def sort_by_z(atom):
            return atom.position[2]

        sorted_atoms = sorted(atoms, key=sort_by_z)

        zn_atoms = [atom for atom in sorted_atoms if atom.symbol == "Zn"][:16]
        ga_atoms = [atom for atom in sorted_atoms if atom.symbol == "Ga"][:32]

        o_atoms = [atom for atom in sorted_atoms if atom.symbol == "O"][:64]

        substrate_atoms = zn_atoms + ga_atoms + o_atoms

        remaining_atoms = [atom for atom in sorted_atoms if atom not in substrate_atoms]

        o_count = sum(1 for atom in remaining_atoms if atom.symbol == "O")

        if o_count > 4:
            o_remaining_atoms = [atom for atom in remaining_atoms if atom.symbol == "O"]

            sorted_o_atoms = [atom for atom in o_remaining_atoms if atom.symbol == "O"][
                :8
            ]

            remaining_atoms = [
                atom for atom in remaining_atoms if atom not in sorted_o_atoms
            ]

            for atom in sorted_o_atoms:
                substrate_atoms.append(atom)

        transition_metals = ["Ag", "Au", "Pd", "Pt"]

        has_transition_metal = any(
            atom.symbol in transition_metals for atom in remaining_atoms
        )

        if has_transition_metal:
            transition_metal_atoms = [
                atom for atom in remaining_atoms if atom.symbol in transition_metals
            ]
            transition_metal_symbols = [atom.symbol for atom in transition_metal_atoms]

            substrate_atoms.extend(transition_metal_atoms)

            remaining_atoms = [
                atom for atom in remaining_atoms if atom.symbol not in transition_metals
            ]

        substrate_atom_count = len(substrate_atoms)

        remaining_atom_count = len(remaining_atoms)
        if remaining_atom_count > 0:
            element_counts = {}

            for atom in remaining_atoms:
                element_symbol = atom.symbol
                if element_symbol in element_counts:
                    element_counts[element_symbol] += 1
                else:
                    element_counts[element_symbol] = 1

            molecular_formula = ""
            for element_symbol, count in sorted(element_counts.items()):
                if count == 1:
                    molecular_formula += element_symbol
                else:
                    molecular_formula += f"{element_symbol}{count}"

            x_sum = 0.0
            y_sum = 0.0
            z_sum = 0.0

            for atom in remaining_atoms:
                x_sum += atom.position[0]
                y_sum += atom.position[1]
                z_sum += atom.position[2]

            num_atoms = len(remaining_atoms)
            center_x = x_sum / num_atoms
            center_y = y_sum / num_atoms
            center_z = z_sum / num_atoms

            if LengthC >= 40:
                target_coords = np.array(
                    [
                        (5.34588, 2.99568, 28.0280),
                        (7.85003, 1.35907, 28.2705),
                        (7.07185, 0.00621015, 29.4480),
                        (8.80020, 2.98463, 28.4945),
                    ]
                )

                closest_atoms = [None] * 4
                min_distances = [float("inf")] * 4

                for atom in substrate_atoms:
                    atom_coords = atom.position
                    distances = np.linalg.norm(target_coords - atom_coords, axis=1)

                    for i, distance in enumerate(distances):
                        if distance < min_distances[i]:
                            min_distances[i] = distance
                            closest_atoms[i] = atom

                O3c, O4c, Ga3c, Zn3c = closest_atoms

            else:
                target_coords = np.array(
                    [
                        (5.18004, 2.99070, 19.7200),
                        (7.69029, 1.35754, 19.9715),
                        (6.90655, 0.000299070, 21.1379),
                        (8.63348, 2.99058, 20.2242),
                    ]
                )

                closest_atoms = [None] * 4
                min_distances = [float("inf")] * 4

                for atom in substrate_atoms:
                    atom_coords = atom.position
                    distances = np.linalg.norm(target_coords - atom_coords, axis=1)

                    for i, distance in enumerate(distances):
                        if distance < min_distances[i]:
                            min_distances[i] = distance
                            closest_atoms[i] = atom

                O3c, O4c, Ga3c, Zn3c = closest_atoms
                print(O3c, O3c, O4c, Ga3c, Zn3c)

            molecular_center_coords = np.array([center_x, center_y, center_z])

            target_point_coords = np.array(
                [Ga3c.position, Zn3c.position, O3c.position, O4c.position]
            )

            distances = np.linalg.norm(
                target_point_coords - molecular_center_coords, axis=1
            )

            closest_point_index = np.argmin(distances)
            closest_point_coords = target_point_coords[closest_point_index]

            if closest_point_index == 0:
                closest_point_name = "Ga3c"
            elif closest_point_index == 1:
                closest_point_name = "Zn3c"
            elif closest_point_index == 2:
                closest_point_name = "O3c"
            else:
                closest_point_name = "O4c"

            target_point_coords = np.array(
                [Ga3c.position, Zn3c.position, O3c.position, O4c.position]
            )

            closest_atom = None
            closest_distance = float("inf")

            for atom in remaining_atoms:
                x_rounded = round(atom.position[0], 3)
                y_rounded = round(atom.position[1], 3)
                z_rounded = round(atom.position[2], 3)

                distance = math.sqrt(
                    (x_rounded - closest_point_coords[0]) ** 2
                    + (y_rounded - closest_point_coords[1]) ** 2
                    + (z_rounded - closest_point_coords[2]) ** 2
                )

                if distance < closest_distance:
                    closest_distance = distance
                    closest_atom = atom

            gas_molecule_center_coords = np.array([center_x, center_y, center_z])

            highest_atom = None
            max_z_coordinate = -float("inf")

            for atom in substrate_atoms:
                z_coordinate = atom.position[2]
                if z_coordinate > max_z_coordinate:
                    max_z_coordinate = z_coordinate
                    highest_atom = atom

            distance_to_highest_atom = np.linalg.norm(
                gas_molecule_center_coords - highest_atom.position
            )

            data = pd.DataFrame(
                columns=[
                    "晶格常數a",
                    "晶格常數b",
                    "晶格常數c",
                    "夾角(alpha)",
                    "夾角(beta)",
                    "夾角(gamma)",
                    "吸附氣體的分子式",
                    "氣體原子的數量",
                    "基板原子的數量",
                    "有無氧鈍化表面",
                    "有無過渡金屬",
                    "過渡金屬",
                    "吸附氣體的中心X坐標",
                    "吸附氣體的中心Y坐標",
                    "吸附氣體的中心Z坐標",
                    "放置點位",
                    "最接近該點位之氣體原子",
                    "氣體與基板的距離 (Å)",
                ]
            )

            data.loc[poscar_file_path] = [
                LengthA,
                LengthB,
                LengthC,
                alpha,
                beta,
                gamma,
                molecular_formula,
                remaining_atom_count,
                substrate_atom_count,
                1 if o_count > 4 else 0,
                1 if has_transition_metal else 0,
                ", ".join(transition_metal_symbols) if has_transition_metal else 0,
                f"{center_x:.4f}",
                f"{center_y:.4f}",
                f"{center_z:.4f}",
                closest_point_name,
                closest_atom.symbol,
                distance_to_highest_atom,
            ]
        else:
            data = pd.DataFrame(
                columns=[
                    "晶格常數a",
                    "晶格常數b",
                    "晶格常數c",
                    "夾角(alpha)",
                    "夾角(beta)",
                    "夾角(gamma)",
                    "吸附氣體的分子式",
                    "氣體原子的數量",
                    "基板原子的數量",
                    "有無氧鈍化表面",
                    "有無過渡金屬",
                    "過渡金屬",
                ]
            )

            data.loc[poscar_file_path] = [
                LengthA,
                LengthB,
                LengthC,
                alpha,
                beta,
                gamma,
                0,
                remaining_atom_count,
                substrate_atom_count,
                1 if o_count > 4 else 0,
                1 if has_transition_metal else 0,
                ", ".join(transition_metal_symbols) if has_transition_metal else 0,
            ]
    return data
