import pandas as pd
from ase.io.vasp import read_vasp
import numpy as np
import math


def preprocessData(poscar_file_path: str) -> pd.DataFrame:
    df_list = []
    try:
    # 使用 ASE 的 read_vasp 函數讀取 POSCAR 文件
        atoms = read_vasp(poscar_file_path)
    # 以下部分代码保持不变
    except Exception as e:
    # 在出现异常时生成"error"数据
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
#-------------------------------------------------------------------------------------------
        # 獲取晶格矩陣
        cell_matrix = atoms.get_cell()

        # 提取晶格常數和晶格角度
        a, b, c = cell_matrix[0], cell_matrix[1], cell_matrix[2]

        # 計算晶格角度（以度為單位）
        alpha = round(atoms.get_cell_lengths_and_angles()[3], 2)
        beta = round(atoms.get_cell_lengths_and_angles()[4], 2)
        gamma = round(atoms.get_cell_lengths_and_angles()[5], 2)

        # 獲取晶格矩陣
        cell_matrix = atoms.get_cell()

        # 提取晶格常數
        LengthA = round(np.linalg.norm(cell_matrix[0]), 4)
        LengthB = round(np.linalg.norm(cell_matrix[1]), 4)
        LengthC = round(np.linalg.norm(cell_matrix[2]), 4)
    #-------------------------------------------------------------------------------------------
        # 使用 ASE 的 read_vasp 函數讀取 POSCAR 文件
        atoms = read_vasp(poscar_file_path)

        # 自定義排序關鍵字函數，按 Z 軸坐標從大到小排序

        def sort_by_z(atom):
            return atom.position[2]

        # 使用 sorted 函數對原子列表進行排序
        sorted_atoms = sorted(atoms, key=sort_by_z)
    #-------------------------------------------------------------------------------------------
        # 提取Z軸坐標從小到大的16顆Zn原子
        zn_atoms = [atom for atom in sorted_atoms if atom.symbol == "Zn"][:16]

        # 提取Z軸坐標從小到大的32顆Ga原子
        ga_atoms = [atom for atom in sorted_atoms if atom.symbol == "Ga"][:32]

        # 提取Z軸坐標從小到大的64顆O原子
        o_atoms = [atom for atom in sorted_atoms if atom.symbol == "O"][:64]

        # 合併基板原子坐標
        substrate_atoms = zn_atoms + ga_atoms + o_atoms

        # 從原子列表中去除基板原子
        remaining_atoms = [
            atom for atom in sorted_atoms if atom not in substrate_atoms]
    #-------------------------------------------------------------------------------------------
        top_4_zn_atoms = sorted(zn_atoms, key=lambda atom: atom.position[2], reverse=True)[:4]
        top_8_ga_atoms = sorted(ga_atoms, key=lambda atom: atom.position[2], reverse=True)[:8]
        top_16_o_atoms = sorted(o_atoms, key=lambda atom: atom.position[2], reverse=True)[:16]
    #-------------------------------------------------------------------------------------------
        target_coords = {
            "Zn3c": (8.80020, 2.98463),
            "Ga3c": (7.07185, 0.00621015),
            "O3c": (5.34588, 2.99568),
            "O4c": (7.85003, 1.35907)
        }

        # 初始化最接近原子和距離
        closest_atoms = {
            "Zn3c": None,
            "Ga3c": None,
            "O3c": None,
            "O4c": None
        }
        min_distances = {
            "Zn3c": float('inf'),
            "Ga3c": float('inf'),
            "O3c": float('inf'),
            "O4c": float('inf')
        }

        # 找到 Zn 元素中最接近 Zn3c 目標座標的原子
        for atom in top_4_zn_atoms:
            atom_coords = atom.position
            distance = np.linalg.norm(target_coords["Zn3c"] - atom_coords[:2])
            if distance < min_distances["Zn3c"]:
                min_distances["Zn3c"] = distance
                closest_atoms["Zn3c"] = atom

        # 找到 Ga 元素中最接近 Ga3c 目標座標的原子
        for atom in top_8_ga_atoms:
            atom_coords = atom.position
            distance = np.linalg.norm(target_coords["Ga3c"] - atom_coords[:2])
            if distance < min_distances["Ga3c"]:
                min_distances["Ga3c"] = distance
                closest_atoms["Ga3c"] = atom

        # 找到 O 元素中最接近 O3c 目標座標的原子
        for atom in top_16_o_atoms:
            atom_coords = atom.position
            distance = np.linalg.norm(target_coords["O3c"] - atom_coords[:2])
            if distance < min_distances["O3c"]:
                min_distances["O3c"] = distance
                closest_atoms["O3c"] = atom

        # 找到 O 元素中最接近 O4c 目標座標的原子
        for atom in top_16_o_atoms:
            atom_coords = atom.position
            distance = np.linalg.norm(target_coords["O4c"] - atom_coords[:2])
            if distance < min_distances["O4c"]:
                min_distances["O4c"] = distance
                closest_atoms["O4c"] = atom
    #-------------------------------------------------------------------------------------------
        o_count = sum(1 for atom in remaining_atoms if atom.symbol == "O")

        if o_count > 4:
            # 獲取剩餘原子中所有O原子的列表
            o_remaining_atoms = [
                atom for atom in remaining_atoms if atom.symbol == "O"]

            # 根據Z軸坐標從小到大對氧原子進行排序

            # 去除Z軸最小的8顆氧原子
            sorted_o_atoms = [
                atom for atom in o_remaining_atoms if atom.symbol == "O"][:8]

            # 更新剩餘原子列表，去除Z軸最小的8顆氧原子
            remaining_atoms = [
                atom for atom in remaining_atoms if atom not in sorted_o_atoms
            ]

            for atom in sorted_o_atoms:
                substrate_atoms.append(atom)
    #-------------------------------------------------------------------------------------------
        # 定義過渡金屬元素列表
        transition_metals = ["Ag", "Au", "Pd", "Pt"]

        # 檢查剩餘原子中是否含有過渡金屬元素
        has_transition_metal = any(
            atom.symbol in transition_metals for atom in remaining_atoms
        )

        if has_transition_metal:
            # 提取過渡金屬元素並打印
            transition_metal_atoms = [
                atom for atom in remaining_atoms if atom.symbol in transition_metals
            ]
            transition_metal_symbols = [
                atom.symbol for atom in transition_metal_atoms]
            # 更新基板座標
            substrate_atoms.extend(transition_metal_atoms)
            # 去除過渡金屬元素
            remaining_atoms = [
                atom for atom in remaining_atoms if atom.symbol not in transition_metals
            ]
    #-------------------------------------------------------------------------------------------
        # 計算基板原子的數量
        substrate_atom_count = len(substrate_atoms)

        # 計算剩餘原子的數量
        remaining_atom_count = len(remaining_atoms)
    #-------------------------------------------------------------------------------------------
        if remaining_atom_count > 0:
            # 初始化元素數量字典
            element_counts = {}

            # 遍歷剩餘原子，計算每個元素的數量
            for atom in remaining_atoms:
                element_symbol = atom.symbol
                if element_symbol in element_counts:
                    element_counts[element_symbol] += 1
                else:
                    element_counts[element_symbol] = 1

            # 構建分子式字串
            molecular_formula = ""
            for element_symbol, count in sorted(element_counts.items()):
                if count == 1:
                    molecular_formula += element_symbol
                else:
                    molecular_formula += f"{element_symbol}{count}"
    #-------------------------------------------------------------------------------------------
            # 初始化坐標總和
            x_sum = 0.0
            y_sum = 0.0
            z_sum = 0.0

            # 遍歷剩餘原子，計算坐標總和
            for atom in remaining_atoms:
                x_sum += atom.position[0]
                y_sum += atom.position[1]
                z_sum += atom.position[2]

            # 計算坐標平均值
            num_atoms = len(remaining_atoms)
            center_x = x_sum / num_atoms
            center_y = y_sum / num_atoms
            center_z = z_sum / num_atoms
    #-------------------------------------------------------------------------------------------
            molecular_center_coords = np.array([center_x, center_y])

            # 目标点位座标
            target_point_coords = np.array([
                closest_atoms["Zn3c"].position[:2],
                closest_atoms["Ga3c"].position[:2],
                closest_atoms["O3c"].position[:2],
                closest_atoms["O4c"].position[:2]
            ])

            # 计算分子中心位置与目标点位之间的距离
            distances = np.linalg.norm(target_point_coords - molecular_center_coords, axis=1)

            # 找到最近的点位
            closest_point_index = np.argmin(distances)
            closest_point_coords = target_point_coords[closest_point_index]

            # 打印最近点位的座标
            if closest_point_index == 0:
                closest_point_name = "Zn3c"
            elif closest_point_index == 1:
                closest_point_name = "Ga3c"
            elif closest_point_index == 2:
                closest_point_name = "O3c"
            else:
                closest_point_name = "O4c"
            x_coord, y_coord = closest_point_coords
    #-------------------------------------------------------------------------------------------
            target_point_coords = np.array([
                closest_atoms["Zn3c"].position,
                closest_atoms["Ga3c"].position,
                closest_atoms["O3c"].position,
                closest_atoms["O4c"].position
            ])

            # 初始化最接近原子的变量
            closest_atom = None
            closest_distance = float('inf')

            for atom in remaining_atoms:
                x_rounded = round(atom.position[0], 3)
                y_rounded = round(atom.position[1], 3)
                z_rounded = round(atom.position[2], 3)
                
                # 计算距离
                distance = math.sqrt((x_rounded - closest_point_coords[0])**2 +
                                    (y_rounded - closest_point_coords[1])**2 +
                                    (z_rounded - closest_atoms[closest_point_name].position[2])**2)
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_atom = atom
    #-------------------------------------------------------------------------------------------
            # 气体分子的中心位置座标
            gas_molecule_center_coords = np.array([center_x, center_y, center_z])

            # 初始化基板Z轴最高原子和Z轴坐标
            highest_atom = None
            max_z_coordinate = -float('inf')

            # 遍历基板原子，找到Z轴最高的原子
            for atom in substrate_atoms:
                z_coordinate = atom.position[2]
                if z_coordinate > max_z_coordinate:
                    max_z_coordinate = z_coordinate
                    highest_atom = atom

            # 计算气体分子中心位置与Z轴最高原子之间的距离（只比较 Z 轴坐标）
            distance_to_highest_atom = np.abs(gas_molecule_center_coords[2] - max_z_coordinate)
    #-------------------------------------------------------------------------------------------
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
                ', '.join(transition_metal_symbols) if has_transition_metal else 0,
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
                ', '.join(transition_metal_symbols) if has_transition_metal else 0,
            ]
    return data
