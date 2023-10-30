import pandas as pd
import joblib


def predictFermi(data: pd.DataFrame):
    data["過渡金屬"] = data["過渡金屬"].map({None: 0, "Ag": 1, "Au": 2, "Pd": 3, "Pt": 4})
    data["過渡金屬"] = data["過渡金屬"].fillna(0)
    data["放置點位"] = data["放置點位"].map({"Ga3c": 1, "Zn3c": 2, "O3c": 3, "O4c": 4})
    data["吸附氣體的分子式"] = data["吸附氣體的分子式"].map(
        {"H2S": 1, "CO": 2, "NO": 3, "0": 0, "0.0": 0, "CO2": 4, "NO2": 5, "O3": 6}
    )
    data["最接近該點位之氣體原子"] = data["最接近該點位之氣體原子"].map(
        {"O": 1, "N": 2, "C": 3, "S": 4, "H": 5}
    )
    data = data.fillna(-1)
    # 提取特徵和目標列
    features = data[
        [
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
    ]
    # 載入模型
    model_filename = "models/linear_regression_model.pkl"
    loaded_model = joblib.load(model_filename)
    # 使用載入的模型進行預測
    predictions = loaded_model.predict(features)
    # 處理預測結果
    return predictions.round(4)
