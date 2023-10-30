import streamlit as st
from streamlit.logger import get_logger
from preprocess import preprocessData
from predict import predictFermi
import os
import tempfile
import atexit

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# 預測費米能API👋")

    st.markdown(
        """
        ### 功能
        通過匯入POSTCAR來預測費米能
    """
    )

    def cleanup_temp_folder():
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(temp_dir)

    atexit.register(cleanup_temp_folder)
    col1, col2 = st.columns([4, 2])
    with col1:
        st.subheader("上傳")
        temp_dir = tempfile.mkdtemp()
        uploads_file = st.file_uploader("上傳POSTCAR")
        if uploads_file is not None:
            with open(os.path.join(temp_dir, uploads_file.name), "wb") as f:
                f.write(uploads_file.read())
            st.success("上傳成功")

        preprocess_data = None
        predict_data = None
        if uploads_file is not None:
            preprocess_data = preprocessData(os.path.join(temp_dir, uploads_file.name))
            st.dataframe(preprocess_data, width=300)

        predict_button = st.button("開始預測", type="secondary")

        if predict_button and uploads_file is None:
            st.error("還未上傳 POSTCAR")

    with col2:
        st.subheader(
            """
        預測結果
"""
        )
        if predict_button and preprocess_data is not None:
            try:
                predict_data = predictFermi(preprocess_data)
            except Exception as e:
                st.error("預測失敗 ")
                st.error("POSTCAR 格式有誤")

        if predict_data is not None:
            st.dataframe(predict_data, width=300)


if __name__ == "__main__":
    run()
