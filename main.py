# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import pandas as pd

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# 預測費米能API👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        ### 功能
        通過匯入POSTCAR來預測費米能
    """
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("上傳")

        st.file_uploader("上傳POSTCAR")
        test = st.button("開始預測", type="secondary")
    with col2:
        st.header(
            """
        預測結果
"""
        )


if __name__ == "__main__":
    run()
