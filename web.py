import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="线虫脂肪含量智能分析", layout="wide")

# ================== 侧边栏参数微调 ==================
with st.sidebar:
    st.header("参数微调")
    thresh_value = st.slider("线虫身体边界阈值 (灰度)", 150, 250, 200)
    st.info("如果透明边缘包不全，请调低该数值。")

# ================== 自定义 CSS 样式 ==================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    html, body, [class*="css"] {
        font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    /* 美化“开始分析”按钮 */
    div.stButton > button {
        background-color: #1e4d8f;
        color: white;
        border-radius: 30px;
        border: none;
        padding: 12px 40px;
        font-size: 18px;
        font-weight: 600;
        transition: 0.3s;
        display: block;
        margin: 20px auto;
    }
    div.stButton > button:hover {
        background-color: #153a6b;
        box-shadow: 0 4px 12px rgba(30,77,143,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ================== 初始化状态 ==================
if 'analyze_clicked' not in st.session_state:
    st.session_state['analyze_clicked'] = False
if 'last_uploaded_file' not in st.session_state:
    st.session_state['last_uploaded_file'] = None

# ================== 页面顶部布局 ==================
try:
    st.image("https://raw.githubusercontent.com/libq201406/nematode_analysis/main/banner.png", use_column_width=True)
except:
    st.warning("⚠️ 找不到 banner.png 文件。")

st.markdown("---")

# 1. 文件上传组件
uploaded_files = st.file_uploader("📤 请在这里上传线虫显微图片 (支持多选)",
                                  type=['tif', 'tiff', 'jpg', 'jpeg', 'png'],
                                  accept_multiple_files=True,
                                  key="file_uploader")

# 2. 上传完成提示（核心新增逻辑）
if uploaded_files:
    # 如果用户刚上传了文件，或者当前上传的文件跟上次分析的文件不一样
    current_files = [f.name for f in uploaded_files]
    if st.session_state.get('last_uploaded_file') != current_files:
        st.success(f"✅ 文件上传完成！当前已选择 {len(uploaded_files)} 张图片，请点击下方【开始分析】按钮启动分析。")

# 3. 开始分析按钮及校验逻辑
if st.button("🚀 开始分析", key="analyze_btn"):
    if uploaded_files:
        st.session_state['analyze_clicked'] = True
        st.session_state['last_uploaded_file'] = [f.name for f in uploaded_files]
    else:
        # 未上传文件时的提醒
        st.warning("⚠️ 请先上传图片文件，再点击开始分析！")

# ================== 核心计算逻辑 ==================
if uploaded_files and st.session_state['analyze_clicked']:
    current_files = [f.name for f in uploaded_files]

    # 如果用户换了文件，重置状态
    if current_files != st.session_state['last_uploaded_file']:
        st.session_state['analyze_clicked'] = False
        st.info("💡 检测到您更换了图片，请重新点击【开始分析】按钮。")
    else:
        st.markdown("---")
        results_data = []

        with st.spinner('正在分析中，请稍候...'):
            for uploaded_file in uploaded_files:
                file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                # 1. 提取身体掩膜
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask_body_raw = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY_INV)

                kernel = np.ones((7, 7), np.uint8)
                mask_body_closed = cv2.morphologyEx(mask_body_raw, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(mask_body_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    results_data.append({"文件名": uploaded_file.name, "状态": "未检测到线虫", "占比": "0%"})
                    continue

                largest_contour = max(contours, key=cv2.contourArea)
                mask_body = np.zeros_like(mask_body_closed)
                cv2.drawContours(mask_body, [largest_contour], -1, 255, -1)

                # 2. 提取红色区域
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                lower1, upper1 = np.array([0, 50, 50]), np.array([10, 255, 255])
                lower2, upper2 = np.array([170, 50, 50]), np.array([180, 255, 255])
                mask_red_raw = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
                mask_final_red = cv2.bitwise_and(mask_red_raw, mask_red_raw, mask=mask_body)

                pixels_body = np.count_nonzero(mask_body)
                pixels_red = np.count_nonzero(mask_final_red)
                ratio = (pixels_red / pixels_body) * 100 if pixels_body > 0 else 0

                # 3. 可视化
                vis_img = img.copy()
                cv2.drawContours(vis_img, [largest_contour], -1, (0, 255, 0), 2)  # 绿框
                vis_img[mask_final_red > 0] = [0, 0, 255]  # 红色标记
                vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

                results_data.append({
                    "文件名": uploaded_file.name,
                    "身体面积(像素)": pixels_body,
                    "红色面积(像素)": pixels_red,
                    "红色占比(%)": f"{ratio:.2f}%"
                })

                st.image(vis_img_rgb, caption=f"处理结果: {uploaded_file.name}", use_column_width=True)
                st.metric(label=f"{uploaded_file.name} - 红色占比", value=f"{ratio:.2f}%")

        if results_data:
            st.markdown("---")
            st.subheader("📋 整体数据统计")
            df = pd.DataFrame(results_data)
            st.dataframe(df)

            csv_data = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                label="📥 一键下载全部统计数据 (CSV)",
                data=csv_data,
                file_name="area_analysis_results.csv",
                mime="text/csv"
            )