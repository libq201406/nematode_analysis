import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="线虫脂肪含量分析工具", layout="wide")
st.title("🧬 线虫脂肪含量智能分析")
st.write("请在下方的按钮处上传一张或多张显微图片，系统将自动分析并给出红色区域的面积占比。")

# 1. 侧边栏参数设置
with st.sidebar:
    st.header("参数微调")
    thresh_value = st.slider("线虫身体边界阈值 (灰度)", 150, 250, 200)
    st.info("如果透明边缘包不全，请调低该数值。")

# 2. 网页上传图片
uploaded_files = st.file_uploader("📤 点击这里上传图片 (支持 .tif, .jpg, .png)",
                                  type=['tif', 'tiff', 'jpg', 'jpeg', 'png'],
                                  accept_multiple_files=True)

if uploaded_files:
    results_data = []
    st.markdown("---")

    # 创建两列布局
    col1, col2 = st.columns([1, 1])

    for uploaded_file in uploaded_files:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            continue

        # --- 核心算法 ---
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

        # 提取红色
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower1, upper1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower2, upper2 = np.array([170, 50, 50]), np.array([180, 255, 255])
        mask_red_raw = cv2.inRange(hsv, lower1, upper1) + cv2.inRange(hsv, lower2, upper2)
        mask_final_red = cv2.bitwise_and(mask_red_raw, mask_red_raw, mask=mask_body)

        pixels_body = np.count_nonzero(mask_body)
        pixels_red = np.count_nonzero(mask_final_red)
        ratio = (pixels_red / pixels_body) * 100 if pixels_body > 0 else 0

        # --- 制作可视化结果图 ---
        vis_img = img.copy()
        cv2.drawContours(vis_img, [largest_contour], -1, (0, 255, 0), 2)  # 绿框
        # 将原来的蓝色 [255, 0, 0] 改成了红色 [0, 0, 255]
        vis_img[mask_final_red > 0] = [0, 0, 255]  # 红色标记

        vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

        results_data.append({
            "文件名": uploaded_file.name,
            "身体面积(像素)": pixels_body,
            "红色面积(像素)": pixels_red,
            "红色占比(%)": f"{ratio:.2f}%"
        })

        with col1:
            st.image(vis_img_rgb, caption=f"处理结果: {uploaded_file.name}", use_column_width=True)

        with col2:
            st.write(f"**📊 {uploaded_file.name}**")
            st.metric(label="红色占比", value=f"{ratio:.2f}%")

    # 3. 生成并下载数据表格
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
else:
    st.info("💡 请点击上方的“上传图片”按钮开始分析。")