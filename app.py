import streamlit as st
import pandas as pd
import numpy as np
import os
import tifffile
from skimage import filters, measure, morphology, segmentation, feature
from scipy import ndimage
import matplotlib.pyplot as plt
from mvi_predictor import MVIPredictor

st.set_page_config(page_title="mviTIMES Spatial AI", page_icon="🧬", layout="wide")

st.title("🔬 mviTIMES: MVI 空间免疫预测平台")
st.markdown("欢迎使用！您可以选择直接上传已提取的坐标数据集，或者上传多色免疫荧光原始图像进行在线全流程分析。")

# ==========================================
# 核心处理函数区
# ==========================================
def load_csv_safely(file):
    """安全读取CSV文件，兼容多种编码"""
    encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']
    for enc in encodings_to_try:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法解析文件编码，请在Excel中将文件‘另存为’ -> ‘CSV (UTF-8)’ 格式后再上传。")

@st.cache_data(show_spinner=False)
def load_tiff_image(file_bytes):
    img = tifffile.imread(file_bytes)
    if img.ndim == 3 and img.shape[2] < min(img.shape[0], img.shape[1]) and img.shape[2] <= 10:
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:
        img = img[np.newaxis, :, :]
    return img

def segment_cells_dapi(dapi_channel):
    blurred = filters.gaussian(dapi_channel, sigma=1.5)
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    binary = morphology.remove_small_objects(binary, min_size=20)
    binary = ndimage.binary_fill_holes(binary)
    distance = ndimage.distance_transform_edt(binary)
    coords = feature.peak_local_max(distance, footprint=np.ones((5, 5)), labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return labels

def quantify_and_phenotype(labels, channels_dict, img_id, parent_id):
    props = measure.regionprops(labels)
    data = []
    
    img_GPC3 = channels_dict.get('GPC3.Tumor')
    img_CD34 = channels_dict.get('CD34.Endothelial')
    img_CD56 = channels_dict.get('CD56.NK')
    img_CXCR4 = channels_dict.get('CXCR4')
    img_CD163 = channels_dict.get('CD163')
    img_CXCL12 = channels_dict.get('CXCL12')

    thresholds = {}
    for name, img in channels_dict.items():
        if img is not None and name != 'DAPI':
            try:
                thresholds[name] = filters.threshold_otsu(img[img > 0])
            except:
                thresholds[name] = img.mean()

    for prop in props:
        centroid_y, centroid_x = prop.centroid
        coords = prop.coords
        
        def is_positive(img_matrix, marker_name):
            if img_matrix is None: return False
            mean_intensity = np.mean(img_matrix[coords[:, 0], coords[:, 1]])
            return mean_intensity > thresholds.get(marker_name, 0)

        cell_type = 'Other'
        if is_positive(img_CD56, 'CD56.NK') and is_positive(img_CXCR4, 'CXCR4'):
            cell_type = 'CXCR4.CD56.NK'
        elif is_positive(img_CD163, 'CD163') and is_positive(img_CXCL12, 'CXCL12'):
            cell_type = 'CXCL12.CD163.M2'
        elif is_positive(img_GPC3, 'GPC3.Tumor'):
            cell_type = 'GPC3.Tumor'
        elif is_positive(img_CD34, 'CD34.Endothelial'):
            cell_type = 'CD34.Endothelial'

        data.append({
            'Image': img_id,
            'Parent': parent_id,
            'x.axis': centroid_x,
            'y.axis': centroid_y,
            'CellType': cell_type
        })
        
    return pd.DataFrame(data)

# ==========================================
# UI 布局：双选项卡设计
# ==========================================
tab_csv, tab_tiff = st.tabs(["📄 方式一：上传空间坐标数据 (CSV)", "🌌 方式二：上传多色免疫荧光原图 (TIFF)"])

# ------------------------------------------
# 方式一：处理 CSV 数据上传
# ------------------------------------------
with tab_csv:
    st.header("基于已提取的空间坐标进行预测")
    st.write("上传的 CSV 数据必须包含：`Image`, `Parent`, `x.axis`, `y.axis`, `CellType` 列。")
    
    uploaded_csv = st.file_uploader("📂 点击这里上传您的 .csv 数据集", type=["csv"], key="csv_uploader")
    
    if uploaded_csv is not None:
        try:
            df_input = load_csv_safely(uploaded_csv)
            required_cols = {'Image', 'Parent', 'x.axis', 'y.axis', 'CellType'}
            
            if not required_cols.issubset(set(df_input.columns)):
                st.error(f"❌ 数据格式错误！缺少必要的列。您的列名：{list(df_input.columns)}")
            else:
                st.success("✅ 数据加载成功！")
                with st.expander("点击查看数据预览前 5 行"):
                    st.dataframe(df_input.head())
                
                if st.button("🚀 立即计算 mviTIMES 分数", use_container_width=True, key="btn_csv"):
                    if not os.path.exists('mvi_scoring_model.pkl'):
                        st.error("⚠️ 服务器未找到预训练模型 mvi_scoring_model.pkl。")
                    else:
                        with st.spinner('🧠 AI 正在提取百万级空间互作特征...'):
                            predictor = MVIPredictor()
                            predictor.load_model('mvi_scoring_model.pkl')
                            results_df = predictor.predict_score(df_input)
                            
                        st.balloons()
                        st.subheader("📊 预测结果 (mviTIMES Scores)")
                        st.dataframe(
                            results_df.style.background_gradient(cmap='OrRd', subset=['mviTIMES_Score']), 
                            use_container_width=True
                        )
                        csv_data = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ 下载完整预测结果 (CSV)", data=csv_data, file_name='mviTIMES_Predictions.csv', mime='text/csv')
        except Exception as e:
            st.error(f"⚠️ 处理数据时发生错误：{str(e)}")

# ------------------------------------------
# 方式二：处理 TIFF 图像上传
# ------------------------------------------
with tab_tiff:
    st.header("全自动多通道图像解析与预测")
    col_setup, col_upload = st.columns([1, 2])
    
    with col_setup:
        user_img_id = st.text_input("图像样本编号 (Image ID)", "Sample_001")
        user_parent_id = st.text_input("组织区域编号 (Parent ID)", "Region_A")
        scale_factor = st.number_input("比例尺 (1像素 = 多少微米 μm)", value=0.5, step=0.1)

    with col_upload:
        uploaded_tiff = st.file_uploader("支持 .tiff, .qptiff 格式（建议单张大小不超过 50MB）", type=["tif", "tiff", "qptiff"], key="tiff_uploader")

    if uploaded_tiff is not None:
        try:
            with st.spinner("正在解析 TIFF 高维矩阵..."):
                img_matrix = load_tiff_image(uploaded_tiff)
                num_channels = img_matrix.shape[0]
            
            st.success(f"成功读取图像！检测到 **{num_channels}** 个通道。")
            st.markdown("---")
            
            st.subheader("👁️ 步骤 3：通道映射 (Visual Channel Mapping)")
            marker_options = ["忽略该通道 (Ignore)", "DAPI (细胞核)", "GPC3.Tumor (肿瘤)", 
                              "CD34.Endothelial (血管)", "CD56.NK", "CXCR4", "CD163", "CXCL12"]
            
            cols = st.columns(min(num_channels, 4))
            channel_mapping = {}
            
            for i in range(num_channels):
                col_idx = i % 4
                if i > 0 and col_idx == 0:
                    cols = st.columns(min(num_channels - i, 4))
                
                with cols[col_idx]:
                    thumb = img_matrix[i][::4, ::4] 
                    p2, p98 = np.percentile(thumb, (2, 98))
                    thumb_vis = np.clip((thumb - p2) / (p98 - p2 + 1e-5), 0, 1)
                    
                    fig, ax = plt.subplots(figsize=(2, 2))
                    ax.imshow(thumb_vis, cmap='magma')
                    ax.axis('off')
                    st.pyplot(fig)
                    
                    selected_marker = st.selectbox(f"通道 {i+1} 标志物", marker_options, key=f"ch_{i}")
                    channel_mapping[i] = selected_marker.split(" ")[0]

            st.markdown("---")
            if st.button("🚀 开始空间组学分析与 MVI 预测", use_container_width=True, key="btn_tiff"):
                mapped_values = list(channel_mapping.values())
                if 'DAPI' not in mapped_values:
                    st.error("❌ 必须指定一个通道作为 'DAPI' 用于细胞核空间定位！")
                elif not os.path.exists('mvi_scoring_model.pkl'):
                    st.error("❌ 找不到 AI 模型 mvi_scoring_model.pkl！")
                else:
                    with st.status("🧠 mviTIMES 引擎正在全速运转...", expanded=True) as status:
                        active_channels = {marker: img_matrix[idx] for idx, marker in channel_mapping.items() if marker != "忽略该通道"}
                                
                        status.update(label="正在进行细胞核分水岭分割...", state="running")
                        labels = segment_cells_dapi(active_channels['DAPI'])
                        
                        status.update(label="正在进行荧光共定位鉴定...", state="running")
                        df_spatial = quantify_and_phenotype(labels, active_channels, user_img_id, user_parent_id)
                        
                        df_spatial['x.axis'] = df_spatial['x.axis'] * scale_factor
                        df_spatial['y.axis'] = df_spatial['y.axis'] * scale_factor
                        df_valid = df_spatial[df_spatial['CellType'] != 'Other']
                        
                        status.update(label="正在提取拓扑特征并计算预测分...", state="running")
                        predictor = MVIPredictor()
                        predictor.load_model('mvi_scoring_model.pkl')
                        results_df = predictor.predict_score(df_valid)
                        
                        status.update(label="分析完成！", state="complete")
                    
                    st.balloons()
                    st.subheader(f"📊 {user_img_id} 预测结果")
                    
                    r_col1, r_col2 = st.columns([2, 1])
                    with r_col1:
                        st.write(f"**提取的空间单细胞坐标 (共提取特征细胞 {len(df_valid)} 个)**")
                        st.dataframe(df_valid.head(100), height=250)
                        
                        # 提供自动生成的坐标数据集下载，方便用户留存
                        extracted_csv = df_valid.to_csv(index=False).encode('utf-8')
                        st.download_button("⬇️ 下载该图像提取的单细胞坐标数据 (CSV)", data=extracted_csv, file_name=f'{user_img_id}_cells.csv', mime='text/csv')

                    with r_col2:
                        st.write("**最终 mviTIMES 空间免疫预测分**")
                        score = results_df['mviTIMES_Score'].iloc[0]
                        st.metric(label="MVI 阳性概率", value=f"{score*100:.1f}%")
                        if score > 0.5:
                            st.error("🚨 高风险：呈现免疫抑制/侵犯特征")
                        else:
                            st.success("✅ 低风险")

        except Exception as e:
            st.error(f"处理图像时发生错误：{str(e)}")
