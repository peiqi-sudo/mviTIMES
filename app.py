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

st.title("🔬 mviTIMES: 多重免疫荧光 (mIF) 空间智能分析平台")
st.markdown("上传多通道 `.tiff` 图像，在线完成**细胞分割、共定位分型与微血管侵犯(MVI)风险预测**。")

# ==========================================
# 核心图像处理函数
# ==========================================
@st.cache_data(show_spinner=False)
def load_tiff_image(file_bytes):
    """读取多通道 TIFF 并标准化维度为 (Channels, Height, Width)"""
    img = tifffile.imread(file_bytes)
    # 智能维度判定：如果第三个维度很小，说明是 (H, W, C)，转为 (C, H, W)
    if img.ndim == 3 and img.shape[2] < min(img.shape[0], img.shape[1]) and img.shape[2] <= 10:
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:
        img = img[np.newaxis, :, :] # 单通道补齐
    return img

def segment_cells_dapi(dapi_channel):
    """基于 DAPI 通道进行分水岭细胞核分割"""
    # 1. 高斯平滑去噪
    blurred = filters.gaussian(dapi_channel, sigma=1.5)
    # 2. Otsu 自适应阈值二值化
    thresh = filters.threshold_otsu(blurred)
    binary = blurred > thresh
    # 3. 移除微小噪点，填充孔洞
    binary = morphology.remove_small_objects(binary, min_size=20)
    binary = ndimage.binary_fill_holes(binary)
    # 4. 距离变换与局部最大值寻找（找细胞核中心）
    distance = ndimage.distance_transform_edt(binary)
    coords = feature.peak_local_max(distance, footprint=np.ones((5, 5)), labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    # 5. 分水岭算法切割粘连细胞
    labels = segmentation.watershed(-distance, markers, mask=binary)
    return labels

def quantify_and_phenotype(labels, channels_dict, img_id, parent_id):
    """提取各通道表达量并进行 Boolean 细胞分型"""
    props = measure.regionprops(labels)
    data = []
    
    # 获取各个通道的图像矩阵
    img_GPC3 = channels_dict.get('GPC3.Tumor')
    img_CD34 = channels_dict.get('CD34.Endothelial')
    img_CD56 = channels_dict.get('CD56.NK')
    img_CXCR4 = channels_dict.get('CXCR4')
    img_CD163 = channels_dict.get('CD163')
    img_CXCL12 = channels_dict.get('CXCL12')

    # 计算全局阳性阈值 (Otsu) 以加速单细胞判定
    thresholds = {}
    for name, img in channels_dict.items():
        if img is not None and name != 'DAPI':
            try:
                thresholds[name] = filters.threshold_otsu(img[img > 0]) # 仅对有背景以上的区域算阈值
            except:
                thresholds[name] = img.mean()

    for prop in props:
        centroid_y, centroid_x = prop.centroid
        coords = prop.coords
        
        # 定义一个内部小函数，快速判断该细胞在某通道是否阳性
        def is_positive(img_matrix, marker_name):
            if img_matrix is None: return False
            # 提取该细胞 mask 覆盖区域的平均荧光强度
            mean_intensity = np.mean(img_matrix[coords[:, 0], coords[:, 1]])
            return mean_intensity > thresholds.get(marker_name, 0)

        # Boolean 逻辑共定位分型
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
            'x.axis': centroid_x, # 注意 x 是列坐标
            'y.axis': centroid_y, # 注意 y 是行坐标
            'CellType': cell_type
        })
        
    return pd.DataFrame(data)

# ==========================================
# UI 前端与业务逻辑
# ==========================================
col_setup, col_upload = st.columns([1, 2])
with col_setup:
    st.subheader("⚙️ 步骤 1：基础信息设置")
    user_img_id = st.text_input("图像样本编号 (Image ID)", "Sample_001")
    user_parent_id = st.text_input("组织区域编号 (Parent ID)", "Region_A")
    # 比例尺输入，保证空间距离计算物理意义正确
    scale_factor = st.number_input("比例尺 (1像素 = 多少微米 μm)", value=0.5, step=0.1)

with col_upload:
    st.subheader("📂 步骤 2：上传多通道 TIFF 原图")
    uploaded_file = st.file_uploader("支持 .tiff, .qptiff 格式（建议单张大小不超过 50MB）", type=["tif", "tiff", "qptiff"])

if uploaded_file is not None:
    try:
        with st.spinner("正在解析 TIFF 高维矩阵..."):
            img_matrix = load_tiff_image(uploaded_file)
            num_channels = img_matrix.shape[0]
        
        st.success(f"成功读取图像！检测到 **{num_channels}** 个通道。分辨率: {img_matrix.shape[2]}x{img_matrix.shape[1]}")
        st.markdown("---")
        
        st.subheader("👁️ 步骤 3：通道映射 (Visual Channel Mapping)")
        st.info("请根据下方各通道的形态缩略图，在下拉菜单中为其指定对应的标志物身份。")
        
        # 预设标志物列表
        marker_options = ["忽略该通道 (Ignore)", "DAPI (细胞核)", "GPC3.Tumor (肿瘤)", 
                          "CD34.Endothelial (血管)", "CD56.NK", "CXCR4", "CD163", "CXCL12"]
        
        # 动态生成列来展示通道
        cols = st.columns(min(num_channels, 4))
        channel_mapping = {}
        
        for i in range(num_channels):
            col_idx = i % 4
            if i > 0 and col_idx == 0:
                cols = st.columns(min(num_channels - i, 4))
            
            with cols[col_idx]:
                # 降采样生成缩略图以加快显示
                thumb = img_matrix[i][::4, ::4] 
                # 标准化对比度使其可见
                p2, p98 = np.percentile(thumb, (2, 98))
                thumb_vis = np.clip((thumb - p2) / (p98 - p2 + 1e-5), 0, 1)
                
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.imshow(thumb_vis, cmap='magma')
                ax.axis('off')
                st.pyplot(fig)
                
                # 用户选择下拉框
                selected_marker = st.selectbox(f"通道 {i+1} 标志物", marker_options, key=f"ch_{i}")
                channel_mapping[i] = selected_marker.split(" ")[0] # 提取英文键值

        st.markdown("---")
        if st.button("🚀 步骤 4：开始空间组学分析与 MVI 预测", use_container_width=True):
            
            # 检查必须通道是否存在
            mapped_values = list(channel_mapping.values())
            if 'DAPI' not in mapped_values:
                st.error("❌ 必须指定一个通道作为 'DAPI' 用于细胞核空间定位！")
            elif not os.path.exists('mvi_scoring_model.pkl'):
                st.error("❌ 找不到 AI 模型 mvi_scoring_model.pkl！")
            else:
                with st.status("🧠 mviTIMES 引擎正在全速运转...", expanded=True) as status:
                    
                    # 1. 构建命名通道字典
                    active_channels = {}
                    for idx, marker in channel_mapping.items():
                        if marker != "忽略该通道":
                            active_channels[marker] = img_matrix[idx]
                            
                    status.update(label="正在进行细胞核分水岭分割 (Watershed Segmentation)...", state="running")
                    labels = segment_cells_dapi(active_channels['DAPI'])
                    cell_count = np.max(labels)
                    
                    status.update(label=f"成功识别 {cell_count} 个细胞。正在进行荧光共定位鉴定...", state="running")
                    # 传入字典、用户设置的ImageID和ParentID
                    df_spatial = quantify_and_phenotype(labels, active_channels, user_img_id, user_parent_id)
                    
                    # 比例尺校准
                    df_spatial['x.axis'] = df_spatial['x.axis'] * scale_factor
                    df_spatial['y.axis'] = df_spatial['y.axis'] * scale_factor
                    
                    # 过滤掉 Other，并展示转化后的 DataFrame
                    df_valid = df_spatial[df_spatial['CellType'] != 'Other']
                    
                    status.update(label="正在提取拓扑特征并计算 MVI 风险预测...", state="running")
                    predictor = MVIPredictor()
                    predictor.load_model('mvi_scoring_model.pkl')
                    results_df = predictor.predict_score(df_valid)
                    
                    status.update(label="分析完成！", state="complete")
                
                st.balloons()
                st.subheader(f"📊 {user_img_id} 预测结果")
                
                # 左右分栏展示：左边展示转化出的数据集，右边展示最终得分
                r_col1, r_col2 = st.columns([2, 1])
                with r_col1:
                    st.write(f"**提取的空间单细胞坐标 (前100行, 共识别特征细胞 {len(df_valid)} 个)**")
                    st.dataframe(df_valid.head(100), height=250)
                with r_col2:
                    st.write("**最终 mviTIMES 空间免疫预测分**")
                    score = results_df['mviTIMES_Score'].iloc[0]
                    # 用类似仪表盘的巨大数字展示
                    st.metric(label="MVI 阳性概率", value=f"{score*100:.1f}%")
                    if score > 0.5:
                        st.error("🚨 高风险：空间拓扑呈现强烈的免疫抑制/侵犯特征")
                    else:
                        st.success("✅ 低风险")

    except Exception as e:
        st.error(f"处理图像时发生错误：{str(e)}")
