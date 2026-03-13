import streamlit as st
import pandas as pd
import numpy as np
import os
import tifffile
from skimage import filters, measure, morphology, segmentation, feature
from scipy import ndimage
import matplotlib.pyplot as plt
from mvi_predictor import MVIPredictor
from datetime import datetime

# ==========================================
# 0. 全局页面配置 & 旗舰级 CSS 注入
# ==========================================
st.set_page_config(page_title="mviTIMES Spatial AI", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

# 深度 CSS 美化：渐变字体、悬浮卡片、精致分隔线
st.markdown("""
    <style>
    /* 渐变主标题 */
    .hero-title {
        font-size: 3.5rem; 
        font-weight: 900; 
        background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6, #06B6D4); 
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent; 
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.25rem; 
        color: #6B7280; 
        font-weight: 500; 
        margin-bottom: 2.5rem;
    }
    /* 悬浮信息卡片 */
    .feature-card {
        padding: 1.5rem; 
        border-radius: 12px; 
        background: #F8FAFC; 
        border: 1px solid #E2E8F0; 
        transition: all 0.3s ease;
        height: 100%;
    }
    .feature-card:hover {
        transform: translateY(-5px); 
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        border-color: #3B82F6;
    }
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E40AF;
        margin-bottom: 0.75rem;
    }
    /* 流程步骤框 */
    .step-box {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(to right, #EFF6FF, #DBEAFE);
        border-radius: 8px;
        color: #1E3A8A;
        font-weight: 600;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    hr {margin-top: 2rem; margin-bottom: 2rem; border-color: #E5E7EB;}
    </style>
    """, unsafe_allow_html=True)

# --- 状态路由管理 (彻底修复按钮跳转) ---
if "nav_menu" not in st.session_state:
    st.session_state.nav_menu = "🏠 首页 (Home)"

def switch_to_workspace():
    st.session_state.nav_menu = "🚀 在线分析 (Workspace)"

# ==========================================
# 1. 核心处理函数区 (保持底层的硬核算法不变)
# ==========================================
def load_csv_safely(file):
    encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']
    for enc in encodings_to_try:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法解析文件编码，请上传 CSV (UTF-8) 格式。")

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
    img_GPC3, img_CD34, img_CD56, img_CXCR4, img_CD163, img_CXCL12 = (
        channels_dict.get('GPC3.Tumor'), channels_dict.get('CD34.Endothelial'), 
        channels_dict.get('CD56.NK'), channels_dict.get('CXCR4'), 
        channels_dict.get('CD163'), channels_dict.get('CXCL12')
    )

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
            'Image': img_id, 'Parent': parent_id,
            'x.axis': centroid_x, 'y.axis': centroid_y, 'CellType': cell_type
        })
    return pd.DataFrame(data)

# ==========================================
# 2. 侧边栏导航 (Sidebar Navigation)
# ==========================================
with st.sidebar:
    st.image("logo.png", width=200) 
    st.markdown("## mviTIMES")
    st.caption("Spatial Immune Microenvironment AI")
    st.divider()
    
    # 绑定 session_state 中的 nav_menu
    nav_selection = st.radio(
        "NAVIGATION",
        ["🏠 首页 (Home)", "🚀 在线分析 (Workspace)", "📖 使用文档 (Docs)"],
        label_visibility="collapsed",
        key="nav_menu"
    )
    st.divider()
    st.markdown("### 🧬 核心系统支持")
    st.markdown("✅ **多模态影像解析引擎**")
    st.markdown("✅ **SVM 图网络拓扑学习**")
    st.markdown("✅ **高维单细胞空间定量**")
    st.divider()
    st.caption("© 2026 mviTIMES Bioinformatics Team")

# ==========================================
# 3. 页面一：首页 (Home) - 旗舰视觉展示
# ==========================================
if nav_selection == "🏠 首页 (Home)":
    st.markdown('<div class="hero-title">mviTIMES: 靶向微血管侵犯的空间免疫网络预测平台</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Decoding the Spatial Topography of the Tumor Immune Microenvironment</div>', unsafe_allow_html=True)
    
    # --- 工作流可视化 (Workflow) ---
    st.markdown("### 🔬 端到端 AI 分析管线 (Analysis Pipeline)")
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1: st.markdown("<div class='step-box'>① 影像上传<br><span style='font-size:0.8rem; font-weight:normal;'>支持高多重 TIFF 矩阵</span></div>", unsafe_allow_html=True)
    with w_col2: st.markdown("<div class='step-box'>② 智能分割<br><span style='font-size:0.8rem; font-weight:normal;'>分水岭提取单细胞</span></div>", unsafe_allow_html=True)
    with w_col3: st.markdown("<div class='step-box'>③ 拓扑重构<br><span style='font-size:0.8rem; font-weight:normal;'>计算 NND 与空间互作</span></div>", unsafe_allow_html=True)
    with w_col4: st.markdown("<div class='step-box'>④ 风险评估<br><span style='font-size:0.8rem; font-weight:normal;'>SVM 预测 MVI 状态</span></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # --- 核心特性卡片 ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">🖼️ 影像直达数据</div>
            直接摄取 mIF/mIHC 格式的原图。无需依赖昂贵的第三方商业软件，平台内置强大算法，瞬间将复杂的像素阵列转化为单细胞坐标图谱。
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">🕸️ 空间图网络拓扑</div>
            突破传统的“细胞计数”局限。利用 KDTree 算法在二维平面重构千万级像元的通信网络，精准捕捉巨噬细胞对 NK 细胞的物理隔离屏障。
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-title">🤖 机器学习预测引擎</div>
            底层集成预训练的径向基核支持向量机 (RBF-SVM)。结合大队列患者真实随访数据，为微血管侵犯 (MVI) 风险提供强鲁棒性的定量化评估。
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # --- 视频与背景区 ---
    st.markdown("### 🎬 系统操作演示与背景")
    row1_col1, row1_col2 = st.columns([1, 1])
    with row1_col1:
        st.video("tutorial.mp4") # 可替换为您自己的本地视频或 YouTube 链接
    with row1_col2:
        st.write("""
        微血管侵犯 (Microvascular Invasion, MVI) 是评估实体瘤预后的核心独立危险因素。传统的病理学评估往往忽略了肿瘤微环境 (TME) 中极其复杂的免疫空间异质性。
        
        **机制突破：** 当 `CXCL12+ CD163+ M2` 细胞在空间上紧密包围 `CXCR4+ CD56+ NK` 细胞时，会形成强烈的局部免疫抑制。本平台即致力于解码这一空间密码。
        """)
        st.info("💡 **一键式部署体验**：无论是计算机视觉分割、共定位表型鉴定，还是 AI 预测，全程在浏览器内闭环完成。")
        
    st.write("") # 留白
    # 完美跳转按钮
    st.button("👉 开启我的首次智能分析", type="primary", use_container_width=True, on_click=switch_to_workspace)

# ==========================================
# 4. 页面二：在线分析 (Workspace)
# ==========================================
elif nav_selection == "🚀 在线分析 (Workspace)":
    st.markdown('<p class="main-title" style="font-size:2.5rem;">🚀 空间免疫分析工作台</p>', unsafe_allow_html=True)
    st.write("请选择您的数据格式。平台已开启最高级别的隐私保护，您的影像与数据仅在当前会话的内存中处理，不会被云端持久化存储。")
    st.divider()
    
    tab_csv, tab_tiff = st.tabs(["📄 方式一：提取后坐标分析 (CSV)", "🌌 方式二：全栈图像解析与分析 (TIFF)"])

    # --- CSV 通道 ---
    with tab_csv:
        st.info("📌 **适用场景**：您已使用 Halo/QuPath 完成细胞分割与表型鉴定，仅需调用 AI 引擎打分。")
        uploaded_csv = st.file_uploader("📂 上传标准 CSV 文件 (含 Image, Parent, x.axis, y.axis, CellType)", type=["csv"], key="csv")
        
        if uploaded_csv is not None:
            try:
                df_input = load_csv_safely(uploaded_csv)
                if st.button("🚀 计算 mviTIMES 风险得分", use_container_width=True, key="btn_csv"):
                    if not os.path.exists('mvi_scoring_model.pkl'):
                        st.error("⚠️ 核心模型载入失败，请联系管理员。")
                    else:
                        with st.spinner('🧠 正在构建全图细胞通信网络并提取空间特征...'):
                            predictor = MVIPredictor()
                            predictor.load_model('mvi_scoring_model.pkl')
                            results_df = predictor.predict_score(df_input)
                        
                        st.success("✅ 网络计算完成！")
                        st.dataframe(results_df.style.background_gradient(cmap='OrRd', subset=['mviTIMES_Score']), use_container_width=True)
            except Exception as e:
                st.error(f"⚠️ 格式解析错误：{str(e)}")

    # --- TIFF 通道 ---
    with tab_tiff:
        st.info("📌 **适用场景**：一站式完成从多通道原图到临床风险评分的全链条测算。")
        
        st.markdown("#### ⚙️ 步骤 1：录入临床切片信息")
        col_setup1, col_setup2, col_setup3 = st.columns(3)
        with col_setup1: user_img_id = st.text_input("样本编号 (Sample ID)", "Patient_001")
        with col_setup2: user_parent_id = st.text_input("分析区域 (Region)", "Core_Tumor")
        with col_setup3: scale_factor = st.number_input("光学比例尺 (μm / px)", value=0.50, step=0.01)

        st.markdown("#### 📂 步骤 2：加载高维影像矩阵")
        uploaded_tiff = st.file_uploader("拖拽上传 .tiff / .qptiff 原图", type=["tif", "tiff", "qptiff"], key="tiff")

        if uploaded_tiff is not None:
            try:
                with st.spinner("正在解构图像张量..."):
                    img_matrix = load_tiff_image(uploaded_tiff)
                    num_channels = img_matrix.shape[0]
                
                st.success(f"✅ 成功解构 {num_channels} 层荧光通道矩阵。分辨率：{img_matrix.shape[1]}x{img_matrix.shape[2]}")
                
                st.markdown("#### 👁️ 步骤 3：高反差通道校验映射")
                st.caption("提示：请根据呈现的高反差白斑形态及导出顺序，为通道赋予明确的生物学意义。")
                
                marker_options = ["忽略该通道 (Ignore)", "DAPI (细胞核)", "GPC3.Tumor (肿瘤)", 
                                  "CD34.Endothelial (血管)", "CD56.NK", "CXCR4", "CD163", "CXCL12"]
                
                cols = st.columns(min(num_channels, 4))
                channel_mapping = {}
                
                for i in range(num_channels):
                    col_idx = i % 4
                    if i > 0 and col_idx == 0:
                        cols = st.columns(min(num_channels - i, 4))
                    
                    with cols[col_idx]:
                        st.markdown(f"<span style='color:#3B82F6; font-weight:bold;'>通道 {i+1}</span>", unsafe_allow_html=True) 
                        
                        thumb = img_matrix[i][::4, ::4] 
                        p_low, p_high = np.percentile(thumb, (5, 99.5)) 
                        if p_high == p_low: p_high = p_low + 1e-5
                        thumb_vis = np.clip((thumb - p_low) / (p_high - p_low), 0, 1)
                        
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(thumb_vis, cmap='gray') 
                        ax.axis('off')
                        st.pyplot(fig)
                        
                        selected_marker = st.selectbox(f"目标靶点", marker_options, key=f"ch_{i}", label_visibility="collapsed")
                        channel_mapping[i] = selected_marker.split(" ")[0]

                st.divider()
                if st.button("🚀 启动深度空间网络预测", use_container_width=True, type="primary"):
                    mapped_values = list(channel_mapping.values())
                    if 'DAPI' not in mapped_values:
                        st.error("❌ 严重异常：未能检测到 'DAPI' 通道，无法进行细胞核定位。")
                    else:
                        with st.status("🧠 mviTIMES 引擎全线启动...", expanded=True) as status:
                            active_channels = {marker: img_matrix[idx] for idx, marker in channel_mapping.items() if marker != "忽略该通道"}
                            
                            status.update(label="正在进行高斯滤波与分水岭细胞核分割...", state="running")
                            labels = segment_cells_dapi(active_channels['DAPI'])
                            
                            status.update(label="提取微环境荧光信号并鉴定共定位亚型...", state="running")
                            df_spatial = quantify_and_phenotype(labels, active_channels, user_img_id, user_parent_id)
                            
                            df_spatial['x.axis'] = df_spatial['x.axis'] * scale_factor
                            df_spatial['y.axis'] = df_spatial['y.axis'] * scale_factor
                            df_valid = df_spatial[df_spatial['CellType'] != 'Other']
                            
                            status.update(label="重构空间图谱，调用 RBF-SVM 进行风险预测...", state="running")
                            predictor = MVIPredictor()
                            predictor.load_model('mvi_scoring_model.pkl')
                            results_df = predictor.predict_score(df_valid)
                            status.update(label="解析与推理完成！", state="complete")
                        
                        # --- 精美的临床报告 UI ---
                        st.markdown("### 📋 临床辅助分析报告")
                        st.markdown(f"**分析日期:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | **样本:** {user_img_id} | **检出微环境特征细胞:** {len(df_valid)} 枚")
                        
                        r_col1, r_col2 = st.columns([2, 1])
                        with r_col1:
                            st.dataframe(df_valid.head(100), height=200)
                            extracted_csv = df_valid.to_csv(index=False).encode('utf-8')
                            st.download_button("⬇️ 导出完整单细胞组学矩阵 (CSV)", data=extracted_csv, file_name=f'{user_img_id}_cells.csv', mime='text/csv')

                        with r_col2:
                            score = results_df['mviTIMES_Score'].iloc[0]
                            st.metric(label="MVI 阳性倾向得分", value=f"{score*100:.2f}%", delta="高风险" if score > 0.5 else "低风险", delta_color="inverse" if score > 0.5 else "normal")
                            
                            if score > 0.5:
                                st.error("🚨 **系统提示：** 该样本空间拓扑呈现强烈的 M2 包裹免疫抑制特征，高度提示微血管侵犯潜在风险。")
                            else:
                                st.success("✅ **系统提示：** 该样本未检测出显著的空间免疫隔离网络，MVI 倾向性较低。")

            except Exception as e:
                st.error(f"矩阵运算异常：{str(e)}")

# ==========================================
# 5. 页面三：使用文档 (Documentation)
# ==========================================
elif nav_selection == "📖 使用文档 (Docs)":
    st.markdown('<p class="main-title" style="font-size:2.5rem;">📖 技术白皮书与使用指南</p>', unsafe_allow_html=True)
    st.write("深入了解 mviTIMES 的图网络数学基础与参数设定指南。")
    st.divider()
    
    col_doc1, col_doc2 = st.columns(2)
    with col_doc1:
        st.markdown("### 🎯 空间靶标 Boolean 鉴定树")
        st.write("系统底层采用局部自适应 Otsu 阈值算法，在剥离背景噪音后，遵循以下逻辑执行单细胞群的精准鉴定：")
        doc_df = pd.DataFrame({
            "鉴定细胞类群 (CellType)": ["CXCR4.CD56.NK", "CXCL12.CD163.M2", "GPC3.Tumor", "CD34.Endothelial"],
            "主要标志物 (Condition A)": ["CXCR4 (+)", "CXCL12 (+)", "GPC3 (+)", "CD34 (+)"],
            "次要标志物 (Condition B)": ["CD56 (+)", "CD163 (+)", "N/A", "N/A"]
        })
        st.table(doc_df)
        
    with col_doc2:
        st.markdown("### 📐 图网络高维特征 (Graph Features)")
        st.write("支持向量机 (SVM) 在超平面中所仰赖的核心图网络特征包括：")
        st.info("📌 **NND (Nearest Neighbor Distance)**: 例如 `M2_to_Tumor_Mean_NND`，反映巨噬细胞对肿瘤的微观物理贴近程度。")
        st.info("📌 **Interaction Density (r=50μm)**: 刻画以特定细胞为中心，固定物理半径内特定亚群的局部浸润密度。")

    st.divider()
    st.markdown("### ❓ 常见问题 (FAQ)")
    with st.expander("Q1: 为什么强烈建议输入正确的物理比例尺 (μm/px)？"):
        st.write("图网络算法中定义的空间接触半径（如 50 微米）完全依赖于该比例尺的精确换算。如果比例尺错误，模型提取的空间拓扑特征将发生严重畸变，导致预测评分失效。")
    with st.expander("Q2: 细胞核分割不准确怎么办？"):
        st.write("本算法高度依赖 DAPI 图像的清晰度。如果您上传的 TIFF 中，DAPI 通道过曝、存在大面积组织折叠发光或离焦模糊，分水岭算法可能会产生过多碎片。建议使用图像预处理软件剔除劣质视野后再上传。")
