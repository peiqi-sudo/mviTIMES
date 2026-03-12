# 文件名：app.py
import streamlit as st
import pandas as pd
import os
from mvi_predictor import MVIPredictor

# 1. 网页全局设置 (页面标题，宽屏模式等)
st.set_page_config(page_title="mviTIMES Database", page_icon="🧬", layout="centered")

# 2. 网站抬头与介绍
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=60) # 可替换为您的机构Logo
st.title("🔬 mviTIMES: MVI 空间免疫预测平台")
st.markdown("""
**欢迎来到 mviTIMES 预测数据库！**
本平台利用高维空间互作算法 (Spatial Interaction Algorithm) 深度提取单细胞数据中巨噬细胞(M2)与自然杀伤细胞(NK)的空间拓扑规律，
为肝癌微血管侵犯 (Microvascular Invasion, MVI) 提供精准的阳性风险预测。

**请在下方上传您的 mIHC/mIF 原始坐标数据 (CSV格式)。**
""")

# 3. 侧边栏：操作说明
with st.sidebar:
    st.header("📋 数据格式说明")
    st.write("上传的 CSV 数据必须包含以下 5 列（区分大小写）：")
    st.code("- Image (样本ID)\n- Parent (区域ID)\n- x.axis (X坐标)\n- y.axis (Y坐标)\n- CellType (细胞类型)")
    st.write("---")
    st.write("💡 **关于 mviTIMES Score**:\n得分介于 0~1 之间。得分越高，代表该样本的微环境空间特征越倾向于发生 MVI。")

# 4. 核心功能：文件上传组件
uploaded_file = st.file_uploader("📂 点击这里上传您的 .csv 数据集", type=["csv"])

if uploaded_file is not None:
    try:
        # 读取用户上传的数据
        # --- 增强版：自动识别并读取多编码格式的 CSV 数据 ---
        def load_csv_safely(file):
            encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'utf-8-sig']
            for enc in encodings_to_try:
                try:
                    file.seek(0) # 每次尝试前，将文件指针拨回开头
                    return pd.read_csv(file, encoding=enc)
                except UnicodeDecodeError:
                    continue
            raise ValueError("无法解析文件编码，请在Excel中将文件‘另存为’ -> ‘CSV (UTF-8)’ 格式后再上传。")
            
        df = load_csv_safely(uploaded_file)
        # ---------------------------------------------------
        
        # 数据合法性校验
        required_cols = {'Image', 'Parent', 'x.axis', 'y.axis', 'CellType'}
        if not required_cols.issubset(set(df.columns)):
            st.error(f"❌ 数据格式错误！缺少必要的列。您的列名：{list(df.columns)}")
        else:
            st.success("✅ 数据加载成功！")
            with st.expander("点击查看数据预览前 5 行"):
                st.dataframe(df.head())
            
            st.markdown("---")
            # 5. 触发预测按钮
            if st.button("🚀 立即计算 mviTIMES 分数", use_container_width=True):
                
                # 检查模型是否存在
                if not os.path.exists('mvi_scoring_model.pkl'):
                    st.error("⚠️ 服务器未找到预训练模型文件 mvi_scoring_model.pkl，请联系管理员部署。")
                else:
                    # 进度提示组件
                    with st.spinner('🧠 AI 正在提取百万级空间互作特征，请耐心等待（通常需几十秒）...'):
                        
                        # 实例化我们写的预测类并加载大脑
                        predictor = MVIPredictor()
                        predictor.load_model('mvi_scoring_model.pkl')
                        
                        # 执行核心计算并获取结果
                        results_df = predictor.predict_score(df)
                        
                    st.balloons() # 庆祝动画
                    st.subheader("📊 预测结果 (mviTIMES Scores)")
                    
                    # 在网页上优雅地展示数据表 (可排序、可过滤)
                    st.dataframe(
                        results_df.style.background_gradient(cmap='OrRd', subset=['mviTIMES_Score']), 
                        use_container_width=True
                    )
                    
                    # 6. 提供结果下载按钮
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ 下载完整预测结果 (CSV)",
                        data=csv_data,
                        file_name='mviTIMES_Predictions.csv',
                        mime='text/csv'
                    )
                    
    except Exception as e:
        st.error(f"⚠️ 处理数据时发生未知错误：{str(e)}")