import pandas as pd
import numpy as np
import joblib
import os
from scipy.spatial import KDTree
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class MVIPredictor:
    def __init__(self):
        self.pipeline = None
        self.feature_columns = None
        
    def _extract_region_features(self, region_df):
        features = {}
        x_range = region_df['x.axis'].max() - region_df['x.axis'].min()
        y_range = region_df['y.axis'].max() - region_df['y.axis'].min()
        area = x_range * y_range if (x_range * y_range) > 0 else 1.0
            
        cell_counts = region_df['CellType'].value_counts()
        for ct in ['CD34.Endothelial', 'GPC3.Tumor', 'CD56.NK', 'CXCR4.CD56.NK', 'CXCL12.CD163.M2']:
            features[f'Density_{ct}'] = cell_counts.get(ct, 0) / area
        
        coords = {ct: region_df[region_df['CellType'] == ct][['x.axis', 'y.axis']].values 
                  for ct in region_df['CellType'].unique()}
        
        def get_interaction_stats(source, target, prefix):
            if len(source) > 0 and len(target) > 0:
                tree = KDTree(target)
                dists, _ = tree.query(source, k=1)
                features[f'{prefix}_Mean_NND'] = np.mean(dists)
                features[f'{prefix}_Interaction_Score_50'] = np.mean(dists <= 50)
                features[f'{prefix}_Interaction_Score_100'] = np.mean(dists <= 100)
            else:
                features[f'{prefix}_Mean_NND'] = 1000.0
                features[f'{prefix}_Interaction_Score_50'] = 0.0
                features[f'{prefix}_Interaction_Score_100'] = 0.0

        get_interaction_stats(coords.get('CXCR4.CD56.NK', []), coords.get('CXCL12.CD163.M2', []), 'NK_to_M2')
        get_interaction_stats(coords.get('CXCL12.CD163.M2', []), coords.get('CXCR4.CD56.NK', []), 'M2_to_NK')
        get_interaction_stats(coords.get('CXCR4.CD56.NK', []), coords.get('GPC3.Tumor', []), 'NK_to_Tumor')
        get_interaction_stats(coords.get('CXCL12.CD163.M2', []), coords.get('GPC3.Tumor', []), 'M2_to_Tumor')

        return features

    def process_data(self, df):
        all_features, image_names, labels = [], [], []
        for image_name, img_df in df.groupby('Image'):
            img_features = []
            for parent_name, parent_df in img_df.groupby('Parent'):
                if len(parent_df) < 10: continue
                img_features.append(self._extract_region_features(parent_df))
                
            if len(img_features) > 0:
                avg_features = pd.DataFrame(img_features).mean().to_dict()
                all_features.append(avg_features)
                image_names.append(image_name)
                
                # 安全获取标签
                if 'MVI.State' in img_df.columns:
                    labels.append(1 if img_df['MVI.State'].iloc[0] == 'MVI.pos' else 0)
                else:
                    labels.append(None)
                    
        X_df = pd.DataFrame(all_features, index=image_names).fillna(0)
        return X_df, labels

    # 这个就是刚才不小心漏掉的核心训练与保存函数
    def train_and_save(self, train_csv_path, model_save_path='mvi_scoring_model.pkl'):
        print(">>> 正在加载训练数据并提取空间特征...")
        df_train = pd.read_csv(train_csv_path)
        X_train, y_train = self.process_data(df_train)
        
        self.feature_columns = X_train.columns.tolist()
        
        print(">>> 正在训练核心 SVM 管道...")
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_classif, k=5)),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ])
        self.pipeline.fit(X_train, y_train)
        
        joblib.dump({'pipeline': self.pipeline, 'features': self.feature_columns}, model_save_path)
        print(f"✅ 模型训练完毕，已持久化保存至: {model_save_path}\n")

    def load_model(self, model_path='mvi_scoring_model.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件 {model_path}。")
        data = joblib.load(model_path)
        self.pipeline = data['pipeline']
        self.feature_columns = data['features']

    def predict_score(self, df_new):
        if self.pipeline is None:
            raise ValueError("模型尚未加载！")
            
        X_new, _ = self.process_data(df_new)
        X_new = X_new.reindex(columns=self.feature_columns, fill_value=0)
        
        mvi_pos_scores = self.pipeline.predict_proba(X_new)[:, 1]
        results_df = pd.DataFrame({
            'Image_ID': X_new.index,
            'mviTIMES_Score': mvi_pos_scores
        }).sort_values(by='mviTIMES_Score', ascending=False).reset_index(drop=True)
        return results_df