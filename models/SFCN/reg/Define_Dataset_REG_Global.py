import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib

# %% 定義dataset

class MRIDataset_REG(Dataset):
    def __init__(self, data_folders_set, labels_df, augment = False, crop_coords = None):
        """
        Args:
            data_folders (list): MRI 資料夾的路徑列表
            labels_df (pd.DataFrame): 包含 [Filename, Age] 的 DataFrame
            bin_range (tuple): 年齡 bin 的範圍，例如 (20, 90)
            bin_step (int): bin 的間隔，例如 2
            sigma (float): 與論文一樣設為1
        """
        self.data_folders_set = data_folders_set
        self.labels_df = labels_df
        self.augment = augment
        self.file_paths = []  # 建立完整檔案清單，包含檔名和完整路徑
        self.crop_coords = crop_coords 
        self.ages = []

        # row會列出（Filename HCA9992517.nii、Age 54.0、Age Bin 45~55）
        for _, row in labels_df.iterrows():
            filename = row["Filename"]       # 取出df內的Filename欄
            age = float(row["Age"])          # 取出df內的Age欄
            for folder in data_folders_set:  # 從data_folders_set中找個資料集檔案路徑
                full_path = os.path.join(folder, filename)  # 找指定檔案
                if os.path.exists(full_path):
                    self.file_paths.append(full_path)
                    self.ages.append(age)
                    break
            else:
                raise FileNotFoundError(f"❌ 找不到 {filename} 在指定資料夾中")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image_path = self.file_paths[idx]
        image = nib.load(image_path).get_fdata().astype('float32')
        
        # ======= 新增裁切 =======
        if self.crop_coords is not None:
            min_coords, max_coords = self.crop_coords
            image = image[
                min_coords[0]:max_coords[0]+1,
                min_coords[1]:max_coords[1]+1,
                min_coords[2]:max_coords[2]+1
            ]
            
        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W, D)

        # ======= Data Augmentation =======
        if self.augment:
            # 1. 隨機平移 (沿每個軸 0, 1, 2 個體素)
            shift_vals = [random.choice([0, 1, 2]) for _ in range(3)]
            image = torch.roll(image, shifts=shift_vals, dims=(1, 2, 3))

            # 2. 50% 機率沿矢狀面 (x軸) 翻轉
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])  # x 軸翻轉

        age_label = self.ages[idx]
        return image, age_label


def sample_balanced_epoch(df, selected_counts, sample_per_decade=274):
    """
    根據 decade 抽樣，保證每個樣本至少被抽一次。
    5~75 歲 decade：先抽未抽過樣本，再補缺額，補缺額時選擇被抽次數最少的樣本
    75~85、85~100：全部取
    """
    sampled_list = []

    # 5~75 歲 decade
    for start in range(5, 75, 10):
        end = start + 10
        subset = df[(df['Age'] >= start) & (df['Age'] < end)].copy()
        if subset.empty:
            continue

        # 記錄每筆 sample 被抽過的次數
        subset['select_count'] = subset['Filename'].map(selected_counts).fillna(0)

        # 剩餘未抽樣的樣本（select_count = 0）
        remaining_unselected = subset[subset['select_count'] == 0].copy()

        # 先取未抽過的
        chosen = remaining_unselected.copy()

        # 計算還需要多少樣本補足 sample_per_decade
        n_needed = sample_per_decade - len(chosen)

        if n_needed > 0:
            # 從已抽過樣本中選 select_count 最少的補足
            already_selected = subset[subset['select_count'] > 0].copy()
            already_selected = already_selected.sort_values('select_count', ascending=True)
            chosen2 = already_selected.head(n_needed)
            chosen = pd.concat([chosen, chosen2], ignore_index=True)

        # 如果剩餘未抽樣數量已經超過 sample_per_decade，隨機抽取 sample_per_decade
        if len(chosen) > sample_per_decade:
            chosen = chosen.sample(n=sample_per_decade, random_state=None)

        sampled_list.append(chosen)

    # 75~85 歲全部加入
    subset_75_85 = df[(df['Age'] >= 75) & (df['Age'] < 85)].copy()
    if not subset_75_85.empty:
        sampled_list.append(subset_75_85)

    # 85~100 歲全部加入
    subset_85_100 = df[(df['Age'] >= 85) & (df['Age'] <= 100)].copy()
    if not subset_85_100.empty:
        sampled_list.append(subset_85_100)

    # 合併
    epoch_df = pd.concat(sampled_list, ignore_index=True)

    # 更新 selected_counts
    for filename, count in epoch_df['Filename'].value_counts().items():
        selected_counts[filename] = selected_counts.get(filename, 0) + count

    return epoch_df, selected_counts

# %%
'''
from Found_Data_SM_Global import Decide_Dataset, Classify_Data_to_agebin
from Split_Data import split_data

datasets = Decide_Dataset(IXI_tnf=True, OASIS_tnf=True, ABIDE1_tnf=True, ABIDE2_tnf=True,
                   FCP_tnf=True, HCA_tnf=True, NKI_RS_tnf=True)
result_df, grouped_files_df = Classify_Data_to_agebin(datasets)

train_df, val_df, test_df = split_data(grouped_files_df)

# 初始化抽樣次數紀錄
selected_counts = {}

# 每個 epoch 動態抽樣
sampled_train_df, selected_counts = sample_balanced_epoch(
    df=train_df,
    selected_counts=selected_counts,
    sample_per_decade=210,
    
)

print(sampled_train_df.head())
print(f"抽樣總數: {len(sampled_train_df)}")
'''
# %%
'''
from Reed_split_csv import reed_split_csv
train_df, val_df, test_df = reed_split_csv()

# ====== 2️⃣ 初始化 selected_counts ======
selected_counts = {}

# ====== 3️⃣ 模擬多個 epoch ======
epochs = 20
for i in range(epochs):
    print(f"\n===== Epoch {i+1} =====")
    sampled_train_df, selected_counts = sample_balanced_epoch(
        df=train_df,
        selected_counts=selected_counts,
        sample_per_decade=274
    )
    print(f"Epoch {i+1} 抽樣數量: {len(sampled_train_df)}")

# ====== 4️⃣ 將 selected_counts 轉成 DataFrame 方便檢視 ======
summary_df = pd.DataFrame.from_dict(selected_counts, orient='index', columns=['抽樣次數'])
summary_df.index.name = 'Filename'
summary_df = summary_df.sort_values(by='抽樣次數', ascending=False)

# ====== 5️⃣ 統計結果 ======
print("\n=== 抽樣統計結果 ===")
print(f"總樣本數: {len(summary_df)}")
print(f"抽樣總次數: {summary_df['抽樣次數'].sum()}")
print(f"平均抽樣次數: {summary_df['抽樣次數'].mean():.2f}")
print(f"最高抽樣次數: {summary_df['抽樣次數'].max()}")
print(f"最低抽樣次數: {summary_df['抽樣次數'].min()}")

# ====== 6️⃣ 查看抽樣次數最多與最少的樣本 ======
print("\n抽樣次數最多的 10 筆:")
print(summary_df.head(10))

print("\n抽樣次數最少的 10 筆:")
print(summary_df.tail(10))
# ====== 7️⃣ 輸出最後一個 epoch 的前 5 筆樣本與累積 select_count ======
# 將最後一個 epoch 的 sampled_train_df 補上 select_count
sampled_train_df['select_count'] = sampled_train_df['Filename'].map(selected_counts)

print("\n最後一個 epoch 的前 5 筆樣本:")
print(sampled_train_df.head())

print("\n抽樣總數:")
print(len(sampled_train_df))
'''

# %%
'''
# 將 selected_counts 轉 DataFrame
summary_df = pd.DataFrame.from_dict(selected_counts, orient='index', columns=['抽樣次數'])
summary_df.index.name = 'Filename'
summary_df = summary_df.sort_values(by='抽樣次數', ascending=False)

# 印前 50 筆抽樣次數最多的樣本
print("抽樣次數最多的 50 筆:")
print(summary_df.head(50))

# 印前 50 筆抽樣次數最少的樣本
print("\n抽樣次數最少的 50 筆:")
print(summary_df.tail(50))

# 可以印 histogram 直觀看分布
import matplotlib.pyplot as plt
plt.hist(summary_df['抽樣次數'], bins=range(summary_df['抽樣次數'].min(), summary_df['抽樣次數'].max()+2))
plt.xlabel("抽樣次數")
plt.ylabel("樣本數")
plt.title("每個樣本累積抽樣次數分布")
plt.show()
'''
# %%
'''
from Check_Function_Plot import soft_label_age_stats, plot_label_distribution
from Found_Data import get_all_global_ndf_img_routes
all_img_routes = get_all_global_ndf_img_routes()
dataset = MRIDataset_SM(
        data_folders_set = all_img_routes,
        labels_df = grouped_files_df,  # 之前產生的 DataFrame
        bin_range = (5, 100),
)

# 你的 dataset
image, label = dataset[1250]
label, _ = num2vect(label, bin_range=(5, 100), bin_step = 1, sigma = 1)
label = torch.tensor(label, dtype=torch.float32)
# 計算正確的（年齡軸）平均與標準差
mean_age, std_age, std_bins = soft_label_age_stats(label, bin_range=(15, 100), bin_step=1)
print(f"mean_age={mean_age:.2f}, std_age={std_age:.2f} (≈ {std_bins:.2f} bins)")
print(label)
# 畫圖 + 顯示正確的 std
plot_label_distribution(label, bin_range=(15, 100), bin_step=1)
'''