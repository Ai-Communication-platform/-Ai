import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = 'C:\\Users\\win\\Documents\\GitHub\\-Ai\\감성대화말뭉치(최종데이터)_Training.xlsx'  # Replace with your file path
data = pd.read_excel(file_path)

# Prepare data for visualization
emotion_major_count = data['감정_대분류'].value_counts()
emotion_minor_count = data['감정_소분류'].value_counts()
age_group_count = data['연령'].value_counts()
gender_count = data['성별'].value_counts()

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plotting Emotion Major Category Distribution
sns.barplot(x=emotion_major_count.index, y=emotion_major_count.values, ax=axes[0, 0])
axes[0, 0].set_title('Emotion Major Category Distribution')
axes[0, 0].set_xlabel('Emotion Major Category')
axes[0, 0].set_ylabel('Count')

# Plotting Emotion Minor Category Distribution
sns.barplot(x=emotion_minor_count.index, y=emotion_minor_count.values, ax=axes[0, 1])
axes[0, 1].set_title('Emotion Minor Category Distribution')
axes[0, 1].set_xlabel('Emotion Minor Category')
axes[0, 1].set_ylabel('Count')
axes[0, 1].tick_params(axis='x', rotation=90)

# Plotting Age Group Distribution
sns.barplot(x=age_group_count.index, y=age_group_count.values, ax=axes[1, 0])
axes[1, 0].set_title('Age Group Distribution')
axes[1, 0].set_xlabel('Age Group')
axes[1, 0].set_ylabel('Count')

# Plotting Gender Distribution
sns.barplot(x=gender_count.index, y=gender_count.values, ax=axes[1, 1])
axes[1, 1].set_title('Gender Distribution')
axes[1, 1].set_xlabel('Gender')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()