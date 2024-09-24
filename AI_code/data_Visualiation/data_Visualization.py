import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl

# Update matplotlib's settings to use a font that supports Korean
mpl.rcParams['font.family'] = 'NanumGothic'

print("Matplotlib Version:", mpl.__version__)
print("Matplotlib Installation Directory:", mpl.__file__)
print("Matplotlib Configuration Directory:", mpl.get_configdir())
print("Matplotlib Cache Directory:", mpl.get_cachedir())

# Load the dataset
file_path = 'C:\\Users\\win\Documents\\GitHub\\-Ai\\감성대화말뭉치(최종데이터)_Training.csv'
data = pd.read_csv(file_path, encoding='utf-8')
# Using dtypes to get the data type of each column
print("Data types of each column:")
print(data.dtypes)

# Using info() to get a summary including data types
print("\nDataFrame Summary:")
data.info()

# Extracting the relevant columns
emotion_major = data['감정_대분류']
emotion_subcategory = data['감정_소분류']

# Creating a dictionary to map major emotion categories to their subcategories
emotion_tree = defaultdict(set)
for major, sub in zip(emotion_major, emotion_subcategory):
    emotion_tree[major].add(sub)

# Creating a network graph from the tree structure
G = nx.DiGraph()
for major in emotion_tree:
    G.add_node(major)
    for sub in emotion_tree[major]:
        G.add_node(sub)
        G.add_edge(major, sub)

# Plotting the tree
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_size=10)
plt.title("Tree Structure of Emotion Categories")
plt.show()