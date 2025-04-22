# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Task 1: Load and explore
try:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[t] for t in iris.target]
    
    print("First 5 rows:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
    
    print("\nMissing values:")
    print(df.isnull().sum())

# Task 2: Basic analysis
    print("\nBasic statistics:")
    print(df.describe())
    
    print("\nMean measurements by species:")
    print(df.groupby('species').mean())

# Task 3: Visualizations
    # Line chart (using index as pseudo-time)
    df['sepal length (cm)'].plot(title='Sepal Length Trend', figsize=(8,4))
    plt.xlabel('Index')
    plt.ylabel('Length (cm)')
    plt.show()

    # Bar chart
    df.groupby('species')['petal length (cm)'].mean().plot(kind='bar')
    plt.title('Average Petal Length by Species')
    plt.ylabel('Length (cm)')
    plt.show()

    # Histogram
    df['sepal width (cm)'].hist()
    plt.title('Sepal Width Distribution')
    plt.xlabel('Width (cm)')
    plt.ylabel('Frequency')
    plt.show()

    # Scatter plot
    df.plot.scatter(x='sepal length (cm)', y='petal length (cm)')
    plt.title('Sepal vs Petal Length')
    plt.show()

except Exception as e:
    print(f"Error occurred: {e}")
