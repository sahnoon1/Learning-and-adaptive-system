import numpy as np
import pandas as pd

# Dataset
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast',
                'Sunny','Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool',
                    'Mild','Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal',
                 'High','Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong',
             'Weak','Weak','Weak','Strong','Strong','Weak','Strong'],
    'PlayTennis': ['No','No','Yes','Yes','Yes','No','Yes',
                   'No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

def entropy(labels):
    values, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(df, feature, target):
    total_entropy = entropy(df[target])
    weighted_entropy = 0

    for value in df[feature].unique():
        subset = df[df[feature] == value][target]
        weight = len(subset) / len(df)
        weighted_entropy += weight * entropy(subset)

    return total_entropy - weighted_entropy


def main():
    target = 'PlayTennis'
    features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

    print(f"Dataset Entropy: {entropy(df[target]):.4f}\n")

    print("Information Gain for each feature:\n")

    ig_scores = {}

    for feature in features:
        ig = information_gain(df, feature, target)
        ig_scores[feature] = ig
        print(f"{feature}: {ig:.4f}")

    best_feature = max(ig_scores, key=ig_scores.get)

    print(f"\nBest feature to split on: {best_feature}")

if __name__ == "__main__":
    main()
  
