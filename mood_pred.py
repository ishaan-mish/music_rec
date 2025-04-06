from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import joblib

# Extended dataset for better accuracy
data = [
    [1, 1, 0, 1, 0, 'happy'],
    [1, 1, 0, 1, 0, 'happy'],
    [0, 0, 1, 0, 1, 'sad'],
    [0, 0, 1, 0, 1, 'sad'],
    [0, 0, 1, 0, 1, 'sad'],
    [1, 1, 0, 1, 0, 'groovy'],
    [1, 0, 0, 1, 0, 'groovy'],
    [0, 0, 1, 0, 1, 'chill'],
    [0, 0, 1, 0, 1, 'chill'],
    [1, 0, 1, 0, 1, 'chill'],
    [0, 1, 0, 1, 0, 'groovy'],
    [1, 1, 1, 1, 0, 'happy'],
    [0, 0, 0, 0, 1, 'chill'],
    [0, 0, 0, 0, 0, 'sad'],
    [0, 1, 0, 1, 1, 'groovy'],
    [1, 0, 0, 0, 1, 'chill'],
    [0, 0, 0, 0, 0, 'sad'],
    [0, 1, 1, 0, 1, 'chill'],
]

df = pd.DataFrame(data, columns=[
    "likes_dancing", "prefers_loud_music", "likes_acoustic", 
    "feels_energetic", "likes_relaxing_music", "mood"
])

X = df.drop(columns="mood")
y = df["mood"]

# Train the decision tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X, y)

# Save the model
joblib.dump(clf, "mood_decision_tree_model.pkl")

# Display the tree rules
rules = export_text(clf, feature_names=X.columns.tolist())
print(rules)
