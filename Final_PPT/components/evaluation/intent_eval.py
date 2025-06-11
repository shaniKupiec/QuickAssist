import json
from sklearn.metrics import accuracy_score

# Prepare training data with intents
# Intent accuracy calculation
def calculate_intent_accuracy(test_data, predicted_intents, save_path="intent_metrics.json"):
 
    y_true = []
    y_pred = []

    for _, row in test_data.iterrows():
        true_intent = row.get('intent')
        predicted_intent = predicted_intents.get(row['input'])
        if true_intent is not None and predicted_intent is not None:
            y_true.append(true_intent)
            y_pred.append(predicted_intent)

    intent_accuracy = accuracy_score(y_true, y_pred)
    print(f"Intent Recognition Accuracy: {intent_accuracy:.4f}")

    if save_path:
        with open(save_path, "w") as f:
            json.dump({"intent_accuracy": intent_accuracy}, f, indent=2)

    return intent_accuracy
