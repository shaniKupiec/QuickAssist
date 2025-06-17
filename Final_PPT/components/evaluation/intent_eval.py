from sklearn.metrics import accuracy_score

def calculate_intent_accuracy(test_data, predicted_intents):
 
    y_true = []
    y_pred = []

    for _, row in test_data.iterrows():
        true_intent = row.get('intent')
        predicted_intent = predicted_intents.get(row['input'])
        if true_intent is not None and predicted_intent is not None:
            y_true.append(true_intent)
            y_pred.append(predicted_intent)

    intent_accuracy = accuracy_score(y_true, y_pred)

    return intent_accuracy
