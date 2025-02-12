import numpy as np

def boosting(train_data, train_labels, test_data, test_labels, model, n_estimators=50):
    
    n_samples = train_data.shape[0]
    weights = np.ones(n_samples) / n_samples
    classifiers = []
    alpha_values = []

    for _ in range(n_estimators):
        clf = model.__class__(**model.get_params())
        clf.fit(train_data, train_labels, sample_weight=weights)
        predictions = clf.predict(train_data)

        misclassified = predictions != train_labels
        error = np.sum(weights[misclassified]) / np.sum(weights)

        if error > 0.5:
            break
        elif error == 0:
            alpha = 1
        else:
            alpha = 0.5 * np.log((1 - error) / error)

        weights *= np.exp(alpha * misclassified * 2)
        weights /= np.sum(weights)

        classifiers.append(clf)
        alpha_values.append(alpha)

    final_predictions = np.zeros(test_data.shape[0])

    for clf, alpha in zip(classifiers, alpha_values):
        final_predictions += alpha * clf.predict(test_data)

    final_predictions = np.sign(final_predictions)
    accuracy = np.mean(final_predictions == test_labels)
    return accuracy
