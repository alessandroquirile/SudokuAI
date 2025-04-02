import os
import pickle

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def load_dataset():
    path = kagglehub.dataset_download("karnikakapoor/digits")
    dataset_path = os.path.join(path, "digits updated/digits updated")
    return dataset_path


def preprocess_images(dataset_path, img_size=(28, 28)):
    x = []
    y = []

    for digit in sorted(os.listdir(dataset_path)):
        digit_path = os.path.join(dataset_path, digit)
        if not os.path.isdir(digit_path):
            continue
        images = os.listdir(digit_path)

        for img_name in images:
            img_path = os.path.join(digit_path, img_name)
            img = Image.open(img_path).convert("L")
            img = img.resize(img_size)
            img_array = np.array(img).flatten()
            x.append(img_array)
            y.append(int(digit))

    x = np.array(x)
    y = np.array(y)
    return x, y


def cross_validation(knn, x_train, y_train, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    cv_scores = cross_val_score(knn, x_train, y_train, cv=kf)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    return cv_scores, cv_mean, cv_std


def grid_search_tuning(x_train, y_train):
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(x_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(best_knn, x_test, y_test):
    y_pred = best_knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return accuracy, y_pred, cm


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix - KNN Classifier with GridSearch")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Salva l'immagine nel percorso specificato
    plt.savefig(save_path)

    # Mostra la matrice di confusione
    plt.show()


def print_classification_report(y_test, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def plot_images_with_predictions(x_test, y_test, y_pred, num_images=10):
    random_indices = np.random.choice(len(x_test), num_images, replace=False)

    plt.figure(figsize=(12, 12))

    for i, idx in enumerate(random_indices):
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    dataset_path = load_dataset()
    print(f"Dataset path: {dataset_path}")

    x, y = preprocess_images(dataset_path)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    print(f"x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
    print(f"x_test shape:", x_test.shape, "y_test shape:", y_test.shape)

    knn = KNeighborsClassifier()

    k = 5
    cv_scores, cv_mean, cv_std = cross_validation(knn, x_train, y_train, k)
    print(f"\nCross-validation scores ({k}-fold): {cv_scores}")
    print(f"Mean performance: {cv_mean:.2%} ± {cv_std:.2%}")

    best_knn, best_params = grid_search_tuning(x_train, y_train)
    print("\nBest parameters found by GridSearch:")
    print(best_params)

    accuracy, y_pred, cm = evaluate_model(best_knn, x_test, y_test)
    print(f"Accuracy of the model with GridSearch: {accuracy:.2%}")
    # plot_images_with_predictions(x_test, y_test, y_pred, num_images=10)
    plot_confusion_matrix(cm, "confusion_matrix.png")
    print_classification_report(y_test, y_pred)

    save_model(best_knn, '../../knn.pkl')
