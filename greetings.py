from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import typer
from typing import Optional
from sklearn.neighbors import KNeighborsClassifier

app = typer.Typer()

@app.command()
def train(
    classifier: str = typer.Argument(["svm", "knn"], help="Classifier to use"),
    kernel: Optional[str] = typer.Option("linear", "--kernel", help="Kernel for SVM"),
    k: Optional[int] = typer.Option(None, "-k", help="Number of neighbors"),
    output: str = typer.Option("model.ckpt", "--output", "-o", help="Flag to enable output")
          ) -> tuple[float, str]:
    """Train and evaluate the model."""
    # Load the dataset
    data = load_breast_cancer()
    x = data.data
    y = data.target

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Train a Support Vector Machine (SVM) model
    if classifier == "svm":
        model = SVC(kernel=kernel, random_state=42)
    elif classifier == "knn":
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        raise ValueError("Invalid classifier")
    model.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    if output:
        print("Saving model to disk... ;)")

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report


if __name__ == "__main__":
    app()
