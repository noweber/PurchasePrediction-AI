import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    months = {
        "Jan" : 0,
        "Feb" : 1,
        "Mar" : 2,
        "Apr" : 3,
        "May" : 4,
        "June" : 5,
        "Jul" : 6,
        "Aug" : 7,
        "Sep" : 8,
        "Oct" : 9,
        "Nov" : 10,
        "Dec" : 11
    }

    visitorTypes = {
        "New_Visitor": 0,
        "Returning_Visitor": 1,
        "Other": 2
    }

    labels = []
    evidence = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        try:
            rows = list(reader)
            # print("rows: ", rows)
            for i in range(len(rows)):
                # Skip the first row because it only contains headers
                if i > 0:
                    # Get the evidence:
                    evidence.append(list(rows[i]))

                    # Convert Month to an integer
                    evidence[i-1][10] = months[evidence[i-1][10]]

                    # Convert VisitorType to an integer:
                    evidence[i-1][15] = visitorTypes[evidence[i-1][15]]

                    # Convert Weekend to an integer:
                    if evidence[i-1][16] == "TRUE":
                        evidence[i-1][16] = 1
                    else:
                        evidence[i-1][16] = 0

                    # Get the label:
                    label = evidence[i-1].pop()
                    if label == "TRUE":
                        labels.append(1)
                    else:
                        labels.append(0)
                    
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

    return evidence, labels
            

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    # Validate input:
    number_of_predictions = len(predictions)
    number_of_labels = len(labels)
    if number_of_predictions != number_of_labels:
        raise ValueError("Number of predictions must equal the number of labels.")
    if number_of_predictions == 0:
        return 0, 0

    # Compare predictions to labels:
    true_positive_predictions = 0 
    true_negative_predictions = 0 
    number_of_positive_labels = 0
    number_of_negative_labels = 0
    for i in range(number_of_predictions):
        if labels[i] == 0:
            number_of_negative_labels += 1
        else:
            number_of_positive_labels += 1

        if predictions[i] == labels[i]:
            if predictions[i] == 1:
                true_positive_predictions += 1
            else:
                true_negative_predictions += 1

    return (true_positive_predictions / number_of_positive_labels), (true_negative_predictions / number_of_negative_labels)


if __name__ == "__main__":
    main()
