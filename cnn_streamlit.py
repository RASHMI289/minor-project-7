
# ============================================================
# MINOR PROJECT: CNN IMAGE CLASSIFICATION AND STREAMLIT APP
# PART A: CNN IMAGE CLASSIFICATION USING MNIST DATASET
# PART B: STUDENT MARKS CALCULATOR USING STREAMLIT
# ============================================================


# -------------------- 1. IMPORT LIBRARIES --------------------

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from io import StringIO


# -------------------- 2. STREAMLIT PAGE SETTINGS --------------------

st.set_page_config(
    page_title="CNN Image Classification Project",
    page_icon="🧠",
    layout="wide"
)


# -------------------- 3. LOAD AND TRAIN CNN MODEL --------------------

@st.cache_resource(show_spinner=False)
def train_cnn_model():

    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize image pixel values
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # Reshape images for CNN
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Create CNN model
    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu"
        ),

        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128, activation="relu"
        ),

        tf.keras.layers.Dense(
            10, activation="softmax"
        )
    ])

    # Compile CNN model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Save model summary
    summary_text = StringIO()
    model.summary(print_fn=lambda line: summary_text.write(line + "\n"))
    model_summary = summary_text.getvalue()

    # Train CNN model
    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )

    # Evaluate model using test dataset
    test_loss, test_accuracy = model.evaluate(
        X_test, y_test, verbose=0
    )

    # Predict test images
    predictions = model.predict(
        X_test, verbose=0
    )

    predicted_labels = np.argmax(predictions, axis=1)

    return (
        model,
        history,
        model_summary,
        test_loss,
        test_accuracy,
        X_test,
        y_test,
        predicted_labels
    )


# -------------------- 4. PROJECT TITLE --------------------

st.title("🧠 CNN Image Classification and Streamlit Application")

st.write(
    "This project uses a Convolutional Neural Network (CNN) "
    "to classify handwritten digits from the MNIST dataset."
)

st.divider()


# -------------------- 5. PART A: CNN IMAGE CLASSIFICATION --------------------

st.header("Part A: CNN Image Classification")

st.subheader("Dataset Information")

st.write("Dataset: MNIST Handwritten Digits Dataset")
st.write("Image size: 28 × 28 pixels")
st.write("Number of classes: 10")
st.write("Classes: Digits 0 to 9")
st.write("Training images: 60,000")
st.write("Testing images: 10,000")

st.write(
    "Each image contains a handwritten digit. "
    "The CNN model learns to identify the digit."
)


# -------------------- 6. TRAIN THE MODEL --------------------

with st.spinner(
    "Loading MNIST dataset and training the CNN model. Please wait..."
):

    (
        model,
        history,
        model_summary,
        test_loss,
        test_accuracy,
        X_test,
        y_test,
        predicted_labels
    ) = train_cnn_model()

st.success("CNN model training completed!")


# -------------------- 7. DISPLAY SAMPLE IMAGES --------------------

st.subheader("Sample Images from MNIST Dataset")

fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, ax in enumerate(axes.flat):

    ax.imshow(X_test[i].reshape(28, 28), cmap="gray")
    ax.set_title(f"Label: {y_test[i]}")
    ax.axis("off")

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# -------------------- 8. DISPLAY CNN MODEL SUMMARY --------------------

st.subheader("CNN Model Summary")

st.text(model_summary)


# -------------------- 9. DISPLAY TRAINING ACCURACY --------------------

st.subheader("Training and Validation Accuracy")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

ax.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("CNN Model Accuracy")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)


# -------------------- 10. DISPLAY TRAINING LOSS --------------------

st.subheader("Training and Validation Loss")

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    history.history["loss"],
    label="Training Loss"
)

ax.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("CNN Model Loss")
ax.legend()
ax.grid(True)

st.pyplot(fig)
plt.close(fig)


# -------------------- 11. DISPLAY TEST PERFORMANCE --------------------

st.subheader("CNN Model Test Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Test Accuracy",
        f"{test_accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Test Loss",
        f"{test_loss:.4f}"
    )


# -------------------- 12. COMPARE ACTUAL AND PREDICTED LABELS --------------------

st.subheader("Actual and Predicted Labels")

comparison = pd.DataFrame({
    "Actual Label": y_test[:10],
    "Predicted Label": predicted_labels[:10]
})

comparison["Result"] = np.where(
    comparison["Actual Label"] == comparison["Predicted Label"],
    "Correct",
    "Incorrect"
)

st.dataframe(comparison, use_container_width=True)


# -------------------- 13. DISPLAY CORRECT AND INCORRECT PREDICTIONS --------------------

correct_indices = np.where(
    y_test == predicted_labels
)[0]

incorrect_indices = np.where(
    y_test != predicted_labels
)[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("Correct Predictions", len(correct_indices))

with col2:
    st.metric("Incorrect Predictions", len(incorrect_indices))


# Show some incorrect predictions if available
st.subheader("Examples of Incorrect Predictions")

if len(incorrect_indices) > 0:

    fig, axes = plt.subplots(1, min(5, len(incorrect_indices)),
                             figsize=(12, 3))

    for i, ax in enumerate(np.atleast_1d(axes)):

        index = incorrect_indices[i]

        ax.imshow(X_test[index].reshape(28, 28), cmap="gray")

        ax.set_title(
            f"Actual: {y_test[index]}\n"
            f"Predicted: {predicted_labels[index]}"
        )

        ax.axis("off")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

else:
    st.write("No incorrect predictions were found.")


# -------------------- 14. CHALLENGES AND IMPROVEMENTS --------------------

st.subheader("Challenges and Future Improvements")

st.write("""
- The model may find some handwritten digits difficult to classify.
- Different handwriting styles can affect prediction accuracy.
- More training epochs may improve learning.
- Additional CNN layers can be tested to improve performance.
""")


st.divider()


# -------------------- 15. PART B: STUDENT MARKS CALCULATOR --------------------

st.header("Part B: Student Marks Calculator")

st.write(
    "This application calculates the total marks, percentage, "
    "and pass or fail result of a student."
)

# Student marks input form
with st.form("marks_form"):

    student_name = st.text_input("Enter Student Name")

    st.subheader("Enter Marks (0 to 100)")

    subject1 = st.number_input(
        "Subject 1", min_value=0, max_value=100, step=1
    )

    subject2 = st.number_input(
        "Subject 2", min_value=0, max_value=100, step=1
    )

    subject3 = st.number_input(
        "Subject 3", min_value=0, max_value=100, step=1
    )

    subject4 = st.number_input(
        "Subject 4", min_value=0, max_value=100, step=1
    )

    subject5 = st.number_input(
        "Subject 5", min_value=0, max_value=100, step=1
    )

    submit = st.form_submit_button("Calculate Result")


# -------------------- 16. CALCULATE AND DISPLAY RESULT --------------------

if submit:

    if not student_name.strip():

        st.error("Please enter the student name.")

    else:

        marks = [
            subject1,
            subject2,
            subject3,
            subject4,
            subject5
        ]

        # Calculate total marks
        total = sum(marks)

        # Calculate percentage
        percentage = (total / 500) * 100

        # Check pass or fail
        # Assumption: minimum 35 marks in each subject to pass
        if all(mark >= 35 for mark in marks):
            result = "PASS"
        else:
            result = "FAIL"

        st.subheader("Student Result")

        st.write("Student Name:", student_name)
        st.write("Total Marks:", total, "/ 500")
        st.write("Percentage:", f"{percentage:.2f}%")

        if result == "PASS":
            st.success("Result: PASS")
        else:
            st.error("Result: FAIL")

        # Display subject-wise marks
        result_table = pd.DataFrame({
            "Subject": [
                "Subject 1",
                "Subject 2",
                "Subject 3",
                "Subject 4",
                "Subject 5"
            ],
            "Marks": marks
        })

        st.subheader("Subject-wise Marks")
        st.dataframe(result_table, use_container_width=True)


# -------------------- 17. PROJECT CONCLUSION --------------------

st.divider()

st.header("Project Conclusion")

st.write("""
The project demonstrates how a Convolutional Neural Network
can classify handwritten digits using the MNIST dataset.

The Streamlit application provides a simple interface for
calculating student marks, percentage, and results.

This project combines machine learning and a web application
in a single Python program.
""")
