
# CNN Fungi Classification

This project demonstrates the implementation of a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of fungi. The model is trained on a dataset of preprocessed images, and the training process is monitored over 10 epochs.

## Project Structure

- **Data Loading**: The data is preprocessed and loaded from external URLs using `requests` and `pickle` libraries.
- **Model Architecture**: The model consists of convolutional layers, max-pooling layers, and dense layers, designed for image classification.
- **Training**: The model is trained using the Adam optimizer and sparse categorical crossentropy loss.
- **Evaluation**: The model's performance is evaluated using accuracy and loss metrics on both training and validation datasets.

## Installation

1. Clone the repository:
   ```bash
   git clone repository-url
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Jupyter notebook or Python script to train the CNN model on the fungi dataset:
```bash
python cnn_fungi_classification.py
```

## Model Architecture

The CNN model is structured as follows:

- **Conv2D Layer**: 32 filters, 3x3 kernel size, ReLU activation
- **MaxPooling2D Layer**: 2x2 pool size
- **Flatten Layer**: Flattens the input
- **Dense Layer**: 64 units, ReLU activation
- **Output Layer**: 2 units, Sigmoid activation (for binary classification)

## Insights on Epoch Results

The model was trained over 10 epochs, and the results showed a gradual improvement in both accuracy and loss on the validation dataset:

- **Epoch 1**: The model started with a low accuracy of around 44.76% and a high loss, indicating that the model was initially underfitting.
- **Epoch 2-4**: Significant improvement was observed, with the accuracy increasing and loss decreasing. By Epoch 4, the validation accuracy reached 76%, showing that the model was learning the features effectively.
- **Epoch 5-7**: The model's performance stabilized, with validation accuracy hovering around 72-74% and a slight decrease in validation loss. This indicates that the model was converging.
- **Epoch 8-10**: The accuracy remained stable, but there was no significant improvement, suggesting that the model might have reached its optimal learning capacity with the given architecture and data.

### Possible Improvements

1. **Data Augmentation**: Implementing data augmentation techniques such as rotation, zoom, or horizontal flip could help improve model generalization.
2. **Model Tuning**: Experimenting with different model architectures, such as deeper networks or alternative activation functions, could lead to better performance.
3. **Regularization**: Adding dropout layers or L2 regularization might help prevent overfitting and improve validation performance.
4. **Learning Rate Adjustment**: Implementing a learning rate scheduler could help the model converge more effectively.
5. **Cross-Validation**: Using cross-validation could provide a more robust evaluation of the model's performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- The dataset and some of the utilities are provided by [BC-EDX](https://static.bc-edx.com/).
- This project uses TensorFlow, Keras, and Scikit-learn for model development and evaluation.

``

### Explanation of the Epoch Results

- **Accuracy & Loss Trends**: As the epochs progress, the model shows an improvement in accuracy and a decrease in loss. This is typical behavior when the model starts to learn the features in the data effectively. However, by the later epochs (8-10), the model's performance stabilizes, indicating that it may not improve much further without adjustments to the model architecture, data, or training process.
- **Validation vs. Training**: The validation accuracy and loss are crucial metrics. If there's a significant gap between training and validation performance, it might indicate overfitting. In this case, the model maintains a relatively close validation accuracy to the training accuracy, suggesting a good generalization to unseen data.

