# MathML-Lab
## Mathematical Foundations of Machine Learning From Scratch

MathML-Lab is a from-scratch exploration of the mathematical foundations underlying machine learning.

The project investigates how concepts from linear algebra, calculus, optimization, probability, and numerical computation translate into practical machine learning algorithms. Core algorithms are implemented using NumPy and their behavior is studied through controlled computational experiments.

The goal is not simply to use machine learning libraries, but to understand the mathematical structure behind the algorithms and investigate their behavior experimentally.

---

## Research Questions

This project explores questions such as:

- How do vectors and matrices represent data and transformations?
- Why does gradient descent converge for some learning rates but become unstable for others?
- How can analytical gradients be verified numerically?
- How does PCA use eigenvectors to identify directions of maximum variance?
- How does backpropagation propagate gradients through a neural network?
- Why do vanishing and exploding gradients occur in deep networks?
- How do optimization algorithms such as Momentum, RMSProp, and Adam differ in their training dynamics?
- How does model complexity affect generalization?
- How do L1/L2 regularization and dropout influence model behavior?
- How do batch normalization and dropout affect the training of deep neural networks?

---

## Topics Covered

### 1. Linear Algebra

- Vectors
- Dot products
- Cosine similarity
- Matrix operations
- Determinants
- Matrix inverse
- Matrix rank
- Vector projection
- Eigenvalues and eigenvectors
- Covariance matrices

### 2. Calculus and Optimization

- Derivatives
- Partial derivatives
- Gradients
- Loss functions
- Gradient descent
- Learning-rate effects
- Multivariable optimization
- Numerical gradient verification

### 3. Linear Regression

- Mathematical formulation
- Mean squared error
- Analytical gradient derivation
- Gradient descent from scratch
- Learning-rate experiments
- Gradient verification
- Validation against scikit-learn

### 4. Logistic Regression

- Linear scores
- Sigmoid function
- Probability interpretation
- Binary cross-entropy
- Gradient derivation
- Gradient descent
- Decision boundaries
- Classification thresholds
- Numerical gradient checking
- Validation against scikit-learn

### 5. Principal Component Analysis

- Mean centering
- Covariance matrices
- Eigenvalue decomposition
- Principal components
- Explained variance
- Dimensionality reduction
- Reconstruction
- Reconstruction error
- Validation against scikit-learn

### 6. Neural Networks

- Neurons and layers
- Linear transformations
- Activation functions
- Hidden layers
- Forward propagation
- Binary classification
- Loss functions
- Manual backward propagation
- Gradient descent
- Nonlinear datasets
- Numerical gradient checking

### 7. Backpropagation and Gradient Flow

- Chain rule
- Computational graphs
- Local derivatives
- Backpropagation derivation
- Matrix-form backpropagation
- Gradient checking
- Multiple hidden layers
- Gradient flow
- Vanishing gradients
- Exploding gradients
- Activation-function effects
- Weight initialization

### 8. Optimization Algorithms

- Batch gradient descent
- Stochastic gradient descent
- Mini-batch gradient descent
- Momentum
- RMSProp
- Adam
- Optimizer state
- Learning-rate sensitivity
- Hyperparameter sensitivity
- Optimizer comparison

### 9. Generalization and Regularization

- Training, validation, and test sets
- Underfitting
- Overfitting
- Model complexity
- Generalization gap
- L2 regularization
- Weight decay
- L1 regularization
- Sparsity
- Bias-variance intuition
- Controlled regularization experiments

### 10. Model Evaluation

- Confusion matrices
- Accuracy
- Precision
- Recall
- Specificity
- F1 score
- Precision-recall tradeoffs
- Classification thresholds
- ROC curves
- AUC
- Class imbalance
- Regression evaluation metrics
- Validation procedures

### 11. Deep Neural Networks

- Composition of layers
- Forward propagation through depth
- Internal representations
- Activation functions
- Weight initialization
- Gradient propagation
- Vanishing and exploding gradients
- Depth and optimization difficulty
- Controlled depth experiments
- Representation analysis

### 12. Batch Normalization and Dropout

- Activation statistics
- Standardization
- Batch normalization mathematics
- Batch normalization from scratch
- Batch normalization backpropagation
- Training vs. inference
- Dropout mathematics
- Dropout from scratch
- Expected activations
- Batch normalization and dropout interaction
- Controlled experiments

---

## Experimental Approach

A major focus of MathML-Lab is connecting mathematical derivations with computational experiments.

Examples include:

1. **Gradient Descent Stability**
   - Investigating how learning rate affects convergence and instability.

2. **Numerical Gradient Verification**
   - Comparing analytical gradients with finite-difference approximations.

3. **PCA and Feature Geometry**
   - Studying covariance structure, principal directions, and dimensionality reduction.

4. **Neural Network Gradient Checking**
   - Verifying backpropagation numerically.

5. **Optimizer Comparison**
   - Comparing SGD, Momentum, RMSProp, and Adam through training dynamics.

6. **Vanishing and Exploding Gradients**
   - Investigating gradient behavior as network depth increases.

7. **Depth Experiments**
   - Studying how network depth affects optimization and representation learning.

8. **Regularization Experiments**
   - Comparing models using no regularization, L2 regularization, dropout, and combinations of regularization techniques.

---

## Mathematical Perspective

The project is organized around the connection between mathematical concepts and machine learning:

| Mathematical Concept | Machine Learning Role |
|---|---|
| Linear Algebra | Data representation and transformations |
| Calculus | Derivatives and gradients |
| Optimization | Model training |
| Probability | Classification and uncertainty |
| Eigenvectors | Principal directions and dimensionality reduction |
| Numerical Methods | Gradient verification and computational validation |

This perspective allows machine learning algorithms to be studied not only as implementations, but as mathematical systems whose behavior can be analyzed experimentally.

---

## Implementation Philosophy

Core learning algorithms are implemented from first principles using NumPy.

Established machine learning libraries such as scikit-learn are used primarily for independent validation and comparison rather than as substitutes for the underlying implementations.

The project emphasizes:

- Mathematical derivation
- From-scratch implementation
- Numerical verification
- Controlled experiments
- Visualization
- Comparison with established implementations
- Interpretation of computational results

---

# How to Run the Project

## 1. Clone the Repository

Open a terminal or PowerShell and run:

```bash
git clone https://github.com/mussabaskari/MathML-Lab.git
```

Then enter the project directory:

```bash
cd MathML-Lab
```

---

## 2. Verify the Project Structure

The repository should contain:

```text
MathML-Lab/
│
├── experiments/
├── notebooks/
├── results/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

The main notebook is located at:

```text
notebooks/MathML_Lab_Full.ipynb
```

---

## 3. Install the Required Dependencies

Make sure Python is installed.

Then install all required packages using:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- NumPy
- Matplotlib
- Pandas
- SciPy
- Scikit-learn
- Jupyter

---

## 4. Start Jupyter Notebook

From the root of the project, run:

```bash
jupyter notebook
```

This starts the Jupyter Notebook server.

A browser window should open automatically.

If it does not, the terminal will provide a local address similar to:

```text
http://localhost:8888/tree
```

Open that address in your browser.

---

## 5. Open the Main Notebook

In Jupyter, navigate to:

```text
notebooks/
```

Then open:

```text
MathML_Lab_Full.ipynb
```

---

## 6. Select the Python Kernel

Make sure the notebook is using the Python environment where the project dependencies were installed.

In Jupyter, select the appropriate Python kernel.

---

## 7. Run the Notebook

You can run the notebook cell by cell to follow the mathematical development.

Alternatively, to execute the complete project:

**Kernel → Restart Kernel and Run All Cells**

This executes the mathematical derivations, implementations, experiments, numerical checks, visualizations, and comparisons throughout the notebook.

---

## 8. Explore the Results

The notebook contains experiments and visualizations covering:

- Gradient descent convergence
- Learning-rate sensitivity
- Gradient verification
- Linear regression
- Logistic regression
- PCA
- Neural network training
- Backpropagation
- Vanishing and exploding gradients
- Optimizer comparisons
- Generalization and regularization
- Batch normalization
- Dropout

The `experiments/` and `results/` directories contain supporting experimental outputs and results.

---

## Running the Project in VS Code

The project can also be run directly from Visual Studio Code.

### Step 1

Open the project folder:

```bash
code .
```

or open `MathML-Lab` manually in VS Code.

### Step 2

Install the **Jupyter** extension if it is not already installed.

### Step 3

Open:

```text
notebooks/MathML_Lab_Full.ipynb
```

### Step 4

Select the Python kernel/environment containing the project dependencies.

### Step 5

Run the notebook cells using the **Run Cell** buttons, or use **Run All** to execute the complete notebook.

---

## Stopping Jupyter

When finished, return to the terminal running Jupyter and press:

```text
Ctrl + C
```

Confirm the shutdown if prompted.

---

## Repository Structure

```text
MathML-Lab/
│
├── experiments/
│   └── regularization/
│       ├── batchnorm_experiment.png
│       └── dropout_training_curves.png
│
├── notebooks/
│   └── MathML_Lab_Full.ipynb
│
├── results/
│
├── src/
│   └── regularized_layers.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Project Goal

MathML-Lab was developed to build a deeper understanding of the mathematical foundations of machine learning by moving from mathematical definitions and derivations to implementations and controlled computational experiments.

Rather than treating machine learning algorithms as black boxes, the project studies their mathematical structure and investigates how changes in optimization, architecture, initialization, and regularization affect their behavior.

---

## Author

**Mussab Askari**

GitHub:  
https://github.com/mussabaskari