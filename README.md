# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.

## Neural Network Model
![Screenshot 2025-03-03 110546](https://github.com/user-attachments/assets/5170c853-f616-4d71-95d1-a81ea679770a)



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        class NeuralNet(nn.Module):
  def _init_(self):
    super()._init_()
    self.fc1 = nn. Linear (1, 12)
    self.fc2 = nn. Linear (12, 10)
    self.fc3 = nn. Linear (10, 1)
    self.relu = nn.ReLU()
    self.history = {'loss': []}

  def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      x = self.fc3(x) # No activation here since it's a regression t.
      return x




# Initialize the Model, Loss Function, and Optimizer



def train_model (ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion (ai_brain (X_train), y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')



```
## Dataset Information
![alt text](<Screenshot 2025-03-03 112507.png>)


## OUTPUT

### Training Loss Vs Iteration Plot

Include your plot 
![alt text](<Screenshot 2025-03-03 113954.png>)

### New Sample Data Prediction

![alt text](<Screenshot 2025-03-03 114133.png>)

## RESULT

The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
