import datetime, time

import torch
import torch.nn as nn
import torch.optim as optim
import optuna, joblib, pandas as pd, numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
from sklearn.metrics import mean_absolute_error as mae, mean_absolute_percentage_error as mape

# Configuration flags
salaryMode = "salaryTo"
validationPath = "./VacanciesValidationDataSet.xlsx"
trainingPath = "./VacanciesDataSet.xlsx"
SCALE_SALARY = False  # Flag to control salary scaling
SCALE_PREDICTIONS = False  # Flag to control prediction scaling



class SalaryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Neural Network Model
class SalaryPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(SalaryPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.output(x)


# Training Function
def train_model(model, dataloader, criterion, optimizer, device):
    model.train(True)
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


# Evaluation Function
def evaluate_model(model, dataloader, device, salary_scaler=None):
    model.eval()
    preds = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy().flatten())
            true_labels.extend(labels.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    preds = np.array(preds).reshape(-1, 1)
    true_labels = np.array(true_labels).reshape(-1, 1)
    
    # Scale predictions and true labels if needed
    if SCALE_PREDICTIONS and salary_scaler is not None:
        preds = salary_scaler.inverse_transform(preds)
        true_labels = salary_scaler.inverse_transform(true_labels)
    
    # Calculate metrics
    r2 = r2_score(true_labels, preds)
    mape_error = mape(true_labels, preds)
    mae_error = mae(true_labels, preds)
    
    print(f"{salaryMode}: R2: {r2}, MAE: {mae_error}, MAPE: {mape_error}")
    return r2, mae_error, mape_error


# Objective Function for Optuna
def objective(trial):
    # Hyperparameters to optimize
    hidden_dim = trial.suggest_int("hidden_dim", 8, 512)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # Preprocess data
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SalaryPredictor(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate
    for epoch in range(10):
        train_model(model, train_loader, criterion, optimizer, device)

    r2, _, _ = evaluate_model(model, test_loader, device, salary_scaler)  # Only use R2 score
    return r2


def main():
    global train_dataset, test_dataset, X_train, X_test, salary_scaler
    
    # Load and prepare data
    data = pd.read_excel(
        trainingPath, 
        sheet_name="data", 
        usecols=["description", "salaryFrom", "salaryTo"]
    )
    descriptions = data['description']
    salary = data[salaryMode]

    # Initialize salary scaler if needed
    salary_scaler = MinMaxScaler() if SCALE_SALARY else None
    
    # Scale salary if needed
    if SCALE_SALARY:
        salary = salary_scaler.fit_transform(salary.values.reshape(-1, 1)).flatten()
    else:
        salary = salary.values

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_features = vectorizer.fit_transform(descriptions).toarray()
    print(tfidf_features.shape)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(tfidf_features, salary, test_size=0.1, random_state=42)

    # Create datasets
    train_dataset = SalaryDataset(X_train, y_train)
    test_dataset = SalaryDataset(X_test, y_test)

    print(X_train.shape)
    
    # Optimize hyperparameters
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    # Print best hyperparameters
    print("Best hyperparameters:", study.best_params)
    print("Best R² score:", study.best_value)

    # Train final model with best hyperparameters
    best_params = study.best_params
    batch_size = best_params["batch_size"]
    hidden_dim = best_params["hidden_dim"]
    dropout_rate = best_params["dropout_rate"]
    learning_rate = best_params["learning_rate"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SalaryPredictor(input_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout_rate=dropout_rate).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    avg_r2 = []
    avg_mae = []
    avg_mape = []
    print("for now we would run the model 10 times and take the average R2 score")
    #--------------------------------
    for i in range(10):
        # Train final model
        epoch_amount = 20
        for epoch in range(epoch_amount):
            train_model(model, train_dataset, criterion, optimizer, device)
            print(f"{epoch}/{epoch_amount}", evaluate_model(model, test_dataset, device, salary_scaler))
        
        r2 = evaluate_model(model, test_dataset, device, salary_scaler)
        print("final result after epochs by training: ", r2)
        model_state_dict = model.state_dict()

        # Evaluate on validation data
        unseen_data = pd.read_excel(
            validationPath, 
            sheet_name="validation", 
            usecols=['description', "salaryFrom", "salaryTo"]
        )

        descriptions_valid = unseen_data['description']
        salary_valid = unseen_data[salaryMode]
        
        # Scale validation salary if needed
        if SCALE_SALARY:
            salary_valid = salary_scaler.transform(salary_valid.values.reshape(-1, 1)).flatten()
        else:
            salary_valid = salary_valid.values

        validation = SalaryDataset(vectorizer.transform(descriptions_valid).toarray(), salary_valid)
        r2, mae_error, mape_error = evaluate_model(model, validation, device, salary_scaler)    
        print(f"{i+1}th run: the {salaryMode} is having score {r2} in unseen data")
        avg_r2.append(r2)
        avg_mae.append(mae_error)
        avg_mape.append(mape_error)
    #--------------------------------
    print(f"the average R2 score is {sum(avg_r2) / len(avg_r2)}")
    print(f"the average MAE score is {sum(avg_mae) / len(avg_mae)}")
    print(f"the average MAPE score is {sum(avg_mape) / len(avg_mape)}")


    # Save model and configuration
    today = str(datetime.datetime.now().day) + "-" + str(datetime.datetime.now().month) + "-" + str(datetime.datetime.now().year)
    torch.save({
        'model_state_dict': model_state_dict,
        'vectorizer': vectorizer,
        'salary_scaler': salary_scaler,
        'best_params': best_params,
        'scale_salary': SCALE_SALARY,
        'scale_predictions': SCALE_PREDICTIONS
    }, f'/salary_predictor_model_{round(r2, 4)}_{today}.pth')


if __name__ == '__main__':
    main()
