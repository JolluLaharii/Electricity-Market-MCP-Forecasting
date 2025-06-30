import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

weather=pd.read_csv('../weather_6months.csv')
weather_july=pd.read_csv('../weather_july_1.csv')
weather_july_np=np.array(weather_july)
mcp=pd.read_csv('../mcp_6months.csv')
weather_np=np.array(weather)
mcp_np=np.array(mcp)
mcp_july=pd.read_csv('../mcp_july.csv')
mcp_july_np=np.array(mcp_july)

mcp_scaler=StandardScaler()
mcp_np=mcp_scaler.fit_transform(mcp_np)

weather_scaler=StandardScaler()
weather_np=weather_scaler.fit_transform(weather_np)
weather_july_np=weather_scaler.fit_transform(weather_july_np)


lags=[24,48,168]
epoches = 100
hidde_size = 64
batch_size = 100
learning_rate=0.001

class NARXDataset(Dataset):
    def __init__(self,y,x_exog,lags):
        self.inputs=[]
        self.output=[]
        max_lag=max(lags)
        # y_lags=[]

        for t in range(max_lag,len(y)):
            y_lags=np.array([y[t-lag] for lag in lags]).flatten()
            x_now=x_exog[t]
            self.inputs.append(np.concatenate([y_lags,x_now]))
            self.output.append(y[t])
        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.output = torch.tensor(np.array(self.output), dtype=torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        return self.inputs[idx], self.output[idx]

class NARXModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)
def train_model(model, dataset, epochs, batch_size, lr):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


def recursive_forecast(model, y_hist, x_future, lags, steps):
    model.eval()
    preds = []
    y_hist = list(y_hist)  # convert to mutable list if it's a tensor/array

    for t in range(steps):
        y_lagged = np.array([y_hist[-lag] for lag in lags]).flatten()
        x_now = x_future[t]
        x_input = torch.tensor(np.concatenate([y_lagged, x_now]), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_next = model(x_input).item()
        preds.append(y_next)
        y_hist.append(y_next)  # update history with prediction

    return np.array(preds)


train_data=NARXDataset(mcp_np,weather_np,lags)
input_size=6
model=NARXModel(input_size)
train_model(model,train_data,epoches,batch_size,learning_rate)
scaled_prediction = recursive_forecast(model,mcp_np,weather_july_np,lags,24)
prediction=mcp_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))

plt.figure(figsize=(12, 5))
plt.plot(mcp_july, label="Actual MCP (July)", linewidth=2, marker='o')
plt.plot(prediction, label="Predicted MCP", linewidth=2, linestyle='--', marker='x')
plt.title("MCP Forecast vs Actuals (July)")
plt.xlabel("Time step (Hour)")
plt.ylabel("MCP (Rs/MWh)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
