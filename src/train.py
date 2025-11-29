import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from  sklearn.metrics import mean_absolute_error,r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor 
import joblib

#1=Load Data
# ... imports stay the same ...

# 1. Load Data
df = pd.read_csv("CAR DETAILS FROM CAR DEKHO.csv")

# === FIX 1: Drop the 'name' column (Text is too messy for now) ===
# We also drop 'selling_price' because that is our target (y)
X = df.drop(columns=['selling_price', 'name'])
Y = df['selling_price']

# === FIX 2: Check column names and add 'owner' to encoding ===
# Printing columns to be safe (check your terminal output if this fails again)
print("Features being used:", X.columns.tolist())

# We must encode ALL text columns. 
# Based on your dataset, 'owner' is also text ("First Owner"), so we add it here.
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']

# Note: If your CSV has capitalized headers like 'Fuel_Type', change the list above to match!
# Let's handle the case sensitivity automatically:
categorical_features = [col for col in X.columns if X[col].dtype == 'object']

print(f"Categorical Columns found: {categorical_features}")

# 2. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# ... The rest of the code (Split Data, MLflow, Training) stays exactly the same ...    remainder='passthrough' #keep numeric columns as is


#3.split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#Mlflowexperiment

#set the experiment name so all runs are grouped together
mlflow.set_experiment('Car_price_predication_experiment')

##Define hyperparameters we want to track 
params = {
    'n_estimators':200,
    'learning_rate':0.1,
    'max_depth':20,
    'random_state':42
}

#Start the Run 

with mlflow.start_run():
    print('Starting Training...')

    #4. Create the full pipeline(preprocessing +model)
    model_pipeline = Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('model',XGBRegressor(**params))

    ])

    #5 .Train
    model_pipeline.fit(X_train, Y_train)

    #6 .Evaluate
    predictions = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_absolute_error(Y_test,predictions))
    r2= r2_score(Y_test,predictions)

    print(f'Trianing complete.Rmse : {rmse}')

    #7.LOG EVERYTHING To MLFLow
    #log parameters(What we used)
    mlflow.log_params(params)

    #Log Metrics(How well it worked)
    mlflow.log_metric('rmse',rmse)
    mlflow.log_metric('r2_score',r2)


    #Log the model itself (Save the file inside Mlflow)
    mlflow.sklearn.log_model(model_pipeline,'model')

    print('Model and metrics logged to Mlflow !')


    print("💾 Saving model to 'model.pkl' for deployment...")
    joblib.dump(model_pipeline, 'model.pkl')
    print("✅ Model saved successfully!")