import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

BASE_DIR = r"C:\Users\elakkiya\json_tutorial\Chennai_Weather_Prediction"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

#LOADING DATA
def load_data():
    x_train = pd.read_csv(os.path.join(DATA_DIR, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
    x_test = pd.read_csv(os.path.join(DATA_DIR, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))
    return x_train, y_train, x_test, y_test
#LOADING THE SCALER - HERE STANDARDSCALER()
def load_scaler():
    return joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
#SAVE THE MODEL - any
def save_model(model, name):
    save_model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, save_model_path)
    print(f"Model saved successfully in {save_model_path}")
#LOAD THE MODEL
def load_model(name):
    return joblib.load(os.path.join(MODELS_DIR, f"{name}.pkl"))
    
#EVALUATE THE MODEL WITH RMSE AND R2SCORE
def evaluate_model(x_test, y_test, name=None):
    print(f"{name} Model Evaluation starts...")
    model  = load_model(name)
    if name =="linear_regression":
        print(f"predicting with {name}...")
        pred = model.predict(x_test)
        print("Calculating Metrics...")
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        print(f"Visualizing {name}...")
        visualize_model(name, y_test, pred)
        return rmse, r2
        
    elif name in ["lasso_regression" ,"lassoCV", "ridge_regression", "ridgeCV"]:
        print(f"predicting with {name}...")
        pred = model.predict(x_test)
        print("Calculating Metrics...")
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        print(f"Visualizing {name}...")
        #sending model to get the coefficients
        if hasattr(x_test, "columns"):
            visualize_model(name, y_test, pred, model = model, 
                            x_test = x_test, rmse=rmse, r2=r2)
        else:
            print("Visualization Not done: Check if feature names are passed with Lasso x_test data set")
        return rmse, r2
    else:
        print("Evaluation is not defined for this model")
        return None
        

def set_up_viz(y_test, pred, color="red", title = "Actual vs Predicted", rmse=None, r2=None):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, pred, alpha=0.6, color="red", label = "Predictions")
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], color = "blue", 
             linestyle="--", label = "Fit Line")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title(title)

    if rmse is not None and r2 is not None:
        text = f"RMSE: {rmse:.2f}\nR-squared: {r2:.3f}"
        plt.text(0.05, 0.95, text, 
                 transform = plt.gca().transAxes,
                 fontsize = 11, color="black",
                 verticalalignment = "top",
                 bbox = dict(boxstyle="round,pad=0.4", 
                             facecolor="lightgray",
                             alpha=0.5))
    
    plt.legend()
    plt.tight_layout()
    plt.show()

##VISUALIZING THE TRAIN TEST SPLIT - TIME SERIES VIZ
def visualize_model(name,y_test, pred, model=None, x_test=None, rmse=None, r2=None):
    #LINEAR REGRESSION
    if name=="linear_regression":
        try:
            set_up_viz(y_test, pred, title = f"{name}: Actual vs Predicted", 
                       rmse=rmse, r2=r2)
            print(f"{name} - Visualization done")
        except Exception as e:
            print(f"An error occured during visualization: {e}")
            
    #2. LASSO REGRESSION, RIDGE REGRESSION
    elif name in ["lasso_regression", "lassoCV", "ridge_regression", "ridgeCV"]:
        try:
            #VIZ 1 on Actual vs Predicted
            set_up_viz(y_test, pred, title = f"{name}: Actual vs Predicted",
                      rmse=rmse, r2=r2)
            print(f"{name} - Visualization - 1 done")
            #VIZ 2 on Coefficients
            feature_names = x_test.columns
            coef_df = pd.DataFrame({"Feature": feature_names, 
                                    "Coefficient": model.coef_})
            if "lasso" in name:
                coef_df = coef_df.sort_values(by="Coefficient", ascending=True)
            elif "ridge" in name:
                coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values().index)
            plt.figure(figsize=(8,6))
            #creating horizontal bar charts
            plt.barh(coef_df["Feature"], coef_df["Coefficient"], color="green")
            plt.xlabel("Coefficient Value")
            plt.ylabel("Feature")
            plt.title(f"{name} Coefficients")
            plt.tight_layout()
            plt.show()
            print(f"{name} - Visualization -2 done")
        except Exception as e:
            print(f"An error occured during visualization: {e}")
    else:
        print("visualization not designed for this model")
        
#FORECASTING THE NEXT 3 DAYS TEMPERATURE
def forecast_next_3_days(model_name,data_df, n_days = 3):
    predictions_list = []
    data = data_df.copy()
    model = joblib.load(os.path.join(MODELS_DIR, f"{model_name}.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))#use if need to scale

    if "date" in data.columns:
        last_date = pd.to_datetime(data["date"].iloc[-1])
        print("Last Date: ", last_date)
    else:
        last_date = pd.Timestamp.today()
        print("Last Date: ", last_date)
        
    for i in range(n_days):
        # 1. Getting the recent data with respect to date - the last row
        x_ip = data.tail(1).copy()
        x_ip = x_ip.drop(columns=["date"], errors = "ignore")
        #no scaling as already x_test is scaled
        # 3. Predicting the next day max temp - if oct 29 is the x_ip then this pred is for oct 30
        next_pred = model.predict(x_ip)[0] #to get the only value in the array as float and not as an array
        # 4. adding this prediction to the output list
        #predictions_list.append(next_pred)
        next_day = last_date + pd.Timedelta(days = i+1)
        print("Next Day: ", next_day)
        predictions_list.append({
            "date": next_day.strftime("%Y-%m-%d"),
            "predicted_temp" : round(next_pred, 2)
        })
        # 5. preparing the row with lag features that is with the predicted oct 30 value, we are creating a new row for oct30 so that it can be used to predict oct31
        new_row = x_ip.copy()
        new_row["temp_max_lag_14"] = new_row["temp_max_lag_7"]
        new_row["temp_max_lag_7"] = new_row["temp_max_lag_3"]
        new_row["temp_max_lag_3"] = new_row["temp_max_lag_2"]
        new_row["temp_max_lag_2"] = new_row["temp_max_lag_1"] 
        new_row["temp_max_lag_1"] = next_pred #setting the predicted value of oct 30 as temp_max_lag_1 feature 
        #now our new row for oct 31 is ready to predict oct 31
        data = pd.concat([data, new_row], ignore_index=False)
    return pd.DataFrame(predictions_list)