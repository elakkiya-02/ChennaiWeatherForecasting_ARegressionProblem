import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap #for XGBoost specifically
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error


BASE_DIR = r"C:\Users\elakkiya\json_tutorial\Chennai_Weather_Prediction"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "reports","plots")

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

#IN PRODUCTION READY DESIGN
def evaluate_model(x_test, y_test, name=None, save_plots=False):
    """
    Evaluate and visualize model performance (supports Linear, Lasso, Ridge, RF, XGB).
    Returns metrics as a dictionary for logging.
    """
    print(f"[INOF] Starting evaluation for model: {name}")
    try:
        model = load_model(name)
    except Exception as e:
        print(f"[ERROR] Failed to load model {name} : {e}")
        return None
    try:
        pred = model.predict(x_test)
    except Exception as e:
        print(f"[ERROR] Prediction Failed for model {name} : {e}")
        return None
    #CALCULATING METRICS
    try:
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        mape = mean_absolute_percentage_error(y_test, pred)*100
        print(f"[SUCCESS] Metrics calculated for {name}")
    except Exception as e:
        print(f"[ERROR] Failed to calculate metrics for {name} : {e}")
        return None
    #VISUALIZATION
    try:
        visualize_model(name=name, y_test = y_test, pred= pred, model = model, x_test=x_test,
                        rmse=rmse, r2=r2, mape=mape,
                        save_plots = save_plots)
        print(f"[SUCCESS] Visualization completed for {name}")
    except Exception as e:
        print(f"[WARNING] Visualization skipped due to error : {e}")
    
    return {'model':name, 'rmse':rmse, "r2":r2, "mape":mape}

def visualize_model(name, y_test, pred, model=None, x_test= None,
                    rmse=None, r2=None, mape=None, 
                    save_plots=False):
    """
    Handles visualization for Linear, Lasso, Ridge, RandomForest, XGB models.
    Produces and optionally saves all key diagnostic plots.
    """
    if name in ["linear_regression", "lasso_regression" ,"lassoCV", "ridge_regression", "ridgeCV"]:
        try:
            fig, ax = plt.subplots(figsize=(8,6))
            ax.scatter(y_test, pred, alpha=0.6, color="red", label="Predictions")
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    color='blue', linestyle='--', label="Fit Line")
            ax.set_title(f"{name}: Actual vs Predicted", fontsize=14, weight="bold")
            ax.set_xlabel("Actual Temperature", fontsize=12, weight="bold")
            ax.set_ylabel("Predicted Temperature", fontsize=12, weight="bold")
            
            if rmse is not None and r2 is not None:
                text = f"RMSE: {rmse:.2f}\nR2: {r2:.3f}"
                ax.text(0.05, 0.95, text, transform=ax.transAxes,
                        fontsize=11, color="Black", verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.5))
            plt.tight_layout()
            plt.show()
            plt.close(fig)
            #coeefficient plot for linear_type models
            if hasattr(model, "coef_") and hasattr(x_test, "columns"):
                coef_df = pd.DataFrame({'Feature':x_test.columns, 
                                        "Coefficient":model.coef_})
                coef_df = coef_df.sort_values(by="Coefficient", ascending=True)
                fig, ax= plt.subplots(figsize=(8,6))
                sns.barplot(data = coef_df, x="Coefficient", y="Feature",
                            ax=ax, palette = "viridis")
                ax.set_title(f"{name}: Coefficeient Importance", fontsize=14, weight="bold")
                plt.tight_layout()
                plt.show()
                plt.close(fig)
        except Exception as e:
            print(f"[ERROR] Visualization failed for {name} linear model: {e}")
    #TREE based models 
    elif "forest" in name or "xgb" in name:
        try:
            residuals = y_test-pred
            if not hasattr(model, "feature_importances_"):
                print("[WARNING] Model has no feature_importances_ atrribute.Skipping feature importance plot")
                feature_imp_df=None
            else:
                feature_imp_df = pd.DataFrame({"Feature":x_test.columns,
                                               "Importance":model.feature_importances_})
                feature_imp_df = feature_imp_df.sort_values(by="Importance",
                                                            ascending=True)
            fig, axes=plt.subplots(4,1, figsize=(10,20))
            axes = axes.flatten()
            
            #1. Actual vs Predicted plot - y_test, pred
            axes[0].scatter(y_test,pred, alpha=0.6, color="red")
            axes[0].plot([y_test.min(), y_test.max()],
                         [y_test.min(), y_test.max()],
                         color="blue", linestyle='--')
            axes[0].set_title(f"{name}: Actual vs Predicted", fontsize=14, weight="bold")
            axes[0].set_xlabel("Actual", fontsize=12, weight="bold")
            axes[0].set_ylabel("Predicted", fontsize=12, weight="bold")
            axes[0].text(0.05, 0.95, f"RMSE:{rmse:.2f}, R2:{r2:.3f}, MAPE:{mape:.2f}%",
                      transform=axes[0].transAxes, fontsize=10, bbox=dict(facecolor="lightgray", alpha=0.5))
            
                      
            #2. Residuals Distribution plot - residuals
            sns.histplot(residuals, bins=30, kde=True, color="coral", ax=axes[1])
            axes[1].axvline(0, color="black", linestyle = '--', lw=2)
            axes[1].set_title("Residuals Distribution", fontsize=14, weight="bold")
    
            #3. Feature Importance plot - feature, importance score
            if feature_imp_df is not None:
                sns.barplot(x="Importance", y="Feature", data=feature_imp_df,
                            ax=axes[2], palette="viridis")
                axes[2].set_title("Feature Importance", fontsize=14, weight="bold")
            else:
                axes[2].text(0.5, 0.5, "No feature Importance available", ha='center')  
            
            #4. Residuals vs Predicted plot - residuals, pred
            axes[3].scatter(pred, residuals, alpha=0.6, color="green")
            axes[3].axhline(0,color='blue', linestyle='--', lw=2)
            axes[3].set_title("Residuals vs Predicted values", fontsize=14, weight="bold")
            axes[3].set_xlabel("Predicted values", fontsize=12)
            axes[3].set_ylabel("Residuals", fontsize=12)

            plt.suptitle(f"{name} Model Diagnostics", fontsize=16, weight='bold')
            plt.tight_layout(rect=[0,0,1,0.96])
            plt.show()
            plt.close(fig)

        except Exception as e:
            print(f"[ERROR] Visualization failed for tree-based models: {e}")
    else:
        print(f"[INFO] Visualization not implemented for model type: {name}")
        

    
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

    needs_scaling = any(text in model_name.lower() for text in ["linear",
                                                                "lasso",
                                                                "rigde"])
    for i in range(n_days):
        # 1. Getting the recent data with respect to date - the last row
        x_ip = data.tail(1).copy()
        x_ip = x_ip.drop(columns=["date"], errors = "ignore")
        #no scaling as already x_test is scaled
        # 3. Predicting the next day max temp - if oct 29 is the x_ip then this pred is for oct 30
        if needs_scaling and scaler is not None:
            x_ip_scaled = scaler.transform(x_ip)
            next_pred = model.predict(x_ip_scaled)[0]
        else:
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
        data = pd.concat([data, new_row], ignore_index=True)
    return pd.DataFrame(predictions_list)