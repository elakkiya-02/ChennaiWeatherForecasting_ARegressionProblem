# ChennaiWeatherForecasting_ARegressionProblem
A regression based model to forecast Chennai's next 3 days maximum temperature using historical weather data from OpenMeteo and lag features

The objective of this regression project is to predict the next_day's maxmium temperature for Chennai using historical weather data from OpenMeteo. We apply a supervised regression approach, where past temperature values, created lag features are used to train each model.
1. Linear Regression Model.
2. RidgeCV
3. LassoCV
4. Random Forest
5. XG Boost

The trained models then perform recursive-day forecasting to predict the maximum temperature for the next 3 days based on the previous predicted values.
LINEAR REGRESSION
This is going to be a lag-based forecasting and is suitable for short term predictions like 1-3days.
