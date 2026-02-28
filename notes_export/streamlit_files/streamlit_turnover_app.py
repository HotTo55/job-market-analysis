import pandas as pd
import streamlit as st
import joblib

model = joblib.load('/Users/sberry5/Documents/teaching/UDA/code/streamlit_examples/lasso_model.joblib')

st.title('Employee Turnover Prediction')
st.write('Use the following form to submit data for a prediction.')

# Create form for user input
form = st.form(key='my_form')

age = form.number_input('Age of employee', min_value=22, max_value=75, value=25)
numberPriorJobs = form.number_input('Number of Prior Jobs', min_value=0, max_value=10, value=0)
proportion401K = form.number_input('Proportion of 401K', min_value=0.0, max_value=1.0, value=0.0)
startingSalary = form.number_input('Starting Salary', min_value=0, max_value=200000, value=0)
currentSalary = form.number_input('Current Salary', min_value=0, max_value=200000, value=0)
performance = form.number_input('Performance', min_value=0, max_value=10, value=0)
monthsToSeparate = form.number_input('Months to Separate', min_value=0, max_value=100, value=0)
workDistance = form.number_input('Work Distance', min_value=0, max_value=100, value=0)
department_1 = form.checkbox('Department 1')
department_2 = form.checkbox('Department 2')
department_3 = form.checkbox('Department 3')

submit_button = form.form_submit_button(label='Submit')

if submit_button:
    data = {
        'age': age,
        'numberPriorJobs': numberPriorJobs,
        'proportion401K': proportion401K,
        'startingSalary': startingSalary,
        'currentSalary': currentSalary,
        'performance': performance,
        'monthsToSeparate': monthsToSeparate,
        'workDistance': workDistance,
        'department_1': department_1,
        'department_2': department_2,
        'department_3': department_3
    }
    
    
    df = pd.DataFrame(data, index=[0])
    
    prediction = model.predict(df)
    
    st.write(f'The predicted probability of employee turnover is {prediction[0]:.2f}')
    if prediction[0] > 0.5:
        st.write('The model predicts that the employee will leave the company.')
    else:
        st.write('The model predicts that the employee will stay with the company.')

# Display the form

#python3 -m streamlit run streamlit_turnover_app.py