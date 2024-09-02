from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the scaler and model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Mapping dictionaries for encoding
department_mapping = {
    'Sales': 1,
    'Human Resources': 2,
    'Development': 3,
    'Data Science': 4,
    'Research & Development': 5,
    'Finance': 6
}

job_role_mapping = {
    'Sales Executive': 19,
    'Manager': 1,
    'Developer': 2,
    'Sales Representative': 3,
    'Human Resources': 4,
    'Senior Developer': 5,
    'Data Scientist': 6,
    'Senior Manager R&D': 7,
    'Laboratory Technician': 8,
    'Manufacturing Director': 9,
    'Research Scientist': 10,
    'Healthcare Representative': 11,
    'Research Director': 12,
    'Manager R&D': 13,
    'Finance Manager': 14,
    'Technical Architect': 15,
    'Business Analyst': 16,
    'Technical Lead': 17,
    'Delivery Manager': 18
}

env_satisfaction_mapping = {
    'Low': 1,
    'Medium': 2,
    'High': 3,
    'Very High': 4
}

work_life_balance_mapping = {
    'Bad': 1,
    'Good': 0,
    'Better': 2,
    'Best': 3
}

# Mapping for performance rating
performance_mapping = {
    1: 'Good',
    2: 'Excellent',
    3: 'Outstanding'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    emp_department = request.form['EmpDepartment']
    emp_job_role = request.form['EmpJobRole']
    emp_environment_satisfaction = request.form['EmpEnvironmentSatisfaction']
    emp_work_life_balance = request.form['EmpWorkLifeBalance']
    emp_last_salary_hike_percent = float(request.form['EmpLastSalaryHikePercent'])
    experience_years_at_company = float(request.form['ExperienceYearsAtThisCompany'])
    distance_from_home = float(request.form['DistanceFromHome'])
    experience_years_in_current_role = float(request.form['ExperienceYearsInCurrentRole'])
    years_since_last_promotion = float(request.form['YearsSinceLastPromotion'])
    years_with_curr_manager = float(request.form['YearsWithCurrManager'])
    emp_job_level = float(request.form['EmpJobLevel'])

    # Map the input data
    data = pd.DataFrame({
        'EmpDepartment': [department_mapping[emp_department]],
        'EmpJobRole': [job_role_mapping[emp_job_role]],
        'EmpEnvironmentSatisfaction': [env_satisfaction_mapping[emp_environment_satisfaction]],
        'EmpLastSalaryHikePercent': [emp_last_salary_hike_percent],
        'EmpWorkLifeBalance': [work_life_balance_mapping[emp_work_life_balance]],
        'ExperienceYearsAtThisCompany': [experience_years_at_company],
        'DistanceFromHome': [distance_from_home],
        'ExperienceYearsInCurrentRole': [experience_years_in_current_role],
        'YearsSinceLastPromotion': [years_since_last_promotion],
        'YearsWithCurrManager': [years_with_curr_manager],
        'EmpJobLevel': [emp_job_level]
    })

    # Scale the data
    scaled_data = scaler.transform(data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    # Map the numerical prediction back to the performance label
    performance_label = performance_mapping[prediction]

    return render_template('result.html', prediction=performance_label)

if __name__ == '__main__':
    app.run(debug=True)




