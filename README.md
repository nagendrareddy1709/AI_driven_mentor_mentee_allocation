# AI Mentor–Mentee Allocation Platform

## Overview

This project is an AI-powered mentor allocation platform designed to help organizations efficiently match mentors with mentees based on their course specialization and expertise.

The system uses **AI-based text similarity (TF-IDF)** and **smart load balancing** to ensure that mentees are assigned to the most suitable mentor while keeping mentor workloads evenly distributed.

This platform is particularly useful for:

* Universities
* Charity organizations
* Mentorship programs
* Academic advisory systems

---

## Key Features

### Intelligent Mentor Matching

Uses TF-IDF based similarity to match mentee courses with mentor specializations.

### Smart Load Balancing

Prevents mentors from being overloaded by considering the number of mentees already assigned.

### Interactive Dashboard

Displays mentor-mentee allocation, statistics, and distribution charts.

### CSV Data Upload

Users can upload mentor and mentee datasets directly through the interface.

### Email Notification System

Automated email notifications are sent to mentors and mentees after assignment.

### Expandable Specialization & Course Lists

Admins can add new specializations and courses dynamically.

---

## Technologies Used

* Python
* Streamlit
* Pandas
* Scikit-learn
* Matplotlib
* SMTP Email Automation

---

## Project Structure

mentor_matching_app/

```
app.py
mentors.csv
mentees.csv
requirements.txt
README.md
```

---

## Running the Application Locally

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the application:

```
streamlit run app.py
```

3. Open in browser:

```
http://localhost:8501
```

---

## Deployment

This application is deployed using **Streamlit Cloud**, which provides a secure HTTPS link that allows users within the organization to access the dashboard through a web browser.

---

## Future Enhancements

* Mentor feedback and rating system
* Engagement tracking between mentors and mentees
* Authentication for secure user access
* Database integration (PostgreSQL / Supabase)
* Advanced NLP matching models

---

## Disclaimer

The dataset used in this repository is for **demonstration and educational purposes only**. Real mentor and mentee data will be provided by the organization during deployment.

---

## Author

Developed as part of an AI-driven mentorship allocation initiative to support educational and charity mentoring programs.
