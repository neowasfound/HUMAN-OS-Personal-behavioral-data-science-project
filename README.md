# HumanOS Intelligence
### Understanding how I work, rest and procrastinate.

**A Personal Data Science, Statistics & Machine Learning Project**

HumanOS Intelligence started with something pretty simple: tracking my own daily life.

I wanted to understand my routines better, especially how my mood, energy, sleep, stress and other everyday behaviours relate to my academic productivity.
Instead of using a dataset downloaded from the internet, I decided to use something much closer to home: **my own personal tracking data, collected from my actual day-to-day experiences.**

What started as personal tracking eventually turned into a full data science project, combining statistics, machine learning, data visualisation and web development.

The idea is simple: take the data I collected about myself and see what it can actually tell me.

---

## The Research Question

**To what extent are daily behavioural and wellbeing indicators associated with perceived academic productivity and deep-work quality?**

I wanted to investigate whether there were noticeable relationships between different aspects of my daily life and how productive I felt.

Some of the variables I explored include:

- Mood
- Energy levels
- Sleep duration
- Stress levels
- Exercise
- Social interactions
- Meal quality
- Perceived productivity
- Deep-work quality

These aren't observations from a group of participants. They're records from my own daily tracking.

---

## What I Wanted to Achieve

The project gave me an opportunity to take something personal and approach it from a technical perspective.

My main objectives were to:

- Clean and organise my personally collected data.
- Explore patterns in my daily routines and behaviours.
- Investigate relationships between wellbeing and perceived productivity.
- Apply statistical methods to test those relationships.
- Build and evaluate machine learning models.
- Develop an interactive dashboard to bring the results together.

---

## The Technical Side

I used a combination of tools and libraries from my Computer Science and Statistics background.

| Area | Tools |
|---|---|
| Programming | Python | 
| Data manipulation | Pandas, NumPy | Claude AI |
| Statistical analysis | SciPy, Statsmodels |
| Machine learning | Scikit-learn | Claude AI
| Visualisation | Matplotlib, Plotly |
| Web development | Flask, HTML, CSS |

---

## What Goes Into the Project?

### 01. Data Preparation

Before doing any analysis, I needed to understand what I was working with.

This involved auditing the dataset, handling missing values, checking duplicates and preparing the recorded observations for analysis.

### 02. Data Exploration

This is where things started getting interesting.

I explored how different variables behaved over time and used visualisations to look for patterns in my mood, energy, stress, sleep and productivity.

### 03. Statistical Analysis

I moved beyond simply looking at charts and started testing relationships in the data.

Some of the methods I used include:

- Descriptive statistics
- Spearman correlation
- Hypothesis testing
- Group comparisons
- Multiple comparison adjustments
- Regression modelling
- Regression diagnostics
- Sensitivity analysis

### 04. Machine Learning

I also explored whether machine learning could identify useful patterns in my recorded behavioural and wellbeing indicators.

This involved developing predictive models, comparing their performance and investigating feature importance.

### 05. Building the Dashboard

I wanted somewhere to bring all the work together.

So I developed a Flask web application with interactive visualisations and dedicated sections for exploring the data, viewing statistical results and examining machine learning models.

The dashboard is designed to make the analysis easier to explore without having to go through every individual Python script.

---


## A Few Things Worth Keeping in Mind

This project is personal, and so are the limitations of the data.

A few things I had to keep in mind while working with it:

- The observations represent one individual, so the findings cannot automatically be generalised to other people.
- Productivity and deep-work scores are subjective assessments.
- A correlation doesn't mean one variable causes another.
- Some periods have missing observations because I stopped tracking for a while. Those gaps shouldn't be interpreted as days when nothing happened.
- Other factors, such as academic deadlines and workload, may influence the relationships I observed.

I think understanding these limitations is just as important as presenting the results themselves.

---

## Getting Started

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd HumanOS-Intelligence
```

Install the required libraries:

```bash
py -m pip install pandas numpy matplotlib scipy statsmodels scikit-learn openpyxl scikit-posthocs Flask plotly
```

The application expects a locally available cleaned dataset at:

```text
data/processed/humanos_cleaned.csv
```

The original personal dataset is not included in this public repository.

Once an appropriate local dataset is available, run:

```bash
py app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

## What I Took Away From This

One of the things I enjoyed most about this project was being able to connect concepts from Computer Science and Statistics to something I actually experience every day.

I got to work through the entire process, from collecting and preparing data to investigating relationships, building models and developing an application to present the results.

It also reminded me that data doesn't always tell the story you expect. Sometimes the interesting part isn't finding a strong relationship, but understanding why a relationship exists, what might be influencing it and what the data cannot tell you.

**HumanOS Intelligence is essentially my attempt to turn personal tracking into a data science investigation.**

---

**Project:** HumanOS Intelligence  
**Focus:** Personal Data Science, Behavioural Analytics, Statistics & Machine Learning  
**Built with:** Python, Flask and Scikit-learn

*All underlying observations originate from my own personal daily tracking.*
