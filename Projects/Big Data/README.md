# Stroke Risk Analysis Using PySpark (Big Data Project)

## 📌 Overview
This project analyzes a healthcare dataset to identify stroke-related patterns using **PySpark RDDs**.  
It demonstrates the ability to work with distributed data structures, apply transformations, perform aggregations, and generate insights from large-scale data.

The analysis addresses three main research tasks using pure RDD operations:

1. **Stroke rate per work type**
2. **Stroke rate by (Residence Type × Ever Married) groups**
3. **Stroke percentage among patients aged 65+, grouped by smoking status and gender**

This project was originally developed as a group assignment and later refined for portfolio purposes.

---

## 🧰 Technologies Used
- **Python 3**
- **Apache Spark / PySpark**
- **RDD transformations & actions**
- `map`, `filter`, `reduceByKey`, `join`, `sortByKey`, etc.

---

## 📂 Dataset
The dataset used is:

**`healthcare-dataset-stroke-data.csv`**

It contains demographic and health-related attributes such as:
- Gender  
- Age  
- Hypertension  
- Heart disease  
- Marital status  
- Work type  
- Residence type  
- Smoking status  
- Stroke (0/1)
---

## 🧪 Project Tasks & Methodology
Task 1 — Stroke Rate per Work Type
Task 2 — Stroke Rate by Residence Type and Marital Status
Task 3 — Stroke Percentage for Seniors (65+), Grouped by Smoking Status & Gender
