# 🎨 Practical Exam: Spectrum Shades LLC — Production Data Analysis

This project focuses on analyzing production data from **Spectrum Shades LLC**, a leading supplier of pigments for concrete applications. The company has recently experienced issues with inconsistent product color quality, resulting in customer dissatisfaction and increased complaints.

The objective of this project is to clean, analyze, and explore production data to uncover potential causes of color variation and provide actionable insights for improving product reliability and quality control.

---

## 🧾 Project Background

Spectrum Shades LLC supplies color pigments used in:
- Decorative concrete  
- Precast concrete  
- Concrete paving products  

Recently, customers reported inconsistencies in color quality. Your role in the data analysis team is to examine production data, identify data issues, calculate key statistics, and investigate the relationships between production factors and final product quality.

This project includes:
- Data cleaning based on strict business rules  
- Supplier performance comparison  
- Filtering for high-impact production cases  
- Statistical & correlation analysis  

---

## 📦 Dataset Description

The analysis uses `production_data.csv`, which contains batch-level production information.

| Column | Description |
|--------|-------------|
| **batch_id** | Batch identifier (no missing values) |
| **production_date** | Date of batch production |
| **raw_material_supplier** | Supplier type (1= national, 2= international) — missing → 'national_supplier' |
| **pigment_type** | ['type_a', 'type_b', 'type_c'] — missing → 'other' |
| **pigment_quantity** | Amount of pigment (1–100 kg) — missing → median |
| **mixing_time** | Mixing duration (minutes) — missing → mean (rounded 2 decimals) |
| **mixing_speed** | 'Low' / 'Medium' / 'High' — missing → 'Not Specified' |
| **product_quality_score** | Quality rating (1–10) — missing → mean (rounded 2 decimals) |

---

## 🧠 Tasks Overview

### **✔ Task 1 — Data Cleaning**
A complete cleaning pipeline was implemented:
- Converted dates
- Mapped supplier codes to labels
- Replaced missing categorical values with specified defaults
- Used median/mean for missing continuous variables
- Normalized pigment types and mixing speed categories

Output: **`clean_data` DataFrame**

---

### **✔ Task 2 — Supplier Comparison**
Calculated:
- Average product quality score  
- Average pigment quantity  

Grouped by raw material supplier type.

Output: **`aggregated_data` DataFrame**

---

### **✔ Task 3 — High-Pigment, Supplier-Specific Analysis**
Filtered production batches:
- Supplier = 2 (international supplier)
- Pigment quantity > 35 kg  

Computed:
- Average pigment quantity  
- Average product quality score  

Output: **`pigment_data` DataFrame**

---

### **✔ Task 4 — Statistical & Correlation Analysis**
Computed:
- Mean & standard deviation of pigment quantity  
- Mean & standard deviation of product quality score  
- Pearson correlation between pigment quantity and product quality score  

Output: **`product_quality` DataFrame**

---

## 🛠️ Skills Demonstrated

- Data cleaning & preprocessing  
- Handling missing values using statistical methods  
- Mapping codes to categorical labels  
- Exploratory data analysis  
- Statistical summary metrics  
- Pearson correlation coefficient  
- Pandas & NumPy data manipulation  
- Insight extraction for real-world business problems  

