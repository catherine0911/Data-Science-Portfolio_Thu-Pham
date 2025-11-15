# 🛒 Practical Exam: Grocery Store Sales (SQL Project)

This project explores and cleans product-level data from FoodYum, a grocery store chain in the United States.  
As food costs rise, FoodYum wants to ensure that all product categories include items across a range of prices so customers with different budgets can be served. The dataset includes information about product type, price, weight, units sold, and stock location.

The goal of this project is to:
- Identify missing values in key variables  
- Clean the dataset according to business rules  
- Explore product price ranges by category  
- Select relevant subsets of interest for further analysis  

---

## 📦 **Dataset Description**

The analysis uses data from the `products` table, which contains product information from the loyalty program for the previous full year.

| Column | Description |
|--------|-------------|
| **product_id** | Unique identifier of the product (no missing values) |
| **product_type** | Category: Produce, Meat, Dairy, Bakery, Snacks (missing → “Unknown”) |
| **brand** | One of 7 brands (missing → “Unknown”) |
| **weight** | Weight in grams (missing → median weight) |
| **price** | Price in USD (missing → median price) |
| **average_units_sold** | Monthly average units sold (missing → 0) |
| **year_added** | Year added to stock (missing → 2022) |
| **stock_location** | Warehouse location: A/B/C/D (missing → “Unknown”) |

---

## 🧠 **Tasks Overview**

### **✔ Task 1 — Count Missing `year_added` Values**
A query was written to identify how many products were missing the `year_added` value due to a system bug in 2022.

### **✔ Task 2 — Clean the Data (Without Modifying Original Table)**
Cleaned output includes:
- Replacing missing values
- Standardizing formats
- Using median values for numeric fields
- Ensuring categories match expectations

### **✔ Task 3 — Price Range by Product Type**
SQL query that returns minimum and maximum price for each `product_type`.

### **✔ Task 4 — Filter Meat & Dairy Products**
Query to return:
- `product_id`
- `price`
- `average_units_sold`  
for Meat and Dairy products with more than 10 units sold per month.

## 🛠️ **Skills Demonstrated**

- SQL data cleaning  
- Handling missing values  
- Conditional transformations (CASE statements)  
- Using medians via window functions  
- Data validation  
- Filtering and aggregation  
- Understanding business rules and converting them into queries  
