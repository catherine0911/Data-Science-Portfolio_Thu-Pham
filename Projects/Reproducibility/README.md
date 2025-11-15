# Final Portfolio — Reproducibility and Model Deployment

## 1. Snapshot: What’s in This Repo

This repository documents my learning journey in reproducibility and model deployment in data science.  
Over 7 weeks, I learnt several new tools such as Makefile, Docker, Fast API, etc to build an end-to-end reproducible workflow, from cleaning and analyzing data to training, deploying, and sharing a model through an API and Streamlit interface.

The work is divided into two connected projects, each focusing on one part of the reproducible pipeline:

| Project                          |  Tools                      | Output                       |
| -------------------------------- | -------------------------- | ---------------------------- |
| **Project 1 – DataProcessor**    | Python, Pytest, Makefile   | Tested statistical processor |
| **Project 2 – Model Deployment** | FastAPI, Docker, Streamlit, Makefile | Deployed text classifier on API and Streamlit app |

- Moreover, each week I reflected on what I learned and documented it in weekly report: [repository](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/reports)


### Repo Structure:
```
PORTFOLIO-25-26-CATHERINE0911/
│
├── dat/
│   ├── raw/                     # Project 2: raw data
│   ├── processed/               
│   └── sample.csv               # Project 1: sample dataset
│
├── models/                      
│
├── src/
│   ├── processor.py             # Project 1: DataProcessor class
│   ├── processor_demo.py        # Project 1: demo script
│   ├── classify.py              # Project 2: train model
│   ├── fetch_data.py            # Project 2: download data
│   ├── scrape_books.py          # Project 2: web scraping
│   ├── clean_books.py           # Project 2: data cleaning
│   ├── api.py                   # Project 2: FastAPI service
│   └── app.py                   # Project 2: Streamlit frontend
│
├── tests/
│   ├── test_processor.py        
│   └── test_smoke.py
│
├── Dockerfile                   # Project 2: container setup
├── Makefile                     # Automation for both projects
├── requirements1.txt            # Dependencies for Project 1
├── requirements2.txt            # Dependencies for Project 2
└── README.md                    # Final portfolio (this file)
```

---

## 2. Project 1: DataProcessor

### Goal
To design a reproducible data analysis tool that calculates descriptive statistics, detects outliers, and summarizes data. This is tested and automated.

### Components
- **DataProcessor:** defines function `DataProcessor` class (object-oriented programming) for mean, median, variance, stdev, quartiles, IQR, and outlier removal with Docstring formats in NumPy style .  
  - [processor.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/src/processor.py)
- **Pytest:** automates testing
  - [test_processor.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/tests/test_processor.py)
- **Demonstration:** how the processor works on sample.csv  
  - [processor_demo.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/src/processor_demo.py)
- **Requirements:** dependencies
  - [requirements1.txt](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/requirements1.txt)

### How to run
```bash
make install1     # Set up environment
make run          # Run demo
make test         # Run pytest
make clean_cache  # Clean cache
#OR run everything once
make project1    
```
### Expected Output
![Output for Project 1](<reports/Week 6/Screenshots/image-2.png>)

### Strengths and Limits
- Strengths:
  - The code structure is clean and easy to read with the Docstring formats in NumPy style
  - Pytest tests makes the functions more reliable and easier to validate.
  - The results are easy to verify as the formulas are built manually by using mathematic definition rather than relying on external libraries (mean, variance, IQR, etc)

- Limits:
  - The project is simple and does not reflect real-world data which normally also includes missing values, and mixed data types.
  - There is only one way to remove outliers. The project does not give other options for other people to choose how they want to remove outliers.

- Trade off:
  - I intentionally want to keep this code simple and transparent to make the project easy to understand, but I have to trade off with the flexibility. For example, I round all the results to 2 decimals, so if other users expect full precision or other rounding rules, they have to manually change it in original code.
  - I used a fixed sample dataset which can make results consistent, but it limits how well the code performs to different kinds of data

---
## 3. Project 2: Text classification
### Goal
To build and deploy a reproducible text classification pipeline that scrapes book data, trains a Naive Bayes model, and serves predictions through FastAPI and Streamlit.

### Components
- **Data & Model Scripts:** data cleaning, preprocessing, and text classification model using Naive Bayes.  
  - [classify.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/src/classify.py)
- **Automation with Makefile:** automates every step from setup, fetching data, cleaning, training, to deployment
  - [Makefile](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/Makefile)
- **API Service:**  FastAPI app serves model predictions and health checks
  - [api.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/src/api.py)
- **Docker:** containerizes the API to run consistently on any system.
  - [Dockerfile](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/Dockerfile)
- **Streamlit:**  web interface to test predictions interactively. 
  - [app.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/src/app.py)

### How to run
Run the code below on WSL/Linux:
```bash
make project2                 # Setup env → fetch → scrape → clean → train
make run-api & make run-app      # Run FastAPI service and Streamlit App (optional)
make docker-run               # Build Docker image and run the container
```
Then visit:
```
FastAPI: http://localhost:8000/docs
Streamlit: http://localhost:8501
```
### Expected output
![Output for Project 2](<reports/Week 6/Screenshots/Screenshot 2025-10-10 101919.png>)

### Strengths and Limits
- Strengths:
  - The project contains a full reproducible pipeline: scraping -> cleaning -> training -> testing
  - I set a global random seed, so anyone running the classifier for the same dataset will get the same train-test split, the same TF-IDF vocabulary, and the same model performance result.
  - Each step in the pipeline (scraping, cleaning, training) is separated into independent code, which makes debugging easier.
  - Docker is added to ensure the environment is fully reproducible, regardless of operating system or local dependencies.
- Limits:
  - Scraping can get error easily if the website structure changes, which affects long-term reproducibility. For example, if the website changes its tags, the scraping step breaks and reproducibility is lost.
  - Although global seed helps model training deterministic, I found out in the scraping step, the raw HTML files are timestamped and may change over time. This means that while model training is reproducible on the same dataset, the full pipeline is not fully deterministic due to changes in the dataset.
- Trade off:
  - I chose not to freeze the dataset because I want to scrape the most up-to-date data. The trade-off is reduced reproducibility, because every time I re-run the scraping step, it produces a slightly different dataset.
  - I chose Naive Bayes because it is lightweight and easy to deploy, but more advanced models should be implemented if the dataset is more complex.
  - I used SQLite to improve reproducibility, but it increases setup steps compared to a single CSV.

## 4. Why These Components Matter for Reproducibility
Each tool and process contributes to make my work consistent, testable, and shareable:
  - Git and Version Control: This helps to track every change, allowing people to roll back the work:
    - Evidence - Commit hash(updating week 4 report): e7de36ce1473f93086cf3f498899ceae7bd3fd9d
  - Virtual Environments & Requirements: This contains required dependencies for each project in text file, which ensures an easy and consistent setups for everyone.
    - Evidence: [requirements2.txt](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/requirements2.txt)
  - Makefile Automation: This helps other people to save time and avoid mistakes when they want to reproduce my work with just few lines of code
   - Evidence: [Makefile](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/Makefile)
  - FastAPI & Streamlit Deployment: These 2 tools make the project results easy to test and share. FastAPI gives us reproducible endpoints, while Streamlit gives a interactive interface to deploy the project.
    - Evidence:[api.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/src/api.py)
  - Docker: It packages the entire environment, including dependencies and system settings, so it can run the same on any computer.
    - Evidence: [Dockerfile](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/Dockerfile)
  - Documentation: The explanation in README helpd other people easily understand the project summary, steps to reproduce the work, which is very essential for reproducibility.
    - [README of Week3.md](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/reports/Week%203/README.md)
  - Determinism: A global random seed was added to classify.py to ensure that everytime I or someone else run the code, it produces the same train-test split and model parameters for the same dataset.  Instead of only setting random_state in the train_test_split, global random seed will help any randomness in entire pipeline be controled,  making results traceable and reproducible.
    - Evidence: [classify.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/fe707eb4251de22ed8fc4ed786f2292ee73c3bb8/src/classify.py)

Together, these components can help to ensure that projects can be rebuilt, tested, and verified in exactly the same way by anyone.

## 5. A few small deep-dives 
- A few things I explored and learned while working on these projects:
  - I explored extra improvements in Project 2 beyond the lecture pipeline. For example, I used SQLite instead of CSV for storing cleaned data so it’s easier to query later, and switched the model from Logistic Regression to Naive Bayes.
  - Evaluation: I also added a confusion matrix to visualize which book categories were often misclassified
    - Artifact: [classify.py](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/72b4e0e21884e81bbeaef33eca5b17b2fae1d883/src/classify.py)
  - Makefile: I tried to integrate all steps, including virtual environment setup, dependency installation, running scripts, testing, and cleaning cache. This allows me to reproduce both projects with a single `make project1` command.
    - Artifact: [Makefile](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/Makefile)
  - Docker: I built a Docker image for Project 2 to ensure the API can run reliably across different environments.

## 6. Reflection 
- Peer Feedback: I received the feedback that my files were scattered and reports lacked links, making it hard to navigate. So, I reorganized the folder structure and added report links to make everything easier to find.
  - Example: From week 4, I added links to my report [week04.md](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/reports/Week%204/week04.md)
  - Commit update folder: 1ee8e0916bc2f3fae259da9acd86f55aebffa1e5
- Self-reflect: Looking back at my Week 3 DataProcessor project, I realized I could automate everything using Makefiles with one command.
- Professor feedback: Professor suggested including output files instead of too many screenshots. From Week 3, I focused on adding more relevant files.
  - Evidence: [week03.md](https://github.com/tcsai/portfolio-25-26-catherine0911/blob/a9ee3ab3ed05a76ad6c81ae2315e510b22280834/reports/Week%203/week03.md)

## 7. Gen AI
I used GenAI tools, mainly ChatGPT:
  - It helped me debug my Makefile when my environment kept closing after make install. The suggestion to define venv Python and pip for each project completely fixed that issue.
  - It also gave me feedback on my documentation like suggesting I use tables and format for title, subtitle to make my README.md easier to read.
  - Another helpful idea was creating a processor_demo.py and sample dataset file for testing Project 1.
  - It explained determinism and how to improve it by adding global seed instead of using seed for the train test split function only. However, I found out the confusion matrix is still changed everytime I re-run the code. I figured out the problem was not training step, but scraping step as the data is collected based on timestamps. Therefore, it's always neccessary to verify AI-instructed method instead of assuming it can definitely solve the problem.
  - One risk I observed throughout the journey is AI's instructions to debug or code might not always run correctly. Sometimes AI suggestions made things more complicated or didn’t run properly, so I always double-check with other tech forums.

## 8. Next Steps
If I continued developing this as a real project, I will try to:
  1. **Improve the Streamlit interface to allow users to upload their own data and see predictions interactively.** This will make the tools more powerful and practical in the real world. However, I need to improve the cleaning step as the real datasets usually have more noise and mixed data types. I also want to deploy different machine learning models so that users can see which model works the best for their data.
  3. **Deploy the API to a cloud service like AWS or Render for public access.** Now, everything runs locally or inside Docker. Therefore, deploying to Cloud service can make the system accessible to others. It can also help to test whether my API, dependencies and Docker setup are truly reproducible outside my machine.
  3. **Set up versioning for data and models to better track experiments and performance changes.**  Because the data can change depending on the time of scraping data, dataset is not fully deterministic. By saving each scraped dataset as a versioned CSV or SQLite file, I can always trace back which data version produced which results. Versioning the model files also keeps the whole pipeline transparent and reproducible over time.
  4. **Collect feedback and prediction logs to monitor its real accuracy and retrain the model when needed.** Logging user inputs and predictions (with privacy considerations and users' consent) would help me understand how the model performs in practice. This will help future retraining, as I can identify patterns in the model’s mistakes, and labels which are frequently misclassified.

In summary, this portfolio shows my study progress to make data projects reproducible from data fetching to model deployment. Through these projects and weekly report, I understood and know how to use tools like Git, Makefiels, and Docker to help my work organized, consistent and reproducible.
