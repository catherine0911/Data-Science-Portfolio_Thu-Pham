from langgraph_workflow import run_pipeline

if __name__ == "__main__":
    # Replace with your actual GPT-5 Mini API key
    OPENAI_API_KEY = "sk-xxxxx"
    
    # Run the full pipeline and get the report
    report = run_pipeline(openai_api_key=OPENAI_API_KEY)
    print(report)