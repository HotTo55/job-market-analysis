from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display, Markdown
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import os
from langgraph.types import Send
from typing import Annotated, List
import operator
from pydantic import BaseModel, Field
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import json

os.environ["OPENAI_API_KEY"] = ""

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# ===== SCHEMA DEFINITIONS =====

class DataQualityReport(BaseModel):
    missing_values: dict = Field(description="Missing values per column")
    duplicates: int = Field(description="Number of duplicate rows")
    summary: str = Field(description="Summary of data quality issues")


class CleaningTask(BaseModel):
    task_id: str = Field(description="Unique identifier for cleaning task")
    action: str = Field(description="Type of cleaning action (remove_duplicates, fill_missing, etc.)")
    column: str = Field(description="Target column name")
    status: str = Field(description="Status of the task")


class ModelMetrics(BaseModel):
    accuracy: float = Field(description="Model accuracy score")
    feature_importance: dict = Field(description="Top feature importances")
    model_type: str = Field(description="Type of model used")


# Graph state
class State(TypedDict):
    csv_path: str  # Path to input CSV file
    raw_data: pd.DataFrame  # Raw data from CSV
    cleaned_data: pd.DataFrame  # Cleaned data
    cleaning_tasks: list[CleaningTask]  # List of cleaning tasks performed
    data_quality_report: DataQualityReport  # Data quality assessment
    model_metrics: dict  # Model performance metrics
    visualization_path: str  # Path to saved visualization
    analysis_insights: str  # LLM-generated insights from analysis
    final_report: str  # Final comprehensive report
    report_path: str  # Path to saved report


# ===== ORCHESTRATION NODES =====

def load_data_node(state: State):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(state["csv_path"])
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        return {
            "raw_data": df,
            "cleaned_data": df.copy()
        }
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return {"raw_data": None, "cleaned_data": None}


def assess_data_quality_node(state: State):
    """Assess data quality and identify cleaning needs"""
    df = state["cleaned_data"]
    
    # Calculate metrics
    missing_values = df.isnull().sum().to_dict()
    duplicates = df.duplicated().sum()
    
    # Create quality report
    quality_report = DataQualityReport(
        missing_values=missing_values,
        duplicates=duplicates,
        summary=f"Missing values: {sum(missing_values.values())}, Duplicates: {duplicates}"
    )
    
    print(f"Data Quality Assessment: {quality_report.summary}")
    return {"data_quality_report": quality_report}


def clean_data_node(state: State):
    """Clean the data based on quality assessment"""
    df = state["cleaned_data"]
    cleaning_tasks = []
    
    # Remove duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        cleaning_tasks.append(CleaningTask(
            task_id="task_001",
            action="remove_duplicates",
            column="all",
            status="completed"
        ))
    
    # Handle missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
            
            cleaning_tasks.append(CleaningTask(
                task_id=f"task_{len(cleaning_tasks):03d}",
                action="fill_missing",
                column=col,
                status="completed"
            ))
    
    print(f"Data cleaning completed: {len(cleaning_tasks)} tasks performed")
    return {
        "cleaned_data": df,
        "cleaning_tasks": cleaning_tasks
    }


def model_training_node(state: State):
    """Train a predictive model on the cleaned data"""
    df = state["cleaned_data"]
    
    # Prepare data (exclude non-numeric columns and handle target)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("✗ Insufficient numeric columns for modeling")
        return {"model_metrics": {"error": "Insufficient data"}}
    
    # For demo: use first numeric column as target
    X = df[numeric_cols[1:]]
    y = (df[numeric_cols[0]] > df[numeric_cols[0]].median()).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    accuracy = model.score(X_test_scaled, y_test)
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    model_metrics = {
        "accuracy": round(accuracy, 4),
        "model_type": "RandomForestClassifier",
        "feature_importance": {k: round(v, 4) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]},
        "test_samples": len(X_test),
        "train_samples": len(X_train)
    }
    
    print(f"Model trained: Accuracy = {accuracy:.4f}")
    return {"model_metrics": model_metrics}


def visualization_node(state: State):
    """Create visualizations of the analysis"""
    df = state["cleaned_data"]
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analytics Dashboard", fontsize=16, fontweight='bold')
    
    # Plot 1: Data distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:3]
    if len(numeric_cols) > 0:
        df[numeric_cols[0]].hist(ax=axes[0, 0], bins=20, edgecolor='black')
        axes[0, 0].set_title("Distribution of First Numeric Column")
        axes[0, 0].set_ylabel("Frequency")
    
    # Plot 2: Correlation heatmap (simplified)
    if len(numeric_cols) >= 2:
        axes[0, 1].scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.5)
        axes[0, 1].set_title(f"{numeric_cols[0]} vs {numeric_cols[1]}")
        axes[0, 1].set_xlabel(numeric_cols[0])
        axes[0, 1].set_ylabel(numeric_cols[1])
    
    # Plot 3: Data shape info
    axes[1, 0].text(0.5, 0.5, f"Dataset Info\nRows: {df.shape[0]}\nColumns: {df.shape[1]}\nMemory: {df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 0].axis('off')
    
    # Plot 4: Missing data info
    missing_pct = (df.isnull().sum() / len(df) * 100).sum()
    axes[1, 1].text(0.5, 0.5, f"Data Quality\nMissing: {missing_pct:.2f}%\nDuplicates: {df.duplicated().sum()}\nColumns: {len(df.columns)}",
                    ha='center', va='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = "analytics_visualization.png"
    plt.savefig(viz_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"nVisualization saved: {viz_path}")
    return {"visualization_path": viz_path}


def generate_insights_node(state: State):
    """Use LLM to generate insights from the analysis"""
    model_metrics = state.get("model_metrics", {})
    data_quality = state.get("data_quality_report")
    
    # Prepare context for LLM
    context = f"""
    Data Quality Assessment:
    - {data_quality.summary if data_quality else 'N/A'}
    
    Model Performance:
    - Accuracy: {model_metrics.get('accuracy', 'N/A')}
    - Model Type: {model_metrics.get('model_type', 'N/A')}
    - Top Features: {list(model_metrics.get('feature_importance', {}).keys())[:5]}
    
    Dataset Stats:
    - Rows: {state['cleaned_data'].shape[0]}
    - Columns: {state['cleaned_data'].shape[1]}
    """
    
    # Generate LLM insights
    response = llm.invoke([
        SystemMessage(content="You are a data analytics expert. Provide 3-4 key insights from the analysis results."),
        HumanMessage(content=f"Based on this analysis:\n{context}\n\nProvide actionable insights and recommendations.")
    ])
    
    print("✓ Insights generated from LLM analysis")
    return {"analysis_insights": response.content}


def write_report_node(state: State):
    """Generate and write the final analytics report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build report
    report = f"""# Analytics Report
Generated: {timestamp}

## Executive Summary
This report presents the results of a comprehensive data analysis and modeling pipeline.

## 1. Data Loading
- **File**: {state.get('csv_path', 'N/A')}
- **Rows**: {state['raw_data'].shape[0]}
- **Columns**: {state['raw_data'].shape[1]}

## 2. Data Quality Assessment
- **Missing Values**: {state['data_quality_report'].summary}
- **Issues Found**: {len(state.get('cleaning_tasks', []))}

## 3. Data Cleaning
Performed the following cleaning operations:
"""
    
    for task in state.get("cleaning_tasks", []):
        report += f"\n- {task.action} on {task.column}: {task.status}"
    
    report += f"""

## 4. Model Training Results
- **Accuracy**: {state['model_metrics'].get('accuracy', 'N/A')}
- **Model Type**: {state['model_metrics'].get('model_type', 'N/A')}
- **Training Samples**: {state['model_metrics'].get('train_samples', 'N/A')}
- **Test Samples**: {state['model_metrics'].get('test_samples', 'N/A')}

### Top Feature Importance
"""
    
    for feature, importance in list(state['model_metrics'].get('feature_importance', {}).items())[:5]:
        report += f"\n- {feature}: {importance}"
    
    report += f"""

## 5. Visualization
- **Location**: {state.get('visualization_path', 'N/A')}

## 6. Key Insights
{state.get('analysis_insights', 'N/A')}

## 7. Recommendations
Based on the analysis, consider:
1. Implementing the identified feature importance scores in production models
2. Monitoring data quality metrics regularly
3. Retraining the model with new data periodically

---
End of Report
"""
    
    # Save report to file
    report_path = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Report written: {report_path}")
    return {
        "final_report": report,
        "report_path": report_path
    }


# ===== BUILD WORKFLOW =====

analytics_workflow = StateGraph(State)

# Add nodes
analytics_workflow.add_node("load_data", load_data_node)
analytics_workflow.add_node("assess_quality", assess_data_quality_node)
analytics_workflow.add_node("clean_data", clean_data_node)
analytics_workflow.add_node("train_model", model_training_node)
analytics_workflow.add_node("visualize", visualization_node)
analytics_workflow.add_node("generate_insights", generate_insights_node)
analytics_workflow.add_node("write_report", write_report_node)

# Add edges (linear workflow)
analytics_workflow.add_edge(START, "load_data")
analytics_workflow.add_edge("load_data", "assess_quality")
analytics_workflow.add_edge("assess_quality", "clean_data")
analytics_workflow.add_edge("clean_data", "train_model")
analytics_workflow.add_edge("train_model", "visualize")
analytics_workflow.add_edge("visualize", "generate_insights")
analytics_workflow.add_edge("generate_insights", "write_report")
analytics_workflow.add_edge("write_report", END)

# Compile the workflow
analytics_pipeline = analytics_workflow.compile()

# Display the workflow
print("\n=== Analytics Workflow ===")
display(Image(analytics_pipeline.get_graph().draw_mermaid_png()))


# ===== EXECUTION =====

def run_analytics_pipeline(csv_path: str):
    """Execute the analytics pipeline on a CSV file"""
    print(f"\nStarting analytics pipeline on: {csv_path}\n")
    
    initial_state = {
        "csv_path": csv_path,
        "raw_data": None,
        "cleaned_data": None,
        "cleaning_tasks": [],
        "data_quality_report": None,
        "model_metrics": {},
        "visualization_path": "",
        "analysis_insights": "",
        "final_report": "",
        "report_path": ""
    }
    
    # Execute pipeline
    result = analytics_pipeline.invoke(initial_state)
    
    print(f"\nPipeline completed successfully!")
    print(f"Report saved to: {result['report_path']}")
    print(f"Visualization saved to: {result['visualization_path']}\n")
    
    return result


# Example usage (uncomment and update with your CSV path)
if __name__ == "__main__":
    # Update this path to your CSV file
    csv_file = "C:/Users/sberry5/Documents/teaching/courses/inf/performanceReviewData.csv"  # Change this to your CSV file path
    
    if os.path.exists(csv_file):
        result = run_analytics_pipeline(csv_file)
        
        # Display the final report
        print("\n" + "="*80)
        print("FINAL REPORT")
        print("="*80)
        display(Markdown(result["final_report"]))
    else:
        print(f"CSV file not found: {csv_file}")
        print("Please provide a valid CSV file path.")