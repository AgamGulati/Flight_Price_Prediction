"""
Generates EDA and Model Performance Visualizations for the Project Report.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    train_df = pd.read_csv('train.csv')
    plt.rcParams['figure.dpi'] = 200
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Price distribution by Flight Class
    sns.histplot(data=train_df, x='price', hue='flight_class', bins=50, kde=True, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('1. Bimodal Ticket Price Distribution by Flight Class', fontsize=11, fontweight='bold')
    axes[0, 0].set_xlabel('Ticket Price')
    axes[0, 0].set_ylabel('Frequency')

    # Plot 2: Simpson's Paradox (Mean Price by Days Left & Flight Class)
    days_trend = train_df.groupby(['days_left', 'flight_class'])['price'].mean().reset_index()
    sns.lineplot(data=days_trend, x='days_left', y='price', hue='flight_class', ax=axes[0, 1], palette='magma', linewidth=2)
    axes[0, 1].set_title("2. Simpson's Paradox: Price vs. Days Left by Class", fontsize=11, fontweight='bold')
    axes[0, 1].set_xlabel('Days Left Before Departure')
    axes[0, 1].set_ylabel('Mean Ticket Price')

    # Plot 3: Ticket Price across Stops and Class
    sns.boxplot(data=train_df, x='stops', y='price', hue='flight_class', ax=axes[1, 0], palette='Set2')
    axes[1, 0].set_title('3. Ticket Price Distribution Across Stops & Class', fontsize=11, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Stops')
    axes[1, 0].set_ylabel('Ticket Price')

    # Plot 4: Benchmark Comparison (CV R2 Score)
    model_res = pd.DataFrame({
        'Model': [
            'Decision Tree', 'Ridge (L2)', 'Lasso (L1)', 'Linear Reg (OLS)',
            'HistGB (Deep)', 'HistGB (Base)', 'Gradient Boosting',
            'Extra Trees', 'Random Forest', 'Stacked Ensemble'
        ],
        'R2': [0.89442, 0.90274, 0.90393, 0.90409, 0.91862, 0.92097, 0.92371, 0.92373, 0.92666, 0.92800]
    })
    sns.barplot(data=model_res, x='R2', y='Model', ax=axes[1, 1], palette='Blues_r')
    axes[1, 1].set_xlim(0.88, 0.935)
    axes[1, 1].set_title('4. 5-Fold Cross-Validation R2 Score Across Models', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('5-Fold CV R2 Score')
    axes[1, 1].set_ylabel('')
    for i, v in enumerate(model_res['R2']):
        axes[1, 1].text(v + 0.0005, i, f"{v:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('eda_and_model_results.png', bbox_inches='tight')
    print("Generated 'eda_and_model_results.png' successfully.")

if __name__ == '__main__':
    main()