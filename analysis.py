'''
    Report 1
    Rohan Negi, Quazi Nafis
    
    Required Libraries:
    pandas, numpy, tabulate, matplotlib, seaborn
'''

import pandas as pd
import numpy as np
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


def load_csv(path):
    df = pd.read_csv(path)
    return df


def analyze(df, attrs):
    res = {}

    for attr in attrs:
        analysis = {
            "Range": f'[{df[attr].min()}-{df[attr].max()}]',
            "Mean": df[attr].mean(),
            "Mode": df[attr].mode().iloc[0], 
            "Median": df[attr].median(),
            "Standard Deviation": df[attr].std(),
        }
        res[attr] = analysis
    return res


def print_as_table(analysis_obj):
    df_stats = pd.DataFrame(analysis_obj)
    print(tabulate(df_stats, headers="keys", tablefmt="grid"))

def assign_popularity(df):
    # percentiles used to grade entries
    bins = [0, 0.30, 0.50, 0.65, 0.80, 0.90, 1.0]
    labels = ['Very Poor', 'Poor', 'Average', 'Good', 'Very Good', 'Excellent']

    # Now we calculate percentile and then assign a popularity label from above
    df['Popularity'] = pd.qcut(df['shares'], q=bins, labels=labels)

    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("File path not found")
        print("Use command: python report_1.py <file_path>")
        sys.exit()
    file_path = sys.argv[1]
    df = load_csv(file_path)
    
    # 
    # Preprocessing
    # 
    df.columns = df.columns.str.strip()
    df = df[df['n_tokens_content'] > 0]
    df = assign_popularity(df)
    df['log_shares'] = np.log1p(df['shares'])  # Use log1p to avoid issues with log(0)

    # 
    # Analysis
    # 
    SELECTED_ATTRS = ['n_tokens_content', "num_imgs", "global_sentiment_polarity", "num_keywords"]
    full_analysis =  analyze(df, SELECTED_ATTRS)
    

    print("Analysis")
    print_as_table(full_analysis)


    print("Wait for graphs to display...")
    # 
    # Visualization
    #       

    # GRAPH - 1
    # Group by Popularity and calculate the mean of shares
    popularity_avg_shares = df.groupby('Popularity')['shares'].mean()
    
    # Plot
    plt.figure(figsize=(10, 6))
    popularity_avg_shares.plot(kind='bar', color='c')
    plt.title('Average Shares by Popularity Category')
    plt.xlabel('Popularity Category')
    plt.ylabel('Average Shares')
    plt.xticks(rotation=45)
    plt.show()

    
    # GRAPH - 2
    # # Number of Articles by Content Category and Popularity

    categories = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 
              'data_channel_is_bus', 'data_channel_is_socmed', 
              'data_channel_is_tech', 'data_channel_is_world']

    # Reshape data for plotting
    df_melted = df.melt(id_vars=['Popularity'], value_vars=categories, 
                        var_name='Content Category', value_name='Is in Category')
    df_melted = df_melted[df_melted['Is in Category'] == 1]

    # Shortening the category labels for readability
    df_melted['Content Category'] = df_melted['Content Category'].map({
        'data_channel_is_lifestyle': 'Lifestyle',
        'data_channel_is_entertainment': 'Entertainment',
        'data_channel_is_bus': 'Business',
        'data_channel_is_socmed': 'Social Media',
        'data_channel_is_tech': 'Tech',
        'data_channel_is_world': 'World'
    })

    # Plot
    plt.figure(figsize=(12, 6))
    sns.countplot(x='Content Category', hue='Popularity', data=df_melted, palette='coolwarm')
    plt.title('Number of Articles by Content Category and Popularity')
    plt.xlabel('Content Category')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)
    plt.show()


    # GRAPH - 3
    # Scatter Plot of n_tokens_content vs shares

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='n_tokens_content', y='log_shares', hue='Popularity', data=df, palette='coolwarm')
    plt.title('Number of Tokens in Content vs. Log of Shares (Colored by Popularity)')
    plt.xlabel('Number of Tokens in Content')
    plt.ylabel('Log of Number of Shares')
    plt.legend(title='Popularity')
    plt.show()


    # Graph 4
    # Pair Plot of Global Positive/Negative Words and Shares by Popularity
    
    # Selecting the relevant columns for the pair plot
    pairplot_data = df[['global_rate_positive_words', 'global_rate_negative_words', 'shares', 'Popularity']]

    # Create the pair plot
    sns.pairplot(pairplot_data, hue='Popularity', diag_kind='kde', palette='coolwarm')

    # Add a title
    plt.suptitle('Pair Plot of Global Positive/Negative Words and Shares by Popularity', fontsize=16)
    plt.subplots_adjust(top=0.95)  # Adjust the title to fit

    # Show the plot
    plt.show()

    # GRAPH - 5 
    # Scatter Plot of number of images vs shares

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='num_imgs', y='log_shares', hue='Popularity', data=df, palette='coolwarm', s=100)
    plt.title('Number of Images vs. Log of Shares (Colored by Popularity)')
    plt.xlabel('Number of Images')
    plt.ylabel('Log of Number of Shares')
    plt.show()