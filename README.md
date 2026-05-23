# Financial-News-Sentiment-Analyzer
```markdown
# Financial News Sentiment Analyzer

## Project Overview

This project implements an NLP-based sentiment analyzer designed for financial news headlines. It's a powerful tool that can classify financial news as positive, negative, or neutral for a given stock, providing valuable insights for portfolio managers, hedge funds, and investment banks. By integrating cutting-edge AI (Natural Language Processing and Hugging Face Transformers), this system aims to enhance decision-making in financial markets.

## Why Sentiment Analysis for Finance?

Sentiment analysis of news headlines offers significant advantages for investment professionals:

*   **Predictive Power**: Forecast market movements by interpreting the emotional tone of news, indicating potential stock price changes.
*   **Risk Management**: Identify potential risks early by monitoring news for negative sentiment or unanticipated events, allowing for proactive strategy adjustments.
*   **Advanced Trading Strategies**: Integrate sentiment data into trading algorithms to improve decision-making and potentially achieve higher returns.
*   **Real-Time Analysis**: Leverage advancements in NLP and machine learning for immediate reactions to market news.

These insights enable informed decisions, better navigation of market fluctuations, and optimized investment strategies.

## Features

*   **Pre-trained Model Deployment**: Utilizes the FinBERT model, a 110-million parameter transformer model specifically trained on financial text.
*   **NLP Pipeline**: Performs text classification on financial data with confidence scoring for each prediction.
*   **Production-Grade Design**:
    *   Includes sample size warnings for low data confidence.
    *   Detects potential "contrarian" investment opportunities based on strong negative sentiment.
    *   Provides data confidence ratings.
*   **Quant + AI Integration**: Merges AI sentiment signals with quantitative financial metrics (e.g., Sharpe Ratio) to create a weighted decision framework.
*   **Actionable Portfolio Signals**: Generates clear `BUY`, `SELL`, or `HOLD` recommendations, including suggestions to `INCREASE WEIGHT` or `REDUCE WEIGHT` based on combined analysis.
*   **Interactive Visualizations**: Presents sentiment distribution, confidence levels, and combined decision frameworks through clear plots.

## Installation

To set up the project, you need to install the necessary Python libraries. It's recommended to use a virtual environment.

```bash
!pip install transformers torch newspaper3k sentencepiece pandas matplotlib
```

*   `transformers`: Hugging Face's library for pre-trained AI models.
*   `torch`: PyTorch, the deep learning framework powering the models.
*   `newspaper3k`: A library for scraping news articles from the web.
*   `sentencepiece`: A tokenizer used by many language models.
*   `pandas`: For data manipulation and analysis.
*   `matplotlib`: For plotting and visualization.

## Usage

### 1. Load the FinBERT Model

The core of the sentiment analysis is the FinBERT model, which is specialized in understanding financial language.

```python
from transformers import pipeline

sentiment_pipeline = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert"
)
print("FinBERT model loaded successfully. This model was trained on financial news, earnings calls and analyst reports.")
```

### 2. Analyze Headlines

Feed financial headlines to the `sentiment_pipeline` to get sentiment predictions.

```python
import pandas as pd

headlines = [
    "Apple reports record quarterly earnings, beating analyst expectations",
    "Tesla stock crashes after CEO sells $4 billion in shares",
    "Federal Reserve holds interest rates steady amid inflation concerns",
    "Google announces $70 billion share buyback program",
    "Silicon Valley Bank collapses in largest bank failure since 2008",
    "Microsoft Azure cloud revenue grows 28% year over year",
    "Amazon faces antitrust investigation by European regulators",
    "JPMorgan upgrades Apple to overweight with $220 price target",
    "Oil prices surge 8% following OPEC production cut announcement",
    "Emerging market currencies slide as dollar strengthens"
]

results = sentiment_pipeline(headlines)

sentiment_df = pd.DataFrame({
    'Headline': headlines,
    'Sentiment': [r['label'] for r in results],
    'Confidence': [round(r['score'], 4) for r in results]
})
print(sentiment_df.to_string(index=False))
```

### 3. Enhanced Stock Sentiment Analysis (Production-Grade)

This function (`analyse_stock_sentiment`) provides a more robust analysis, including data confidence and contrarian flags, by processing multiple headlines per stock.

```python
def analyse_stock_sentiment(ticker, headlines, min_headlines=5):
    # ... (function implementation as in the notebook)
    # See notebook cell '8WWFWZ1PuPgK' for full code
    pass # Placeholder for brevity

# Example usage with expanded headlines
expanded_headlines = {
    'AAPL': [
        "Apple reports record quarterly earnings beating expectations",
        "iPhone 16 demand surges in Asian markets",
        "Apple services revenue hits all time high",
        "Warren Buffett increases Berkshire stake in Apple",
        "Apple announces $110 billion share buyback",
    ],
    'TSLA': [
        "Tesla stock crashes after CEO sells $4 billion in shares",
        "Tesla misses delivery targets for third consecutive quarter",
        "Tesla faces increasing competition from Chinese EV makers",
    ],
    'GOOGL': [
        "Google announces $70 billion share buyback program",
        "Alphabet AI division reports breakthrough in quantum computing",
    ],
    'JPM': [
        "JPMorgan posts record profit driven by investment banking fees",
        "Federal Reserve holds rates steady benefiting bank margins",
        "JPMorgan expands African operations with Kenya headquarters",
    ],
    'MSFT': [
        "Microsoft Azure cloud revenue grows 28% year over year",
        "Microsoft Copilot AI integration drives enterprise adoption",
        "Microsoft acquires AI startup for $1.5 billion",
        "Microsoft named most valuable company globally",
    ]
}

enhanced_results = []
for ticker, headlines in expanded_headlines.items():
    result = analyse_stock_sentiment(ticker, headlines)
    if result:
        enhanced_results.append(result)

enhanced_df = pd.DataFrame(enhanced_results)
print("\n=== ENHANCED SENTIMENT ANALYSIS WITH CONFIDENCE RATINGS ===\n")
print(enhanced_df[[
    'Ticker', 'Signal', 'Avg_Score',
    'Headlines_Analysed', 'Data_Confidence', 'Contrarian_Flag'
]].to_string(index=False))
```

### 4. Combine with Portfolio Data

Integrate the AI sentiment signals with existing quantitative portfolio data (e.g., Sharpe Ratio, Volatility, Current Weight) to generate actionable trading suggestions.

```python
# Example portfolio data
portfolio_data = pd.DataFrame({
    'Ticker':        ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM'],
    'Sharpe_Ratio':  [1.93,   2.10,   1.90,    0.82,   1.12],
    'Volatility':    [0.224,  0.246,  0.281,   0.584,  0.198],
    'Current_Weight':[0.20,   0.20,   0.15,    0.05,   0.20]
})

sentiment_scores = enhanced_df[['Ticker', 'Avg_Score', 'Signal', 'Data_Confidence']]
combined = portfolio_data.merge(sentiment_scores, on='Ticker')

# Calculate Combined Score and Suggested Action (see notebook cell 'C340ybbWzyid' for full implementation)
# ...

print("\n=== QUANT + AI COMBINED DECISION FRAMEWORK ===\n")
print(combined[[
    'Ticker', 'Sharpe_Ratio', 'Signal',
    'Combined_Score', 'Suggested_Action'
]].to_string(index=False))
```

## Visualizations

The project includes visualizations to better understand the sentiment distribution, model confidence, and the combined quant+AI decision framework. Refer to the notebook cells for the `matplotlib` code to generate these plots.

## AI Engineering Principles Applied

This project demonstrates several key AI engineering principles:

*   **Pre-trained Model Deployment**: Efficiently leveraging existing, high-performance models.
*   **NLP Pipeline Design**: Building a robust and modular natural language processing workflow.
*   **Production-Grade Considerations**: Addressing real-world challenges like sample size bias and offering nuanced insights.
*   **Quant + AI Integration**: Combining symbolic reasoning (quantitative metrics) with connectionist AI (NLP models) for a hybrid, powerful decision system.

## Conclusion

This Financial News Sentiment Analyzer provides a blueprint for an AI-driven system that processes financial information faster and more consistently than manual methods, offering a data-backed foundation for investment decisions while still allowing for human oversight and strategic thinking.

```
