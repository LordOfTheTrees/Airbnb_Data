"""
Visualization style configuration
"""
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 10)

# Configuration constants
MAX_TEXT_LENGTH = 100  # Maximum characters for text fields in variable summary
TOP_CORRELATIONS_N = 25  # Number of top correlations to save and display

def set_style(figsize=(12, 10)):
    """Set matplotlib and seaborn style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = figsize

