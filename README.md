{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/MadihaTech/Infosys-project/blob/main/README.md\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qVATwJPdpOmR"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Real-Time Competitor Strategy Tracker for E-Commerce**"
      ],
      "metadata": {
        "id": "OeSLdILepVf2"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Project Overview**"
      ],
      "metadata": {
        "id": "Va6GOW-epti8"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "This project focuses on creating a real-time competitive intelligence tool for e-commerce businesses. It provides actionable insights by monitoring competitor pricing, discount strategies, and customer sentiment. The solution leverages:"
      ],
      "metadata": {
        "id": "8o29V_TdpxD3"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "*   Machine Learning: Predictive modeling with ARIMA.\n",
        "*   LLMs: Sentiment analysis using Hugging Face Transformers and Groq.\n",
        "*   Integration: Slack notifications for real-time updates.\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "CMtFAy5Yq2C7"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Features**"
      ],
      "metadata": {
        "id": "xN4tCz7JreiE"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "1.   Competitor Data Aggregation: Track pricing and discount strategies.\n",
        "2.   Sentiment Analysis: Analyze customer reviews for actionable insights.\n",
        "3.   Predictive Modeling: Forecast competitor\n",
        "4. Slack Integration: Get real-time notifications on competitor activity.  \n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "5naoOLEarjCd"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Setup Instructions**"
      ],
      "metadata": {
        "id": "t8ZZHyRkHFRV"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**1.Clone the repository**\n"
      ],
      "metadata": {
        "id": "0Q2Jrh9zHUjV"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "git clone <repository-url>\n",
        "cd <repository-directory>"
      ],
      "metadata": {
        "id": "ZXWwSZmnqeG4"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**2.Install Dependencies**"
      ],
      "metadata": {
        "id": "EyrkqlxCHw2L"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Install the required Python libraries using pip:"
      ],
      "metadata": {
        "id": "eLbGeaCQH3ds"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pip install .r requirements.txt"
      ],
      "metadata": {
        "id": "Aw0O-yYIIQ_V"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "**3.Configure API Keys**"
      ],
      "metadata": {
        "id": "5BMu_yu_IZtR"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "This project requires the following keys:\n",
        "\n",
        "\n",
        "*   **Groq API Key:** For generating strategic redcommendations.\n",
        "*   **Slack Webhook URL:** For sending notifications.\n",
        "\n"
      ],
      "metadata": {
        "id": "gD-xAACSIeRf"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**Steps:**\n",
        "\n",
        "\n",
        "1.  **Groq API Key:**\n",
        "\n",
        "\n",
        "*   sign up for a Groq account at https://groq.com.\n",
        "*   Obtain your API Key from the Groq dashboard.\n",
        "*   Use the API Key in the app.py file.\n",
        "\n",
        "\n",
        "\n",
        "2.   **Slack Webhook Integration:**\n",
        "\n",
        "\n",
        "*   Go to the Slack API.\n",
        "*   Create a new app and enable incoming webhooks.\n",
        "*   Add a webhook to a channel and copy the generated URL.\n",
        "*   Add this to URL to the app.py file.\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "Hj3VoNHkJSBP"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "**4.Run the Application**"
      ],
      "metadata": {
        "id": "5btQk_dnK8Qi"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Run the Streamlit app:"
      ],
      "metadata": {
        "id": "gN7wpZNjLDPo"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "streamlit run app.py"
      ],
      "metadata": {
        "id": "K5iFgYpKLHad"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Project Files**"
      ],
      "metadata": {
        "id": "Rc6oNnEqLLfe"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "*   **app.py:** Main application script.\n",
        "*   **scrape.py:** Script for web scraping competitor data.\n",
        "*   **review.csv:** Sample reviews data for sentiment analysis.\n",
        "*   **competitor_data.csv:** Sample competitor data for analysis.\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "BppgIyNkLP8C"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Usage**"
      ],
      "metadata": {
        "id": "mmjrWm2jMAzW"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "1. Launch the Steamlit app.  \n",
        "2. Select a product from the sidebar.\n",
        "3. View competitor analysis, sentiment trends, and discount forecasts.\n",
        "4. Get strategic recommendations and real-time Slack notifications.\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "OhaQq2fNMEhX"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **License**\n",
        "\n",
        "This project is licensed by MIT License."
      ],
      "metadata": {
        "id": "MpLNUTTVNBPD"
      }
    }
  ]
}