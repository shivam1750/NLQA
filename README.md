

# Natural-Language-Query-Agent

## Ema Intern Take-Home Challenge

This project is part of the EMA AI Intern Challenge. The goal is to build a Natural Language Query Agent that fulfills the requirements specified in the challenge. The detailed task description can be found [here](https://docs.google.com/document/d/1y-kxX5LbRd8g6Jw47rxU_eY5k1C7WaGp9YcvfklldKM/edit).
The complete explanation and the working of code can be found [here](https://docs.google.com/document/d/1F6xuKN0pS29-n7Wzssm-w9YPUizDR-im/edit?usp=sharing&ouid=100449829076689359206&rtpof=true&sd=true).

## Features

- The application allows users to query a dataset using natural language.
- It processes the queries and returns relevant results from the dataset.
- The application is built to handle various types of natural language queries.

## Requirements

- Python 3.x
- The necessary Python packages are listed in `requirements.txt`.

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/RaviAnand59/Natural-Language-Query-Agent.git
    cd Natural-Language-Query-Agent
    ```

2. **Ensure you have Python installed:**

    If you don't have Python installed, you can download and install it from [python.org](https://www.python.org/).

3. **Create a virtual environment:**

    It's recommended to use a virtual environment to avoid conflicts with global package versions.

    ```sh
    python -m venv env
    ```

    For Mac/Linux, refer to online resources for creating a virtual environment.

4. **Activate the virtual environment:**

    On Windows:

    ```sh
    .\env\Scripts\activate
    ```

    On Mac/Linux:

    ```sh
    source env/bin/activate
    ```

5. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

1. **Start the application:**

    ```sh
    python Main.py
    ```

2. **Troubleshooting:**

    If you encounter the following error:

    ```
    OMP: Error #15: Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized.
    OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program.
    ```

    Run the command:

    ```sh
    set KMP_DUPLICATE_LIB_OK=TRUE
    ```

    Then, run the application again:

    ```sh
    python Main.py
    ```

## Usage

Once the application is running, you can input natural language queries, and the agent will process and return relevant results from the dataset.


## Contact

For any questions or issues, please contact [Ravi Anand](mailto:anandravi977@gmail.com) or call +91-6207656307.


---

