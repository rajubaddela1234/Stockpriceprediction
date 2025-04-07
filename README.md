Follow these steps in your command prompt to set up and run the project:

1.  **Upgrade pip (if necessary):**

    
    pip install --upgrade pip
    
    This command will check if your `pip` version is up-to-date and upgrade it if a newer version is available.

2.  **Create a virtual environment using conda:**

    
    conda create -p venv python==3.12 -y

    This command creates a new conda environment named `venv` in the current directory and installs Python version 3.12 within it. The `-y` flag automatically confirms any prompts.

3.  **Activate the virtual environment:**

    
    conda activate venv/

    This command activates the newly created conda environment. Your command prompt should now show `(venv)` at the beginning, indicating that the environment is active.

4.  **Install dependencies from requirements.txt:**

    
    pip install -r requirements.txt
    
    This command reads the `requirements.txt` file (which should be in the root of your project) and installs all the necessary Python packages listed in it, including Streamlit.

5.  **Run the Streamlit application:**

    
    streamlit run app.py

    This command executes the `app.py` file using Streamlit. This will typically open a new tab in your web browser displaying the Streamlit application.
