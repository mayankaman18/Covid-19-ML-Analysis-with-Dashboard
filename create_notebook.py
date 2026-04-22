import os

# %% [markdown]
# # Optional Jupytext Helper Script
# Run this script to generate `Covid19_Analysis.ipynb` from `covid_analysis.py`.

# %%
def convert_to_notebook():
    print("Converting covid_analysis.py to a Jupyter Notebook...")
    # This uses jupytext CLI via python module
    os.system("python -m jupytext --to notebook covid_analysis.py")
    if os.path.exists("covid_analysis.ipynb"):
        print("✅ Successfully generated covid_analysis.ipynb!")
    else:
        print("❌ Conversion failed. Please ensure 'jupytext' is installed.")

if __name__ == "__main__":
    convert_to_notebook()
