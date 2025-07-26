"""
Main application entry point for the Loan Eligibility Predictor.
First run train_model.py to generate the model, then run this script.
"""

import os.path
import tkinter as tk
from tkinter import messagebox

# Import GUI application
from loan_eligibility_gui import LoanEligibilityGUI

def main():
    # Check if model file exists
    if not os.path.exists('loan_eligibility_model.joblib'):
        messagebox.showerror(
            "Model Not Found", 
            "The model file 'loan_eligibility_model.joblib' was not found.\n\n"
            "Please run train_model.py first to generate the model."
        )
        return
    
    # Initialize main window
    root = tk.Tk()
    
    # Create the application
    app = LoanEligibilityGUI(root)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()