import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from loan_prediction_system import LoanEligibilityModel
import joblib
from tkinter.font import Font
# Using try/except for ttkthemes in case it's not installed
try:
    from ttkthemes import ThemedTk
except ImportError:
    ThemedTk = None
# Using try/except for PIL in case it's not installed
try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None
import os

class LoanEligibilityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Loan Eligibility Predictor")
        self.root.geometry("1280x800")
        
        # Set app theme
        self.root.configure(bg="#f5f5f7")
        
        # Set custom fonts
        self.header_font = Font(family="Helvetica", size=16, weight="bold")
        self.section_font = Font(family="Helvetica", size=12, weight="bold")
        self.normal_font = Font(family="Helvetica", size=10)
        self.result_font = Font(family="Helvetica", size=12, weight="bold")
        
        # Load the trained model
        try:
            self.model = LoanEligibilityModel.load_model('loan_eligibility_model.joblib')
        except FileNotFoundError:
            messagebox.showerror("Error", "Model file not found! Please train the model first.")
            root.destroy()
            return
            
        # Configure custom styles
        self.configure_styles()
        
        # Create main container
        self.main_frame = ttk.Frame(root, style="Main.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Create header frame
        self.create_header()
        
        # Create notebook for tabs
        self.create_notebook()
        
        # Create application form tab
        self.create_application_tab()
        
        # Create dashboard tab
        self.create_dashboard_tab()
        
        # Create history tab
        self.create_history_tab()
        
        # Initialize loan requirements data
        self.initialize_loan_requirements()
        
        # Update requirements display
        self.update_requirements()

    def configure_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        
        # Main frames
        style.configure("Main.TFrame", background="#f5f5f7")
        
        # Header style
        style.configure("Header.TLabel", 
                        font=self.header_font, 
                        background="#0066cc", 
                        foreground="white",
                        padding=10)
        
        # Section headers
        style.configure("Section.TLabel", 
                        font=self.section_font, 
                        background="#f5f5f7", 
                        foreground="#333333",
                        padding=(0, 10, 0, 5))
        
        # Normal labels
        style.configure("TLabel", 
                        font=self.normal_font, 
                        background="#f5f5f7",
                        foreground="#333333")
        
        # Entry fields
        style.configure("TEntry", 
                        font=self.normal_font, 
                        fieldbackground="white")
        
        # Comboboxes
        style.configure("TCombobox", 
                        font=self.normal_font,
                        fieldbackground="white")
        
        # Buttons
        style.configure("TButton", 
                        font=self.normal_font,
                        padding=8)
        
        style.configure("Primary.TButton",
                        background="#0066cc",
                        foreground="black")
        
        style.map("Primary.TButton",
                  background=[("active", "#004d99")],
                  foreground=[("active", "black")])
        
        # Notebook tabs
        style.configure("TNotebook", 
                        background="#f5f5f7",
                        tabmargins=[2, 5, 2, 0])
        
        style.configure("TNotebook.Tab", 
                        font=self.normal_font,
                        padding=[15, 5],
                        background="#e0e0e0")
        
        style.map("TNotebook.Tab",
                  background=[("selected", "#0066cc")],
                  foreground=[("selected", "white")])
        
        # Frames
        style.configure("Card.TFrame", 
                        background="white", 
                        relief="raised")
        
        # Result styles
        style.configure("Approved.TLabel", 
                        font=self.result_font,
                        foreground="#28a745",
                        background="white")
        
        style.configure("Denied.TLabel", 
                        font=self.result_font,
                        foreground="#dc3545",
                        background="white")
                        
        # Progress bar
        style.configure("Confidence.Horizontal.TProgressbar", 
                        troughcolor="#f0f0f0", 
                        background="#0066cc")

    def create_header(self):
        """Create application header"""
        header_frame = ttk.Frame(self.main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # App logo/name
        logo_frame = ttk.Frame(header_frame, style="Header.TFrame")
        logo_frame.pack(fill=tk.X)
        
        # Load logo if available and PIL is installed
        if Image is not None and ImageTk is not None:
            try:
                logo_img = Image.open("loan_logo.png")
                logo_img = logo_img.resize((40, 40), Image.LANCZOS)
                self.logo_photo = ImageTk.PhotoImage(logo_img)
                logo_label = ttk.Label(logo_frame, image=self.logo_photo, style="Header.TLabel")
                logo_label.pack(side=tk.LEFT, padx=10)
            except:
                pass  # No logo available or can't load
        
        app_name = ttk.Label(logo_frame, text="Loan Eligibility Predictor", style="Header.TLabel")
        app_name.pack(side=tk.LEFT, padx=10)

    def create_notebook(self):
        """Create notebook for tabs"""
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

    def create_application_tab(self):
        """Create application form tab"""
        self.app_tab = ttk.Frame(self.notebook, style="Main.TFrame")
        self.notebook.add(self.app_tab, text="New Application")
        
        # Split into two columns using a PanedWindow
        self.app_paned = ttk.PanedWindow(self.app_tab, orient=tk.HORIZONTAL)
        self.app_paned.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for form
        self.left_panel = ttk.Frame(self.app_paned, style="Main.TFrame")
        
        # Create right panel for requirements (with fixed width)
        self.right_panel = ttk.Frame(self.app_paned, style="Main.TFrame")
        
        # Add both panels to the PanedWindow
        self.app_paned.add(self.left_panel, weight=3)  # Form gets more space
        self.app_paned.add(self.right_panel, weight=1)  # Requirements panel gets less space
        
        # Create scrollable form
        self.create_application_form()
        
        # Create requirements panel
        self.create_requirements_panel()

    def create_application_form(self):
        """Create the application form with inputs"""
        # Create canvas with scrollbar for the form
        canvas_frame = ttk.Frame(self.left_panel, style="Main.TFrame")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10)
        
        form_canvas = tk.Canvas(canvas_frame, bg="#f5f5f7", highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=form_canvas.yview)
        
        self.scrollable_frame = ttk.Frame(form_canvas, style="Main.TFrame")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: form_canvas.configure(scrollregion=form_canvas.bbox("all"))
        )
        
        form_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        form_canvas.configure(yscrollcommand=scrollbar.set)
        
        form_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Form content
        self.create_form_sections()
        
        # Bind mousewheel for scrolling
        form_canvas.bind_all("<MouseWheel>", lambda e: form_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        # For Linux and Mac (if needed)
        form_canvas.bind_all("<Button-4>", lambda e: form_canvas.yview_scroll(-1, "units"))
        form_canvas.bind_all("<Button-5>", lambda e: form_canvas.yview_scroll(1, "units"))

    def create_form_sections(self):
        """Create form sections with all inputs"""
        # Personal Information Section
        personal_frame = ttk.LabelFrame(self.scrollable_frame, text="Personal Information", padding=15)
        personal_frame.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=10)
        
        # Gender
        ttk.Label(personal_frame, text="Gender:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(personal_frame, textvariable=self.gender_var, width=25, state="readonly")
        gender_combo['values'] = ('Male', 'Female', 'Other')
        gender_combo.grid(row=0, column=1, padx=10, pady=8, sticky=tk.W)
        gender_combo.set('Male')
        
        # Marital Status
        ttk.Label(personal_frame, text="Marital Status:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.marital_var = tk.StringVar()
        marital_combo = ttk.Combobox(personal_frame, textvariable=self.marital_var, width=25, state="readonly")
        marital_combo['values'] = ('Single', 'Married', 'Divorced', 'Widowed')
        marital_combo.grid(row=1, column=1, padx=10, pady=8, sticky=tk.W)
        marital_combo.set('Single')
        
        # Education
        ttk.Label(personal_frame, text="Education:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.education_var = tk.StringVar()
        education_combo = ttk.Combobox(personal_frame, textvariable=self.education_var, width=25, state="readonly")
        education_combo['values'] = ('Graduate', 'Not Graduate')
        education_combo.grid(row=2, column=1, padx=10, pady=8, sticky=tk.W)
        education_combo.set('Graduate')
        
        # Employment Information Section
        employment_frame = ttk.LabelFrame(self.scrollable_frame, text="Employment Information", padding=15)
        employment_frame.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=10)
        
        # Employment Status
        ttk.Label(employment_frame, text="Employment Status:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.employment_var = tk.StringVar()
        employment_combo = ttk.Combobox(employment_frame, textvariable=self.employment_var, width=25, state="readonly")
        employment_combo['values'] = ('Employed', 'Self-employed', 'Unemployed')
        employment_combo.grid(row=0, column=1, padx=10, pady=8, sticky=tk.W)
        employment_combo.set('Employed')
        
        # Financial Information Section
        financial_frame = ttk.LabelFrame(self.scrollable_frame, text="Financial Information", padding=15)
        financial_frame.grid(row=2, column=0, sticky=tk.EW, padx=10, pady=10)
        
        # Annual Income (LPA)
        ttk.Label(financial_frame, text="Annual Income (LPA):").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.income_var = tk.StringVar()
        income_entry = ttk.Entry(financial_frame, textvariable=self.income_var, width=25)
        income_entry.grid(row=0, column=1, padx=10, pady=8, sticky=tk.W)
        ttk.Label(financial_frame, text="(in lakhs per annum)").grid(row=0, column=2, sticky=tk.W, pady=8)
        
        # Co-applicant Income (LPA)
        ttk.Label(financial_frame, text="Co-applicant Income (LPA):").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.coapplicant_income_var = tk.StringVar()
        coapplicant_income_entry = ttk.Entry(financial_frame, textvariable=self.coapplicant_income_var, width=25)
        coapplicant_income_entry.grid(row=1, column=1, padx=10, pady=8, sticky=tk.W)
        ttk.Label(financial_frame, text="(in lakhs per annum)").grid(row=1, column=2, sticky=tk.W, pady=8)
        
        # Credit Score
        ttk.Label(financial_frame, text="Credit Score:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.credit_var = tk.StringVar()
        credit_entry = ttk.Entry(financial_frame, textvariable=self.credit_var, width=25)
        credit_entry.grid(row=2, column=1, padx=10, pady=8, sticky=tk.W)
        ttk.Label(financial_frame, text="(300-850)").grid(row=2, column=2, sticky=tk.W, pady=8)
        
        # Credit History
        ttk.Label(financial_frame, text="Credit History:").grid(row=3, column=0, sticky=tk.W, pady=8)
        self.credit_history_var = tk.StringVar()
        credit_history_combo = ttk.Combobox(financial_frame, textvariable=self.credit_history_var, width=25, state="readonly")
        credit_history_combo['values'] = ('Good (1+ year)', 'Limited (< 1 year)', 'None')
        credit_history_combo.grid(row=3, column=1, padx=10, pady=8, sticky=tk.W)
        credit_history_combo.set('Good (1+ year)')
        
        # Loan Information Section
        loan_frame = ttk.LabelFrame(self.scrollable_frame, text="Loan Information", padding=15)
        loan_frame.grid(row=3, column=0, sticky=tk.EW, padx=10, pady=10)
        
        # Loan Type
        ttk.Label(loan_frame, text="Loan Type:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.loan_type_var = tk.StringVar()
        loan_type_combo = ttk.Combobox(loan_frame, textvariable=self.loan_type_var, width=25, state="readonly")
        loan_type_combo['values'] = ('Home Loan', 'Education Loan', 'Car Loan', 'Personal Loan', 'Business Loan')
        loan_type_combo.grid(row=0, column=1, padx=10, pady=8, sticky=tk.W)
        loan_type_combo.set('Home Loan')
        loan_type_combo.bind('<<ComboboxSelected>>', self.update_requirements)
        
        # Loan Amount
        ttk.Label(loan_frame, text="Loan Amount:").grid(row=1, column=0, sticky=tk.W, pady=8)
        self.loan_amount_var = tk.StringVar()
        loan_amount_entry = ttk.Entry(loan_frame, textvariable=self.loan_amount_var, width=25)
        loan_amount_entry.grid(row=1, column=1, padx=10, pady=8, sticky=tk.W)
        ttk.Label(loan_frame, text="(in lakhs)").grid(row=1, column=2, sticky=tk.W, pady=8)
        
        # Loan Term
        ttk.Label(loan_frame, text="Loan Term:").grid(row=2, column=0, sticky=tk.W, pady=8)
        self.loan_term_var = tk.StringVar()
        loan_term_combo = ttk.Combobox(loan_frame, textvariable=self.loan_term_var, width=25, state="readonly")
        loan_term_combo['values'] = ('12', '24', '36', '48', '60', '72', '84', '120', '180', '240', '300', '360')
        loan_term_combo.grid(row=2, column=1, padx=10, pady=8, sticky=tk.W)
        loan_term_combo.set('36')
        ttk.Label(loan_frame, text="(in months)").grid(row=2, column=2, sticky=tk.W, pady=8)
        
        # Property Information (for Home Loans)
        property_frame = ttk.LabelFrame(self.scrollable_frame, text="Property Information", padding=15)
        property_frame.grid(row=4, column=0, sticky=tk.EW, padx=10, pady=10)
        
        # Property Area
        ttk.Label(property_frame, text="Property Area:").grid(row=0, column=0, sticky=tk.W, pady=8)
        self.property_area_var = tk.StringVar()
        property_area_combo = ttk.Combobox(property_frame, textvariable=self.property_area_var, width=25, state="readonly")
        property_area_combo['values'] = ('Urban', 'Suburban', 'Rural')
        property_area_combo.grid(row=0, column=1, padx=10, pady=8, sticky=tk.W)
        property_area_combo.set('Urban')
        
        # Action Buttons
        action_frame = ttk.Frame(self.scrollable_frame, style="Main.TFrame")
        action_frame.grid(row=5, column=0, pady=20)
        
        predict_btn = ttk.Button(action_frame, text="Check Eligibility", 
                               command=self.predict_eligibility, style="Primary.TButton")
        predict_btn.grid(row=0, column=0, padx=5)
        
        clear_btn = ttk.Button(action_frame, text="Clear Form", 
                             command=self.clear_form)
        clear_btn.grid(row=0, column=1, padx=5)
        
        # Result Section
        self.create_result_section()

    def create_result_section(self):
        """Create the result section"""
        self.result_frame = ttk.Frame(self.scrollable_frame, style="Card.TFrame")
        self.result_frame.grid(row=6, column=0, sticky=tk.EW, padx=10, pady=15)
        self.result_frame.grid_remove()  # Hide initially
        
        # Result content
        result_container = ttk.Frame(self.result_frame, style="Card.TFrame")
        result_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Result header
        result_header = ttk.Frame(result_container, style="Card.TFrame")
        result_header.pack(fill=tk.X, pady=(0, 15))
        
        self.result_title = ttk.Label(result_header, text="Loan Application Result", font=self.section_font, background="white")
        self.result_title.pack(side=tk.LEFT)
        
        # Result status
        self.eligibility_label = ttk.Label(result_container, text="", style="Approved.TLabel")
        self.eligibility_label.pack(pady=10, anchor=tk.W)
        
        # Confidence progress bar
        confidence_frame = ttk.Frame(result_container, style="Card.TFrame")
        confidence_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(confidence_frame, text="Confidence:", background="white").pack(side=tk.LEFT)
        
        self.confidence_var = tk.IntVar()
        self.confidence_bar = ttk.Progressbar(confidence_frame, orient=tk.HORIZONTAL, 
                                             length=250, mode='determinate', 
                                             variable=self.confidence_var,
                                             style="Confidence.Horizontal.TProgressbar")
        self.confidence_bar.pack(side=tk.LEFT, padx=10)
        
        self.confidence_label = ttk.Label(confidence_frame, text="", background="white")
        self.confidence_label.pack(side=tk.LEFT)
        
        # Recommendation
        recommendation_frame = ttk.Frame(result_container, style="Card.TFrame")
        recommendation_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(recommendation_frame, text="Recommendation:", background="white", font=self.normal_font).pack(anchor=tk.W)
        
        self.recommendation_label = ttk.Label(recommendation_frame, text="", 
                                            background="white", wraplength=550)
        self.recommendation_label.pack(pady=(5, 0), anchor=tk.W)
        
        # Financial Metrics
        metrics_frame = ttk.Frame(result_container, style="Card.TFrame")
        metrics_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(metrics_frame, text="Financial Metrics:", background="white", font=self.normal_font).pack(anchor=tk.W)
        
        self.metrics_frame = ttk.Frame(metrics_frame, style="Card.TFrame")
        self.metrics_frame.pack(pady=(5, 0), fill=tk.X)
        
        # Will be populated when prediction is made

    def create_requirements_panel(self):
        """Create the requirements panel"""
        # Requirements card
        requirements_card = ttk.Frame(self.right_panel, style="Card.TFrame")
        requirements_card.pack(fill=tk.BOTH, expand=True, padx=10)
        
        # Header
        req_header = ttk.Frame(requirements_card, style="Card.TFrame", padding=10)
        req_header.pack(fill=tk.X)
        
        req_title = ttk.Label(req_header, text="Loan Requirements", 
                             font=self.section_font, background="white")
        req_title.pack(side=tk.LEFT)
        
        # Requirements text with styling
        self.requirements_text = tk.Text(requirements_card, wrap=tk.WORD, width=30, height=30,
                                      font=self.normal_font, bg="white", relief="flat",
                                      padx=15, pady=10)
        self.requirements_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure text tags
        self.requirements_text.tag_configure('header', font=('Helvetica', 13, 'bold'))
        self.requirements_text.tag_configure('category', font=('Helvetica', 11, 'bold'), foreground="#0066cc")
        self.requirements_text.tag_configure('item', lmargin1=20, lmargin2=20)
        self.requirements_text.tag_configure('note', font=('Helvetica', 9, 'italic'), foreground="#666666")
        self.requirements_text.tag_configure('bullet', foreground="#0066cc")

    def create_dashboard_tab(self):
        """Create dashboard tab with visualizations"""
        dashboard_tab = ttk.Frame(self.notebook, style="Main.TFrame")
        self.notebook.add(dashboard_tab, text="Dashboard")
        
        # Dashboard Header
        dash_header = ttk.Frame(dashboard_tab, style="Header.TFrame")
        dash_header.pack(fill=tk.X)
        
        dash_title = ttk.Label(dash_header, text="Application Dashboard", style="Header.TLabel")
        dash_title.pack(padx=20, pady=10)
        
        # Dashboard content - Statistics cards
        stats_frame = ttk.Frame(dashboard_tab, style="Main.TFrame")
        stats_frame.pack(fill=tk.X, pady=20, padx=20)
        
        # Create statistics cards
        self.create_stat_card(stats_frame, "Total Applications", "24", 0)
        self.create_stat_card(stats_frame, "Approved", "16", 1)
        self.create_stat_card(stats_frame, "Pending", "5", 2)
        self.create_stat_card(stats_frame, "Rejected", "3", 3)
        
        # Charts section - would include actual charts in a real implementation
        charts_frame = ttk.Frame(dashboard_tab, style="Main.TFrame")
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create placeholder for charts
        self.create_chart_placeholder(charts_frame, "Loan Applications by Type", 0, 0)
        self.create_chart_placeholder(charts_frame, "Monthly Application Volume", 0, 1)
        self.create_chart_placeholder(charts_frame, "Approval Rate by Loan Type", 1, 0)
        self.create_chart_placeholder(charts_frame, "Average Loan Amount", 1, 1)

    def create_stat_card(self, parent, title, value, column):
        """Create a statistics card for the dashboard"""
        card = ttk.Frame(parent, style="Card.TFrame")
        card.grid(row=0, column=column, padx=10, sticky=tk.NSEW)
        
        card_content = ttk.Frame(card, style="Card.TFrame")
        card_content.pack(padx=20, pady=15)
        
        # Card title
        ttk.Label(card_content, text=title, 
                 background="white", foreground="#666666").pack(anchor=tk.W)
        
        # Card value
        ttk.Label(card_content, text=value, 
                 font=("Helvetica", 24, "bold"), 
                 background="white").pack(anchor=tk.W, pady=5)
        
        # Configure column weights
        parent.columnconfigure(column, weight=1)

    def create_chart_placeholder(self, parent, title, row, column):
        """Create a placeholder for a chart"""
        chart_frame = ttk.Frame(parent, style="Card.TFrame")
        chart_frame.grid(row=row, column=column, padx=10, pady=10, sticky=tk.NSEW)
        
        # Chart title
        ttk.Label(chart_frame, text=title, 
                 font=self.section_font,
                 background="white").pack(anchor=tk.W, padx=15, pady=10)
        
        # Chart placeholder
        placeholder = ttk.Frame(chart_frame, height=200, style="Card.TFrame")
        placeholder.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Chart would be created here in a real implementation
        ttk.Label(placeholder, text="Chart visualization placeholder", 
                 background="white").pack(expand=True)
        
        # Configure grid weights
        parent.rowconfigure(row, weight=1)
        parent.columnconfigure(column, weight=1)

    def create_history_tab(self):
        """Create application history tab"""
        history_tab = ttk.Frame(self.notebook, style="Main.TFrame")
        self.notebook.add(history_tab, text="Application History")
        
        # History Header
        history_header = ttk.Frame(history_tab, style="Header.TFrame")
        history_header.pack(fill=tk.X)
        
        history_title = ttk.Label(history_header, text="Application History", style="Header.TLabel")
        history_title.pack(padx=20, pady=10)
        
        # Search and filter controls
        filter_frame = ttk.Frame(history_tab, style="Main.TFrame")
        filter_frame.pack(fill=tk.X, padx=20, pady=15)
        
        ttk.Label(filter_frame, text="Filter:").pack(side=tk.LEFT, padx=(0, 10))
        
        filter_var = tk.StringVar()
        filter_combo = ttk.Combobox(filter_frame, textvariable=filter_var, width=15, state="readonly")
        filter_combo['values'] = ('All', 'Approved', 'Pending', 'Denied')
        filter_combo.pack(side=tk.LEFT, padx=5)
        filter_combo.set('All')
        
        ttk.Label(filter_frame, text="Date Range:").pack(side=tk.LEFT, padx=(20, 10))
        
        date_from = ttk.Entry(filter_frame, width=12)
        date_from.pack(side=tk.LEFT, padx=5)
        date_from.insert(0, "2024-01-01")
        
        ttk.Label(filter_frame, text="to").pack(side=tk.LEFT, padx=5)
        
        date_to = ttk.Entry(filter_frame, width=12)
        date_to.pack(side=tk.LEFT, padx=5)
        date_to.insert(0, "2024-04-15")
        # Search button
        search_btn = ttk.Button(filter_frame, text="Search")
        search_btn.pack(side=tk.LEFT, padx=20)
        
        # Applications table
        table_frame = ttk.Frame(history_tab, style="Card.TFrame")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Create Treeview for applications
        columns = ('id', 'date', 'name', 'loan_type', 'amount', 'term', 'status')
        self.history_tree = ttk.Treeview(table_frame, columns=columns, show='headings')
        
        # Define headings
        self.history_tree.heading('id', text='App ID')
        self.history_tree.heading('date', text='Date')
        self.history_tree.heading('name', text='Applicant')
        self.history_tree.heading('loan_type', text='Loan Type')
        self.history_tree.heading('amount', text='Amount')
        self.history_tree.heading('term', text='Term')
        self.history_tree.heading('status', text='Status')
        
        # Define columns width
        self.history_tree.column('id', width=70)
        self.history_tree.column('date', width=100)
        self.history_tree.column('name', width=150)
        self.history_tree.column('loan_type', width=120)
        self.history_tree.column('amount', width=100)
        self.history_tree.column('term', width=80)
        self.history_tree.column('status', width=100)
        
        # Add sample data
        sample_data = [
            ('2023-09-05', 'Sophia Hughes', 'Education Loan', '₹53L', '84 mo', 'Denied'),
            ('2022-07-07', 'Jessica Gonzalez', 'Car Loan', '₹15L', '60 mo', 'Denied'),
            ('2024-03-16', 'Emily Davis', 'Car Loan', '₹25L', '48 mo', 'Denied'),
            ('2023-04-06', 'Charles Harris', 'Education Loan', '₹35L', '96 mo', 'Denied'),
            ('2022-10-14', 'Robert Wilson', 'Education Loan', '₹53L', '96 mo', 'Denied'),
            ('2023-12-28', 'Steven Baker', 'Car Loan', '₹34L', '60 mo', 'Denied'),
            ('2022-03-08', 'Chloe Morgan', 'Business Loan', '₹64L', '60 mo', 'Approved'),
            ('2022-11-01', 'Sophia Hughes', 'Home Loan', '₹202L', '360 mo', 'Pending'),
            ('2022-02-18', 'David Thompson', 'Education Loan', '₹57L', '84 mo', 'Denied'),
            ('2023-12-02', 'Barbara Scott', 'Education Loan', '₹59L', '120 mo', 'Approved'),
            ('2024-01-31', 'Chloe Morgan', 'Business Loan', '₹74L', '84 mo', 'Denied'),
            ('2022-11-21', 'Sophia Hughes', 'Business Loan', '₹112L', '60 mo', 'Denied'),
            ('2022-12-17', 'Edward Roberts', 'Car Loan', '₹23L', '60 mo', 'Approved'),
            ('2023-07-25', 'Thomas Anderson', 'Education Loan', '₹41L', '120 mo', 'Approved'),
            ('2022-07-08', 'Frank Parker', 'Education Loan', '₹31L', '84 mo', 'Pending'),
            ('2022-08-13', 'Nancy Moore', 'Car Loan', '₹23L', '48 mo', 'Pending'),
            ('2022-03-04', 'Amanda Carter', 'Car Loan', '₹22L', '60 mo', 'Approved'),
            ('2022-12-12', 'Robert Wilson', 'Business Loan', '₹104L', '84 mo', 'Denied'),
            ('2022-07-24', 'Frank Parker', 'Education Loan', '₹49L', '120 mo', 'Pending'),
            ('2024-01-10', 'Karen Clark', 'Education Loan', '₹41L', '84 mo', 'Pending'),
            ('2024-01-13', 'Thomas Anderson', 'Business Loan', '₹71L', '72 mo', 'Pending'),
            ('2023-10-19', 'Sophia Hughes', 'Education Loan', '₹40L', '96 mo', 'Approved'),
            ('2023-10-08', 'Amanda Carter', 'Car Loan', '₹24L', '60 mo', 'Approved'),
            ('2022-08-29', 'Grace Simmons', 'Business Loan', '₹61L', '84 mo', 'Approved'),
            ('2022-01-16', 'Karen Clark', 'Business Loan', '₹106L', '60 mo', 'Approved'),
            ('2022-08-01', 'Carol Evans', 'Business Loan', '₹95L', '72 mo', 'Approved'),
            ('2022-08-16', 'Grace Simmons', 'Car Loan', '₹30L', '48 mo', 'Approved'),
            ('2022-08-10', 'Frank Parker', 'Car Loan', '₹23L', '60 mo', 'Pending'),
            ('2022-12-15', 'Barbara Scott', 'Car Loan', '₹26L', '60 mo', 'Approved'),
            ('2022-09-09', 'Charles Harris', 'Business Loan', '₹97L', '84 mo', 'Denied'),
            ('2024-03-24', 'Anthony Young', 'Home Loan', '₹186L', '360 mo', 'Approved'),
            ('2023-07-24', 'Betty Hall', 'Business Loan', '₹89L', '72 mo', 'Approved'),
            ('2022-10-04', 'Amanda Carter', 'Car Loan', '₹18L', '48 mo', 'Pending'),
            ('2023-05-19', 'Andrew Green', 'Home Loan', '₹314L', '360 mo', 'Approved'),
            ('2022-08-26', 'Sophia Hughes', 'Car Loan', '₹24L', '60 mo', 'Pending'),
            ('2023-09-25', 'Robert Wilson', 'Business Loan', '₹89L', '84 mo', 'Pending'),
            ('2022-06-24', 'Grace Simmons', 'Car Loan', '₹25L', '48 mo', 'Approved'),
            ('2022-03-12', 'Steven Baker', 'Business Loan', '₹94L', '60 mo', 'Approved'),
            ('2023-07-04', 'Jessica Gonzalez', 'Education Loan', '₹60L', '120 mo', 'Denied'),
            ('2023-02-06', 'Karen Clark', 'Business Loan', '₹66L', '84 mo', 'Pending'),
            ('2023-11-21', 'Carol Evans', 'Car Loan', '₹22L', '60 mo', 'Denied'),
            ('2023-04-01', 'Jessica Gonzalez', 'Car Loan', '₹32L', '48 mo', 'Approved'),
            ('2023-02-07', 'Jessica Gonzalez', 'Car Loan', '₹33L', '60 mo', 'Approved'),
            ('2023-08-27', 'Jessica Gonzalez', 'Home Loan', '₹234L', '360 mo', 'Pending'),
            ('2023-04-16', 'Betty Hall', 'Business Loan', '₹76L', '84 mo', 'Pending'),
            ('2022-06-26', 'Karen Clark', 'Home Loan', '₹324L', '360 mo', 'Denied'),
            ('2023-09-30', 'Ryan Jenkins', 'Education Loan', '₹38L', '96 mo', 'Approved'),
            ('2022-04-29', 'Zoe Barker', 'Home Loan', '₹271L', '360 mo', 'Approved'),
            ('2023-03-23', 'Lily Crawford', 'Personal Loan', '₹19L', '36 mo', 'Pending'),
            ('2023-07-16', 'Grace Simmons', 'Business Loan', '₹119L', '72 mo', 'Pending'),
            ('2024-03-30', 'Thomas Anderson', 'Car Loan', '₹29L', '60 mo', 'Approved'),
            ('2022-11-11', 'Karen Clark', 'Car Loan', '₹17L', '60 mo', 'Approved'),
            ('2023-05-17', 'Sophia Hughes', 'Personal Loan', '₹15L', '36 mo', 'Denied'),
            ('2022-11-14', 'David Thompson', 'Education Loan', '₹51L', '120 mo', 'Approved'),
            ('2023-02-01', 'Nancy Moore', 'Home Loan', '₹269L', '360 mo', 'Approved')
        ]
        
        for item in sample_data:
            self.history_tree.insert('', tk.END, values=item)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Details button
        details_frame = ttk.Frame(history_tab, style="Main.TFrame")
        details_frame.pack(fill=tk.X, padx=20, pady=15)
        
        view_details_btn = ttk.Button(details_frame, text="View Details")
        view_details_btn.pack(side=tk.RIGHT)
        
        export_btn = ttk.Button(details_frame, text="Export Data")
        export_btn.pack(side=tk.RIGHT, padx=10)
    
    def initialize_loan_requirements(self):
        """Initialize loan requirements data for each loan type"""
        self.loan_requirements = {
            'Home Loan': {
                'minimum_income': 5.0,  # LPA
                'minimum_credit_score': 650,
                'maximum_loan_amount': 80.0,  # In lakhs
                'documents': [
                    'Income proof (Tax returns for last 2 years)',
                    'Employment verification',
                    'Property documents',
                    'Bank statements (6 months)',
                    'Identity and address proof'
                ],
                'eligibility_criteria': [
                    'Minimum 2 years of employment history',
                    'Debt-to-income ratio below 45%',
                    'Property valuation must exceed loan amount by 20%'
                ]
            },
            'Education Loan': {
                'minimum_income': 3.0,
                'minimum_credit_score': 620,
                'maximum_loan_amount': 25.0,
                'documents': [
                    'Admission letter from institution',
                    'Course fee structure',
                    'Income proof of co-applicant',
                    'Academic records',
                    'Identity and address proof'
                ],
                'eligibility_criteria': [
                    'Admission to recognized institution',
                    'Co-applicant with steady income',
                    'Loan amount not exceeding total course fee plus 20%'
                ]
            },
            'Car Loan': {
                'minimum_income': 3.5,
                'minimum_credit_score': 630,
                'maximum_loan_amount': 15.0,
                'documents': [
                    'Income proof (3 months salary slips)',
                    'Vehicle quotation',
                    'Bank statements (3 months)',
                    'Identity and address proof'
                ],
                'eligibility_criteria': [
                    'Minimum 1 year of employment at current job',
                    'Loan amount not exceeding 85% of vehicle cost',
                    'EMI should not exceed 20% of monthly income'
                ]
            },
            'Personal Loan': {
                'minimum_income': 4.0,
                'minimum_credit_score': 680,
                'maximum_loan_amount': 10.0,
                'documents': [
                    'Income proof (3 months salary slips)',
                    'Bank statements (6 months)',
                    'Identity and address proof'
                ],
                'eligibility_criteria': [
                    'Minimum 2 years of employment history',
                    'No defaults in past 24 months',
                    'EMI should not exceed 40% of monthly income'
                ]
            },
            'Business Loan': {
                'minimum_income': 6.0,
                'minimum_credit_score': 700,
                'maximum_loan_amount': 50.0,
                'documents': [
                    'Business financials (2 years)',
                    'Business plan/proposal',
                    'GST returns',
                    'Bank statements (12 months)',
                    'Identity and address proof',
                    'Business registration documents'
                ],
                'eligibility_criteria': [
                    'Business operational for at least 2 years',
                    'Profitable for at least the last financial year',
                    'No defaults on existing loans'
                ]
            }
        }

    def update_requirements(self, event=None):
        """Update loan requirements display based on selected loan type"""
        loan_type = self.loan_type_var.get()
        
        # Clear previous content
        self.requirements_text.delete(1.0, tk.END)
        
        if loan_type in self.loan_requirements:
            req = self.loan_requirements[loan_type]
            
            # Add loan type header
            self.requirements_text.insert(tk.END, f"{loan_type} Requirements\n\n", 'header')
            
            # Basic requirements
            self.requirements_text.insert(tk.END, "Basic Requirements:\n", 'category')
            self.requirements_text.insert(tk.END, f"• ", 'bullet')
            self.requirements_text.insert(tk.END, f"Minimum Income: {req['minimum_income']} LPA\n", 'item')
            self.requirements_text.insert(tk.END, f"• ", 'bullet')
            self.requirements_text.insert(tk.END, f"Minimum Credit Score: {req['minimum_credit_score']}\n", 'item')
            self.requirements_text.insert(tk.END, f"• ", 'bullet')
            self.requirements_text.insert(tk.END, f"Maximum Loan Amount: ₹{req['maximum_loan_amount']} lakhs\n\n", 'item')
            
            # Document requirements
            self.requirements_text.insert(tk.END, "Required Documents:\n", 'category')
            for doc in req['documents']:
                self.requirements_text.insert(tk.END, f"• ", 'bullet')
                self.requirements_text.insert(tk.END, f"{doc}\n", 'item')
            self.requirements_text.insert(tk.END, "\n")
            
            # Eligibility criteria
            self.requirements_text.insert(tk.END, "Eligibility Criteria:\n", 'category')
            for criteria in req['eligibility_criteria']:
                self.requirements_text.insert(tk.END, f"• ", 'bullet')
                self.requirements_text.insert(tk.END, f"{criteria}\n", 'item')
                
            # Add a note
            self.requirements_text.insert(tk.END, "\nNote: Meeting the minimum requirements does not guarantee loan approval. Additional factors may be considered during evaluation.", 'note')
            
        self.requirements_text.config(state=tk.DISABLED)  # Make read-only after updating

    def clear_form(self):
        """Clear all form fields"""
        # Reset all variables to default values
        self.gender_var.set('Male')
        self.marital_var.set('Single')
        self.education_var.set('Graduate')
        self.employment_var.set('Employed')
        self.income_var.set('')
        self.coapplicant_income_var.set('')
        self.credit_var.set('')
        self.credit_history_var.set('Good (1+ year)')
        self.loan_type_var.set('Home Loan')
        self.loan_amount_var.set('')
        self.loan_term_var.set('36')
        self.property_area_var.set('Urban')
        
        # Hide result section
        self.result_frame.grid_remove()
        
        # Update requirements display
        self.update_requirements()

    def predict_eligibility(self):
        """Predict loan eligibility based on form inputs"""
        try:
            # Validate inputs
            if not self._validate_inputs():
                return
                
            # Get values from form
            gender = self.gender_var.get()
            marital_status = self.marital_var.get()
            education = self.education_var.get()
            employment = self.employment_var.get()
            income = float(self.income_var.get())
            coapplicant_income = float(self.coapplicant_income_var.get()) if self.coapplicant_income_var.get() else 0.0
            credit_score = int(self.credit_var.get())
            credit_history = self.credit_history_var.get()
            loan_type = self.loan_type_var.get()
            loan_amount = float(self.loan_amount_var.get())
            loan_term = int(self.loan_term_var.get())
            property_area = self.property_area_var.get()
            
            # Convert inputs to model format
            # In a real app, this would match the format expected by the model
            input_data = {
                'Gender': 1 if gender == 'Male' else 0,
                'Married': 1 if marital_status == 'Married' else 0,
                'Education': 1 if education == 'Graduate' else 0,
                'Self_Employed': 1 if employment == 'Self-employed' else 0,
                'ApplicantIncome': income * 100000,  # Convert LPA to actual amount
                'CoapplicantIncome': coapplicant_income * 100000,
                'LoanAmount': loan_amount * 100000,
                'Loan_Amount_Term': loan_term,
                'Credit_History': 1 if credit_history == 'Good (1+ year)' else 0,
                'Property_Area_Urban': 1 if property_area == 'Urban' else 0,
                'Property_Area_Rural': 1 if property_area == 'Rural' else 0,
                'Credit_Score': credit_score
            }
            
            # Here you would normally use the model to predict
            # For demonstration, use a simple rule-based approach
            loan_req = self.loan_requirements[loan_type]
            
            # Check if meets minimum requirements
            meets_income = income >= loan_req['minimum_income']
            meets_credit = credit_score >= loan_req['minimum_credit_score']
            meets_loan_amount = loan_amount <= loan_req['maximum_loan_amount']
            
            # Calculate debt-to-income ratio (simplified)
            monthly_payment = self._calculate_monthly_payment(loan_amount, loan_term)
            monthly_income = income * 100000 / 12  # Convert annual to monthly
            dti_ratio = (monthly_payment / monthly_income) * 100
            
            # Determine eligibility based on rules
            if meets_income and meets_credit and meets_loan_amount and dti_ratio < 45:
                eligibility = True
                confidence = min(100, int(70 + (credit_score - loan_req['minimum_credit_score'])/5 + 
                                         (income - loan_req['minimum_income'])*2 - dti_ratio/2))
            else:
                eligibility = False
                confidence = min(100, int(30 + (credit_score - loan_req['minimum_credit_score'])/10 + 
                                         (income - loan_req['minimum_income']) - dti_ratio/3))
            
            confidence = max(5, min(95, confidence))  # Ensure within reasonable bounds
            
            # Display result
            self._display_result(eligibility, confidence, dti_ratio, monthly_payment, loan_type)
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please check your inputs: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def _calculate_monthly_payment(self, loan_amount, term_months):
        """Calculate monthly loan payment (simplified)"""
        # Simplified calculation, assumes 10% interest rate
        principal = loan_amount * 100000  # Convert to actual amount
        rate = 0.10 / 12  # Monthly interest rate
        term = term_months  # Term in months
        
        # EMI formula: P × r × (1+r)^n / ((1+r)^n - 1)
        emi = principal * rate * (1 + rate)**term / ((1 + rate)**term - 1)
        return emi

    def _validate_inputs(self):
        """Validate form inputs"""
        try:
            # Check if numeric fields have valid values
            if not self.income_var.get():
                messagebox.showwarning("Validation Error", "Please enter your annual income.")
                return False
                
            income = float(self.income_var.get())
            if income <= 0:
                messagebox.showwarning("Validation Error", "Income must be greater than zero.")
                return False
            
            if self.coapplicant_income_var.get():
                coapplicant_income = float(self.coapplicant_income_var.get())
                if coapplicant_income < 0:
                    messagebox.showwarning("Validation Error", "Co-applicant income cannot be negative.")
                    return False
            
            if not self.credit_var.get():
                messagebox.showwarning("Validation Error", "Please enter your credit score.")
                return False
                
            credit_score = int(self.credit_var.get())
            if credit_score < 300 or credit_score > 850:
                messagebox.showwarning("Validation Error", "Credit score must be between 300 and 850.")
                return False
            
            if not self.loan_amount_var.get():
                messagebox.showwarning("Validation Error", "Please enter the loan amount.")
                return False
                
            loan_amount = float(self.loan_amount_var.get())
            if loan_amount <= 0:
                messagebox.showwarning("Validation Error", "Loan amount must be greater than zero.")
                return False
            
            return True
            
        except ValueError:
            messagebox.showwarning("Validation Error", "Please enter valid numeric values.")
            return False

    def _display_result(self, is_eligible, confidence, dti_ratio, monthly_payment, loan_type):
        """Display loan eligibility result"""
        # Show result frame
        self.result_frame.grid()
        
        # Update status
        if is_eligible:
            self.eligibility_label.config(text="✓ Loan Eligible", style="Approved.TLabel")
        else:
            self.eligibility_label.config(text="✗ Loan Not Eligible", style="Denied.TLabel")
        
        # Update confidence bar
        self.confidence_var.set(confidence)
        self.confidence_label.config(text=f"{confidence}%")
        
        # Update recommendation
        if is_eligible:
            if confidence > 80:
                recommendation = f"Your application for a {loan_type} has strong approval potential. We recommend proceeding with the application."
            else:
                recommendation = f"Your application for a {loan_type} is eligible, but with some reservations. Consider increasing your down payment or providing additional income proof."
        else:
            if confidence > 40:
                recommendation = f"While you don't currently meet all requirements for a {loan_type}, you could improve eligibility by improving your credit score or reducing the loan amount."
            else:
                recommendation = f"Unfortunately, your profile doesn't align with our {loan_type} requirements at this time. Consider applying for a smaller loan amount or improving your financial profile."
                
        self.recommendation_label.config(text=recommendation)
        
        # Clear previous metrics
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()
        
        # Add financial metrics
        metrics = [
            ("Debt-to-Income Ratio:", f"{dti_ratio:.2f}%", "Higher values indicate greater financial strain. Target: below 45%."),
            ("Monthly Payment:", f"₹{monthly_payment:.2f}", "This is your estimated EMI based on current interest rates."),
            ("Loan-to-Income Ratio:", f"{float(self.loan_amount_var.get())/float(self.income_var.get()):.2f}", "Lower values indicate better loan affordability.")
        ]
        
        for i, (label, value, tooltip) in enumerate(metrics):
            metric_row = ttk.Frame(self.metrics_frame, style="Card.TFrame")
            metric_row.pack(fill=tk.X, pady=5)
            
            ttk.Label(metric_row, text=label, background="white").grid(row=0, column=0, sticky=tk.W)
            ttk.Label(metric_row, text=value, background="white", font=('Helvetica', 10, 'bold')).grid(row=0, column=1, padx=15)
            ttk.Label(metric_row, text=tooltip, background="white", font=('Helvetica', 8, 'italic')).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))


def main():
    # Use ThemedTk if available, else use standard Tk
    if ThemedTk:
        root = ThemedTk(theme="arc")  # Use a modern theme
    else:
        root = tk.Tk()
        
    app = LoanEligibilityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()