import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

def generate_loan_data(num_samples):
    """Generate synthetic loan data with realistic eligibility criteria"""
    np.random.seed(42)
    
    # Generate base features with realistic distributions
    income = np.random.lognormal(mean=11, sigma=0.5, size=num_samples)  # More realistic income distribution
    income = np.clip(income, 20000, 500000)  # Clip to reasonable range
    
    credit_score = np.random.normal(loc=680, scale=80, size=num_samples)  # Center around 680
    credit_score = np.clip(credit_score, 300, 850).astype(int)  # Clip to valid range
    
    employment_status = np.random.choice(
        ['Employed', 'Self-employed', 'Unemployed'],
        size=num_samples,
        p=[0.75, 0.20, 0.05]  # More realistic employment distribution
    )
    
    loan_type = np.random.choice(
        ['Home Loan', 'Education Loan', 'Car Loan'],
        size=num_samples,
        p=[0.35, 0.25, 0.40]
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'employment_status': employment_status,
        'loan_type': loan_type
    })
    
    # Define realistic eligibility criteria
    def determine_eligibility(row):
        # Base criteria for different loan types
        loan_criteria = {
            'Home Loan': {
                'min_income': 50000,
                'preferred_income': 80000,
                'min_credit': 640,
                'preferred_credit': 700
            },
            'Education Loan': {
                'min_income': 30000,
                'preferred_income': 50000,
                'min_credit': 620,
                'preferred_credit': 680
            },
            'Car Loan': {
                'min_income': 25000,
                'preferred_income': 40000,
                'min_credit': 600,
                'preferred_credit': 660
            }
        }
        
        criteria = loan_criteria[row['loan_type']]
        
        # Automatic rejections
        if row['credit_score'] < criteria['min_credit']:
            return 'Not Eligible'
        if row['employment_status'] == 'Unemployed' and row['loan_type'] != 'Education Loan':
            return 'Not Eligible'
        if row['income'] < criteria['min_income']:
            return 'Not Eligible'
            
        # Automatic approvals for very strong applications
        if (row['credit_score'] >= criteria['preferred_credit'] and 
            row['income'] >= criteria['preferred_income'] * 1.5 and 
            row['employment_status'] == 'Employed'):
            return 'Eligible'
            
        # For borderline cases, use a scoring system
        score = 0
        score += (row['credit_score'] - criteria['min_credit']) / 200
        score += (row['income'] - criteria['min_income']) / criteria['min_income']
        
        if row['employment_status'] == 'Employed':
            score += 0.5
        elif row['employment_status'] == 'Self-employed':
            score += 0.3
            
        return 'Eligible' if score > 1 else 'Not Eligible'
    
    df['eligibility'] = df.apply(determine_eligibility, axis=1)
    
    return df

class LoanEligibilityModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoder = LabelEncoder()
    
    def preprocess_data(self, df, training=True):
        df_processed = df.copy()
        
        # Create derived features
        df_processed['income_to_loan_ratio'] = df_processed.apply(
            lambda x: self._calculate_income_ratio(x['income'], x['loan_type']), axis=1
        )
        
        # Encode categorical variables
        categorical_columns = ['employment_status', 'loan_type']
        for column in categorical_columns:
            if training:
                self.label_encoders[column] = LabelEncoder()
                df_processed[column] = self.label_encoders[column].fit_transform(df_processed[column])
            else:
                df_processed[column] = self.label_encoders[column].transform(df_processed[column])
        
        # Scale numerical features
        numerical_columns = ['income', 'credit_score', 'income_to_loan_ratio']
        if training:
            df_processed[numerical_columns] = self.scaler.fit_transform(df_processed[numerical_columns])
        else:
            df_processed[numerical_columns] = self.scaler.transform(df_processed[numerical_columns])
        
        return df_processed
    
    def _calculate_income_ratio(self, income, loan_type):
        typical_loan_amounts = {
            'Home Loan': 250000,
            'Education Loan': 50000,
            'Car Loan': 35000
        }
        return income / typical_loan_amounts.get(loan_type, 50000)
    
    def train(self, df):
        df_processed = self.preprocess_data(df, training=True)
        
        features = ['income', 'credit_score', 'employment_status', 'loan_type', 'income_to_loan_ratio']
        X = df_processed[features]
        
        # Explicitly define the encoding mapping
        self.target_encoder.fit(['Not Eligible', 'Eligible'])
        y = self.target_encoder.transform(df['eligibility'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        return X_test, y_test, self.model.predict(X_test)
    
    def predict(self, df):
        df_processed = self.preprocess_data(df, training=False)
        features = ['income', 'credit_score', 'employment_status', 'loan_type', 'income_to_loan_ratio']
        X = df_processed[features]
        
        predictions = self.model.predict(X)
        prediction_proba = self.model.predict_proba(X)
        
        # Apply post-prediction rules
        predictions = self._apply_custom_rules(df, predictions)
        
        # Convert numeric predictions back to labels
        prediction_labels = self.target_encoder.inverse_transform(predictions)
        
        return prediction_labels, prediction_proba
    
    def _apply_custom_rules(self, df, predictions):
        """Apply hard business rules after model prediction"""
        for i, (pred, row) in enumerate(zip(predictions, df.iterrows())):
            # Strict rejection criteria
            if (row[1]['credit_score'] < 500 or 
                (row[1]['employment_status'] == 'Unemployed' and row[1]['loan_type'] == 'Home Loan') or
                (row[1]['income'] < 25000 and row[1]['loan_type'] == 'Home Loan')):
                predictions[i] = self.target_encoder.transform(['Not Eligible'])[0]
                
            # Automatic approval for exceptional cases
            elif (row[1]['credit_score'] > 800 and 
                  row[1]['income'] > 150000 and 
                  row[1]['employment_status'] == 'Employed'):
                predictions[i] = self.target_encoder.transform(['Eligible'])[0]
        
        return predictions
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder
        }
        joblib.dump(model_data, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        model_data = joblib.load(filepath)
        loan_model = cls()
        loan_model.model = model_data['model']
        loan_model.scaler = model_data['scaler']
        loan_model.label_encoders = model_data['label_encoders']
        loan_model.target_encoder = model_data['target_encoder']
        return loan_model

if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic data...")
    synthetic_data = generate_loan_data(5000)  # Increased sample size
    
    # Create and train the model
    print("Training model...")
    loan_model = LoanEligibilityModel()
    X_test, y_test, y_pred = loan_model.train(synthetic_data)
    
    # Save the model
    print("Saving model...")
    loan_model.save_model('loan_eligibility_model.joblib')
    
    # Test predictions with diverse cases
    test_cases = pd.DataFrame([
        {
            'income': 35000,
            'credit_score': 580,
            'employment_status': 'Unemployed',
            'loan_type': 'Car Loan'
        },
        {
            'income': 150000,
            'credit_score': 820,
            'employment_status': 'Employed',
            'loan_type': 'Home Loan'
        },
        {
            'income': 60000,
            'credit_score': 700,
            'employment_status': 'Self-employed',
            'loan_type': 'Education Loan'
        }
    ])
    
    predictions, probabilities = loan_model.predict(test_cases)
    
    print("\nTest Case Predictions:")
    for i, (case, pred, prob) in enumerate(zip(test_cases.iterrows(), predictions, probabilities)):
        print(f"\nCase {i+1}:")
        print(f"Details: {case[1].to_dict()}")
        print(f"Prediction: {pred}")
        print(f"Confidence: {max(prob) * 100:.2f}%")
        
    # Print distribution of eligibility in training data
    print("\nTraining Data Eligibility Distribution:")
    print(synthetic_data['eligibility'].value_counts(normalize=True))






# class LoanEligibilityModel:
#     """Class for loan eligibility prediction model"""
    
#     def __init__(self):
#         """Initialize the model with preprocessing pipeline"""
#         # Define categorical and numeric features
#         self.categorical_features = [
#             'Gender', 'Married', 'Education', 'Self_Employed', 
#             'Credit_History', 'Property_Area', 'Loan_Type'
#         ]
#         self.numeric_features = [
#             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
#             'Loan_Amount_Term', 'Credit_Score'
#         ]
        
#         # Create preprocessing pipeline
#         self.preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', StandardScaler(), self.numeric_features),
#                 ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
#             ])
        
#         # Create the full pipeline with classifier
#         self.model = Pipeline(steps=[
#             ('preprocessor', self.preprocessor),
#             ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
#         ])
        
#     @staticmethod
#     def generate_sample_data(n_samples=1000):
#         """Generate realistic sample data for training
        
#         Args:
#             n_samples: Number of samples to generate
            
#         Returns:
#             X: Features DataFrame
#             y: Target array (0 for denied, 1 for approved)
#         """
#         # Random seed for reproducibility
#         np.random.seed(42)
        
#         # Generate features
#         data = {
#             'Gender': np.random.choice(['Male', 'Female'], n_samples),
#             'Married': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples),
#             'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples, p=[0.7, 0.3]),
#             'Self_Employed': np.random.choice(['Employed', 'Self-employed', 'Unemployed'], n_samples),
#             'ApplicantIncome': np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples),
#             'CoapplicantIncome': np.random.lognormal(mean=7, sigma=3, size=n_samples) * 
#                                 np.random.binomial(1, 0.5, n_samples),  # 50% have co-applicants
#             'LoanAmount': np.random.lognormal(mean=10, sigma=0.8, size=n_samples),
#             'Loan_Amount_Term': np.random.choice([12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 300, 360], n_samples),
#             'Credit_History': np.random.choice(['Good (1+ year)', 'Limited (< 1 year)', 'None'], n_samples, p=[0.7, 0.2, 0.1]),
#             'Property_Area': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
#             'Loan_Type': np.random.choice(['Home Loan', 'Education Loan', 'Car Loan', 'Personal Loan', 'Business Loan'], n_samples),
#             'Credit_Score': np.random.normal(loc=680, scale=100, size=n_samples)
#         }
        
#         X = pd.DataFrame(data)
        
#         # Create target variable with realistic rules
#         # Higher income, good credit history, educated people more likely to be approved
#         approval_prob = (
#             (X['ApplicantIncome'] > 50000) * 0.3 +
#             (X['CoapplicantIncome'] > 0) * 0.1 +
#             (X['Credit_Score'] > 650) * 0.3 +
#             (X['Credit_History'] == 'Good (1+ year)') * 0.2 +
#             (X['Education'] == 'Graduate') * 0.1 +
#             (X['Self_Employed'] != 'Unemployed') * 0.1 -
#             (X['LoanAmount'] > 100000) * 0.2
#         )
        
#         # Normalize to 0-1 range
#         approval_prob = (approval_prob - approval_prob.min()) / (approval_prob.max() - approval_prob.min())
        
#         # Generate binary outcome based on probability
#         y = np.random.binomial(1, approval_prob)
        
#         return X, y
    
#     def train(self, X, y):
#         """Train the model on given data
        
#         Args:
#             X: Features DataFrame
#             y: Target array
#         """
#         self.model.fit(X, y)
        
#     def predict(self, X):
#         """Predict loan eligibility
        
#         Args:
#             X: Features DataFrame
            
#         Returns:
#             result: 'Approved' or 'Denied'
#             confidence: Confidence percentage
#         """
#         # Get probability of approval
#         proba = self.model.predict_proba(X)[0, 1] * 100
        
#         # Decision threshold is 50%
#         if proba >= 50:
#             return "Approved", proba
#         else:
#             return "Denied", 100 - proba
    
#     def save_model(self, filename):
#         """Save model to file
        
#         Args:
#             filename: File path to save the model
#         """
#         joblib.dump(self.model, filename)
    
#     @classmethod
#     def load_model(cls, filename):
#         """Load model from file
        
#         Args:
#             filename: File path of the saved model
            
#         Returns:
#             LoanEligibilityModel: Model instance with loaded model
#         """
#         instance = cls()
#         instance.model = joblib.load(filename)
#         return instance