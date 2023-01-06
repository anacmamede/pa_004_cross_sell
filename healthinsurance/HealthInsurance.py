import pickle
import inflection
import pandas as pd
# import scikit-learn

class HealthInsurance(object):
    def __init__(self):
        self.home_path = ''          
        self.age_scaler                 = pickle.load(open(self.home_path + 'features/age_scaler.pkl', 'rb'))
        self.annual_premium_scaler      = pickle.load(open(self.home_path + 'features/annual_premium_scaler.pkl', 'rb'))
        self.fe_policy_sales_channel    = pickle.load(open(self.home_path + 'features/fe_policy_sales_channel.pkl', 'rb'))
        self.target_encode_gender       = pickle.load(open(self.home_path + 'features/target_encode_gender.pkl', 'rb'))
        self.target_encode_region_code  = pickle.load(open(self.home_path + 'features/target_encode_region_code.pkl', 'rb'))
        self.vintage_scaler             = pickle.load(open(self.home_path + 'features/vintage_scaler.pkl', 'rb'))
        
    def data_cleaning(self, df_raw):
        cols_old = df_raw.columns
        snakecase = lambda x: inflection.underscore( x )
        cols_new = list( map( snakecase, cols_old ) )
        df_raw.columns = cols_new
        return df_raw
    
    def feature_engineering(self, df1):
        #df1['region_code'] = df1['region_code'].astype(int)
        #df1['policy_sales_channel'] = df1['policy_sales_channel'].astype(int)
        # vehicle_age
        df1['vehicle_age'] = df1['vehicle_age'].apply(lambda x: 'over_2_years' if x=='> 2 Years' else 'between_1_2_year' if x=='1-2 Year' else 'below_1_year')
        # vehicle_damage
        df1['vehicle_damage'] = df1['vehicle_damage'].apply(lambda x: 1 if x=='Yes' else 0)
        return df1

    def data_preparation(self, df2):
        # 'annual_premium'
        df2['annual_premium']  = self.annual_premium_scaler.transform(df2[['annual_premium']].values)

        # 'age'
        df2['age']  = self.age_scaler.transform(df2[['age']].values)

        # 'vintage'
        df2['vintage']  = self.vintage_scaler.transform(df2[['vintage']].values)

        # gender
        df2['gender'] = df2['gender'].map( self.target_encode_gender )

        # region_code  
        df2['region_code'] = df2['region_code'].map( self.target_encode_region_code )

        # vehicle_age - **One hot enconding** / Order enconding / Frequency encoding
        df2 = pd.get_dummies( df2, prefix=['vehicle_age'], columns=['vehicle_age'] )

        # policy_sales_channel - **Frequency Enconding** / Target Encoding
        df2.loc[:,'policy_sales_channel'] = df2['policy_sales_channel'].map(self.fe_policy_sales_channel)
        
        cols_selected = ['age','region_code', 'policy_sales_channel','vehicle_damage','previously_insured','annual_premium','vintage']
        
        return df2[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # model prediction
        pred = model.predict_proba(test_data)
        
        # join prediction into original data
        original_data['score'] = pred[:,1].tolist()
        
        return original_data.to_json(orient='records', date_format='iso')
