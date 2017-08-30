import pandas as pd
import csv

def factorize_cols(df=NJ):
    '''
    Factorize columns according to the accidents sample csv and the accidents manual and list of columns.
    '''
    df[u'13'] = pd.factorize(df[u'13'])[0]
    df[u'4'] = pd.factorize(df[u'4'])[0]
    df['44'] = pd.factorize(df['44'])[0] # cell phone use
    df['30'] = pd.factorize(df['30'])[0] # environmental conditions
    df['17'] = pd.factorize(df['17'])[0] # crash type code
    df['involves_injury'] = df[u'13'].isin([1,2])
    return df

if __name__ == '__main__':

    NJ = pd.read_csv('/Users/CamillaNawaz/Documents/Capstone_Data/NJ_2001_14/NJ_master_subsample.csv', quoting=csv.QUOTE_NONE)
    NJ = factorize_cols()


#Notes:
#Column 9: lines up with total killed.
#10: total injured
#11: pedestrians killed
#12: pedestrians injured
#13: severity
# don't include these as they will be data leakage
