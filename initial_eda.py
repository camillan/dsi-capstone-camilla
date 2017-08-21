import pandas as pd
import matplotlib.pyplot as plt



def make_histogram(feature, num_bins=50):
    '''
    Takes in a feature column and returns a histogram of the data, to understand the distribution of the data.
    Clears the plot space to make room for the next plot.
    '''
    plt.hist(feature, bins=num_bins)
    plt.show()
    plt.clf()



if __name__ == '__main__':

    # Load up the data
    accidents = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/accident.csv')
    persons = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/person.csv')
    vehicles = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/vehicle.csv')
    maneuvers = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/maneuver.csv')
    violations = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/violatn.csv')

    # Inspect the features
    # Refer to the manual to understand what the features mean
    # The manual is available at https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315
    # Overall, the primary keys/unique identifiers are ST_CASE
    accidents.info()
    print '-------------------'
    vehicles.info()
    print '-------------------'
    persons.info()
    print '-------------------'
    maneuvers.info()
    print '-------------------'
    violations.info()
    print '-------------------'


    # Make some histograms of features that initially seem important
    make_histogram(accidents['STATE'])
    make_histogram(accidents['HOUR'])
    make_histogram(accidents['VE_FORMS']) # Indicates number of vehicles involved in the crash
    make_histogram(accidents.ROUTE)
    make_histogram(accidents.SP_JUR)
    '''
    You see here that there are certainly some outliers and incorrectly entered data.
    I'll clean those up in another file.
    '''

    # Another good way to look at it is by value_counts
    print accidents.VE_FORMS.value_counts() # Lots of people by themselves
    print accidents.HOUR.value_counts() # Lots around evening rush hour, a lull during the day except for lunch time

    # How many total fatalities in accidents dataset, in total?
    print sum(accidents.FATALS)
    # How many total people were involved?
    print sum(accidents.PERSONS) # about twice as many as number of fatalities

    # Let's check out the distribution of some really nosey personal information
    make_histogram(persons.DR_WGT, num_bins=30)
    make_histogram(persons.PREV_SUS, num_bins=12)
    make_histogram(persons.PREV_DWI, num_bins=12)

    # What would be some interesting lables?
    print persons.INJ_SEV.value_counts()
    print vehicles.deformed.value_counts()
