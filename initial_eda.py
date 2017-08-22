import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    pbtype = pd.read_csv('/Users/CamillaNawaz/Google Drive/Galvanize/dsi-capstone-camilla/FARS2015NationalCSV/pbtype.csv')


    # Inspect the features
    # Refer to the manual to understand what the features mean
    # The manual is available at https://crashstats.nhtsa.dot.gov/Api/Public/ViewPublication/812315
    # See page 12 to understand how they are all joined by unique identifiers
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
    print 'State:'
    make_histogram(accidents['STATE'], num_bins=50)
    print
    print 'Hour:'
    make_histogram(accidents['HOUR'], num_bins=24)
    print
    print 'Number of vehicles involved:'
    make_histogram(accidents['VE_FORMS'], num_bins=30) # Indicates number of vehicles involved in the crash
    print
    print 'Special Jurisdictions:'
    make_histogram(accidents.SP_JUR, num_bins=10)
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
    make_histogram(vehicles.DR_WGT, num_bins=30)
    make_histogram(vehicles.DR_HGT, num_bins=30)
    make_histogram(vehicles.PREV_SUS, num_bins=12)
    make_histogram(vehicles.PREV_DWI, num_bins=12)
    make_histogram(vehicles.PREV_ACC, num_bins=12)
    make_histogram(persons.RACE, num_bins=30)

    # What would be some interesting labels?
    print persons.INJ_SEV.value_counts()
    print vehicles.DEFORMED.value_counts()

    # I'll go with injury severity. What's the dumbest, baseline average?
    print np.mean(persons.INJ_SEV)
