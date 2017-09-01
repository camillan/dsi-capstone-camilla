from urllib import urlretrieve

def get_URLs_and_filenames():
    '''
    Generates lists of URLs where the data is downloadable and the filenames where I will save them.
    '''
    base_string = 'http://www.state.nj.us/transportation/refdata/accident/'
    years = map(str, range(2001, 2016))
    counties = ['Atlantic', 'Bergen', 'Burlington', 'Camden', 'CapeMay', 'Cumberland', 'Essex', 'Gloucester', 'Hudson', 'Hunterdon', 'Mercer', 'Middlesex', 'Monmouth', 'Morris', 'Ocean', 'Passaic', 'Salem', "Somerset", "Sussex", 'Union', 'Warren']
    download_base_string = '/Users/CamillaNawaz/Documents/Capstone_Data/NJ_Vehicles/'
    file_names = []
    URLs = []
    for yr in years:
            for county in counties:
                URLs.append(base_string + yr + '/' + county + yr + 'Vehicles' + '.zip')
                file_names.append(download_base_string + county + yr + '.zip')
    return file_names, URLs


def download_vehicle_data(file_names, URLs):
    '''
    Access the site and actually do the downloading and save it in my directory.
    Note: this will take a long time (>5 hours).
    '''
    for URL, filename in zip(URLs, file_names)[24]:
        urlretrieve(URL, filename)
    # no need to return anything


if __name__ == '__main__':
    file_names, URLs = get_URLs_and_filenames()
    download_vehicle_data(file_names, URLs)

urlretrieve('http://www.state.nj.us/transportation/refdata/accident/2005/Essex2005Vehicles.zip','/Users/CamillaNawaz/Documents/Capstone_Data/NJ_Vehicles/Essex2005.zip')
