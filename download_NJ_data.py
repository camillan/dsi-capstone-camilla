from urllib import urlretrieve

base_string = 'http://www.state.nj.us/transportation/refdata/accident/'
years = map(str, range(2001, 2016))
counties = ['Atlantic', 'Bergen', 'Burlington', 'Camden', 'CapeMay', 'Cumberland', 'Essex', 'Gloucester', 'Hudson', 'Hunterdon', 'Mercer', 'Middlesex', 'Monmouth', 'Morris', 'Ocean', 'Passaic', 'Salem', "Somerset", "Sussex", 'Union', 'Warren']

download_base_string = '/Users/CamillaNawaz/Documents/Capstone_Data/NJ_2001_14/'
file_names = []

URLs = []
for yr in years:
        for county in counties:
            URLs.append(base_string + yr + '/' + county + yr + 'Accidents' + '.zip')
            file_names.append(download_base_string + county + yr + '.zip')


NJ_data = []

for URL, filename in zip(URLs, file_names)[97:]:
    NJ_data.append(urlretrieve(URL, filename))
