import urllib

base_string = 'http://www.state.nj.us/transportation/refdata/accident/'
years = map(str, range(2001, 2016))
counties = ['Atlantic', 'Bergen', 'Burlington', 'Camden', 'CapeMay', 'Cumberland', 'Essex', 'Gloucester', 'Hudson', 'Hunterdon', 'Mercer', 'Middlesex', 'Monmouth', 'Morris', 'Ocean', 'Passaic', 'Salem', "Somerset", "Sussex", 'Union', 'Warren']

URLs = []
for yr in years:
        for county in counties:
            URLs.append(base_string + yr + '/' + county + yr + 'Accidents' + '.zip')


NJ_data = []

for URL in URLs:
    NJ_data.append(urllib.urlretrieve(URL))
# urllib.request.Request('ftp://example.com/')
