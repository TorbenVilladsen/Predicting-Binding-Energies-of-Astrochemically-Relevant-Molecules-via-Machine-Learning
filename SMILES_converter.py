import time
import numpy as np
import pandas as pd
import warnings
from urllib.request import urlopen
from urllib.parse import quote

warnings.filterwarnings("ignore")

start = time.time()

# Do you want SMILES for 'mono' or 'multi'?
LayerType = 'mono'

# Import data
data = pd.read_excel(r'Data_set_BEs.xlsx')
Rows = 300
df = data.head(Rows)

df = df[(df['Ebin (K)'] > 1)]

# Exclude all energies abbove 6700 K
if LayerType == 'multi':
    df = df[(df['Ebin (K)'] < 6700)]

# Only include the monolayer
if LayerType == 'mono':
    df = df[(df["Multi / Mono"].isin(["Mono"]))]
elif LayerType == 'multi':
    df = df[(df["Multi / Mono"].isin(["Multi"]))]

# Only include the pure mixture
df = df[(df["Mixed/Pure"].isin(["Pure"]))]

# Only include the four most popular surfaces
if LayerType == 'mono':
    df = df[(df["Surface simplified"].isin(["metal", "carbon", "si", "water"]))]

# Defines colums for all relevant variables
Name = df["Name"].values.tolist()


# Guide for this program https://stackoverflow.com/questions/54930121/converting-molecule-name-to-smiles
# Note, sometimes the website is temporally down, but just wait a few hours
def CIRconvert(ids):
    try:
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans
    except:
        return 'Did not work'

# If 'Did not work' is returned, restart the program

identifiers = Name
index = 0
# To convert and visualize
for ids in identifiers:
    print(index, ids, CIRconvert(ids))
    index += 1

# to convert and save to list in txt file
lst = []
for ids in identifiers:
    lst.append(CIRconvert(ids))

if LayerType == 'mono':
    with open('SMILES_mono.txt', 'w') as filehandle:
        for listitem in lst:
            filehandle.write('%s\n' % listitem)
    filehandle.close()
else:
    with open('SMILES_multi.txt', 'w') as filehandle:
        for listitem in lst:
            filehandle.write('%s\n' % listitem)
    filehandle.close()

end = time.time()

print("The time of execution of above program is :", end - start, "s")
