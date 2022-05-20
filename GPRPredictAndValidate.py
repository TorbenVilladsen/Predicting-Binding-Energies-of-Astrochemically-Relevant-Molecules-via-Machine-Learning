# Import necessary modules
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, RationalQuadratic
from sklearn.model_selection import RepeatedKFold
import warnings
warnings.filterwarnings("ignore")

# Start time
start = time.time()

# The control panel - here strings can be changed to manage the script

# To use monolayer type 'mono' or 'multi' for multilayer
LayerType = 'mono'

# What do you want to test? Type 'kfold' to test performance with k-fold cross validation
# if you want to predict a new molecule type 'predict'
SplitType = 'kfold'

# Load the excel file with data
data = pd.read_excel(r'Data_set_BEs.xlsx')
Rows = 300
df = data.head(Rows)

# Exclude rows with no binding energy
df = df[(df['Ebin (K)'] > 1)]

# Only include the mono- or multilayer
if LayerType == 'mono':
    df = df[(df["Multi / Mono"].isin(["Mono"]))]
elif LayerType == 'multi':
    df = df[(df["Multi / Mono"].isin(["Multi"]))]

# Only include the pure depositions
df = df[(df["Mixed/Pure"].isin(["Pure"]))]

# Only include the four most popular surfaces
if LayerType == 'mono':
    df = df[(df["Surface simplified"].isin(["metal", "carbon", "si", "water"]))]

# Defines columns for all relevant features
C = df["C"].values.tolist()
H = df["H"].values.tolist()
O = df["O"].values.tolist()
N = df["N"].values.tolist()
Cl = df["Cl"].values.tolist()
OH = df["OH"].values.tolist()
CO = df["-C(O)-"].values.tolist()
COO = df["-C(O)O-"].values.tolist()
OO = df["-O-"].values.tolist()
NH2 = df["NH2"].values.tolist()
CN = df["CN"].values.tolist()
NCO = df["-N-C(O)-"].values.tolist()
Mass = df["Mass"].values.tolist()
N_atoms = df["#atoms"].values.tolist()
Surface = df["Surface simplified"].values.tolist()

Ebin = df["Ebin (K)"].values.tolist()

# Read SMILES string
if LayerType == 'mono':
    SMILES = pd.read_csv (r'SMILES_mono.txt', header=None)
else:
    SMILES = pd.read_csv (r'SMILES_multi.txt', header=None)
mdf = SMILES.head(313)

mdf = mdf.values.tolist()

mdf = [x[0] for x in mdf]

ValE = []
HBA = []
HBD = []
TPSA = []

# RDkit features
for moles in mdf:
    molec = Chem.MolFromSmiles(moles)
    ValEl = Descriptors.NumValenceElectrons(molec)
    ValE.append(ValEl)
    HBAl = rdMolDescriptors.CalcNumHBA(molec)
    HBA.append(HBAl)
    HBDl = rdMolDescriptors.CalcNumHBD(molec)
    HBD.append(HBDl)
    TPSAl = rdMolDescriptors.CalcTPSA(molec)
    TPSA.append(TPSAl)


# Define a scaler
scaler = StandardScaler()

if LayerType == 'mono':
    data_columns = ['C', 'H', 'O', 'N', 'Cl', 'OH', 'CO', 'COO', 'OO', 'NH2', 'CN', 'NCO', 'Mass', 'N_atoms', 'ValE', 'HBA', 'HBD', 'TPSA', 'Surface']

    # One hot encode the surfaces
    df = pd.DataFrame(list(zip(C, H, O, N, Cl, OH, CO, COO, OO, CN, NH2, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA, Surface)),
                      columns=data_columns)

    df["Surface"] = df["Surface"].astype('category')

    # One-hot encoding of dataframe
    one_hot_encoded_dataframe = pd.get_dummies(df, prefix='Surface')

    one_hot_encoded_dataframe[data_columns[:-1]] = scaler.fit_transform(df[data_columns[:-1]])

    print(one_hot_encoded_dataframe.shape)

# same for multilayer
elif LayerType == 'multi':
    data_columns = ['C', 'H', 'O', 'N', 'Cl', 'OH', 'CO', 'COO', 'OO', 'NH2', 'CN', 'NCO', 'Mass', 'N_atoms', 'VelE', 'HBA', 'HBD', 'TPSA']

    df = pd.DataFrame(list(zip(C, H, O, N, Cl, OH, CO, COO, OO, CN, NH2, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA)), columns=data_columns)

    one_hot_encoded_dataframe = scaler.fit_transform(df)
    print(one_hot_encoded_dataframe.shape)

# Define a kernel using the RBF and RQ kernel with white noise kernel
kernel = RBF() + RationalQuadratic() + WhiteKernel()

# Performing cross validation
if SplitType == 'kfold':
    prediction_tuples = []
    results = []
    r_squared = []
    index = 0
    k = 5
    n = 1
    if LayerType == 'mono':
        rs = 29
    elif LayerType == 'multi':
        rs = 21
    rkf = RepeatedKFold(n_splits=k, n_repeats=n, random_state=rs)

    Ebin = np.array(Ebin)

    for train_index, test_index in rkf.split(one_hot_encoded_dataframe):
        if LayerType == 'mono':
            df_train, df_test = one_hot_encoded_dataframe.iloc[train_index, :], one_hot_encoded_dataframe.iloc[test_index, :]
        elif LayerType == 'multi':
            df_train, df_test = one_hot_encoded_dataframe[train_index, :], one_hot_encoded_dataframe[test_index, :]

        Ebin_train, Ebin_test = Ebin[train_index], Ebin[test_index]

        regressor = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=25, normalize_y=True)
        regressor.fit(df_train, Ebin_train)

        Ebin_pred = regressor.predict(df_test).reshape(np.shape(Ebin_test))
        zipped_lists = zip(Ebin_pred, Ebin_test)
        squared_list = [(e[0] - e[1]) ** 2 for e in zipped_lists]
        rmse = np.sqrt(np.mean(squared_list))
        results.append(rmse)
        print(index, "GPR RMSE in run = " + str(round(rmse, 1)))
        r_squared.append(regressor.score(df_test, Ebin_test))
        prediction_tuples.extend([(e[0], e[1]) for e in zip(Ebin_test, Ebin_pred)])
        # save values for plot
        if LayerType == 'mono':
            np.save(r'GPRmono', prediction_tuples)
        else:
            np.save("GPRmulti", prediction_tuples)
        index += 1
    print(f'Mean = {np.mean(results):.1f} K')
    print(f'Mean = {np.mean(results) * 8.617 * 10 ** (-5):.3f} eV')
    print(f'R^2 = {np.mean(r_squared):.3f}')
    print("Random state =", rs, ", k = ", k, ", n =", n, "W constant")


# Get SMILES from here https://cactus.nci.nih.gov/chemical/structure
# Get RDKit feature values from here http://www.scbdd.com/rdk_desc/index/
#
if SplitType == 'predict':
    # Feature data for molecules used in the article

    # New predictions
                                                                                          # C,   H,   O,   N,   Cl, CO,  OH,  COO, OO,  NH2,  CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "1", "2", "0", "2", "0", "0", "0", "0", "0", "1", "1", "0", "42", "5", "16", "2", "1", "49.81"         # Cyanamide               NH2CN
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "5", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "43", "8", "20", "1", "1", "26.02"         # Ethanimine              C2H4NH
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "4", "1", "0", "0", "0", "1", "0", "0", "0", "0", "0", "44", "7", "18", "1", "1", "20.23"         # vinylalcohol            CH2=CHOH
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "3", "3", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "53", "7", "20", "1", "1", "23.85"         # Propargylimine          HC3HNH
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "2", "0", "2", "0", "0", "0", "0", "0", "0", "1", "0", "54", "7", "20", "2", "1", "47.64"         # Cyanomethanimine        NHCHCN
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "3", "1", "1", "0", "0", "0", "0", "0", "0", "0", "0", "57", "7", "22", "2", "0", "29.43"         # Methyl isocyanate       C2H3NO
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "5", "1", "1", "0", "0", "0", "0", "0", "0", "0", "1", "59", "9", "24", "1", "1", "43.09"         # Acetamide               CH3CONH2
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "5", "1", "1", "0", "0", "0", "0", "0", "0", "0", "1", "59", "9", "24", "1", "1", "29.1"          # N-methylformamide       C2H5NO
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "1", "4", "1", "2", "0", "0", "0", "0", "0", "0", "0", "2", "60", "8", "24", "1", "2", "69.11"         # Urea                    CH4N2O
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "4", "2", "0", "0", "0", "2", "0", "0", "0", "0", "0", "60", "8", "24", "2", "2", "40.46"         # ethenediol              HOCH=CHOH
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "7", "1", "1", "0", "0", "1", "0", "0", "1", "0", "0", "61", "11", "26", "2", "2", "46.25"        # Ethanolamine            C2H7NO
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "5", "4", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "64", "9", "24", "0", "0", "0"             # Allenyl acetylene       H2CCCHCCH
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "4", "3", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "65", "8", "24", "1", "0", "23.79"         # Propargyl cyanide       CH3C3N
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "4", "3", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "65", "8", "24", "1", "0", "23.79"         # Cyanoallene             C4H3N
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "4", "3", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "65", "8", "24", "1", "0", "23.79"         # Cynaopropyne            CH3C3N
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "4", "7", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "69", "12", "28", "1", "0", "23.79"        # Propylcyanide           C3H7CN
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "3", "6", "2", "0", "0", "1", "1", "0", "0", "0", "0", "0", "74", "11", "30", "2", "1", "37.30"        # hydroxyacetone          CH3COCH2OH
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "5", "3", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "77", "9", "28", "1", "0", "23.79"         # Cyanovinylacetylene     HCCCHCHCN  (Vinylcyanoacetylene)
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "6", "3", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "89", "10", "28", "1", "0", "23.79"        # Methylcyanodiacetylene  CH3C5N
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "6", "3", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "89", "10", "36", "1", "0", "23.79"        # Cyanoacetyleneallene    H2CCCHC3N
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "6", "5", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "91", "12", "34", "1", "0", "20.23"        # cyano-cyclopentadiene   c-C5H5CN
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "7", "1", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "99", "9", "34", "1", "0", "23.79"         # Cyanotriacetylene       HC7N
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "9", "1", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "123", "11", "42", "1", "0", "23.79"       # Cyanotetraacetylene     HC9N
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "11", "1", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "147", "13", "50", "1", "0", "23.79"      # Cyanopentaacetylene     HC11N
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "9", "8", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "116", "17", "44", "0", "0", "0"           # indene                  c-C9H8

    # Leave-one-out-cross-validation
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "1", "3", "2", "1", "0", "1", "1", "0", "0", "0", "1", "0", "61", "7", "24", "1", "2", "63.32"         # Carbamic acid           H2NCOOH
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "29", "60", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "408", "89", "176", "0", "0", "0"        # Nonacosane              C29H60
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "3", "6", "1", "0", "0", "1", "0", "0", "0", "0", "0", "0", "58", "10", "24", "1", "0", "17.07"        # Acetone                 CH3COCH3
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "3", "0", "1", "0", "0", "0", "0", "0", "0", "1", "0", "41", "6", "16", "1", "0", "23.79"         # Acetonitile             CH3CN
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "2", "4", "2", "0", "0", "0", "0", "1", "0", "0", "0", "0", "60", "8", "24", "2", "0", "26.3"          # Methyl formate          CH3OCHO
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "5", "6", "2", "2", "0", "0", "0", "0", "0", "0", "0", "1", "126", "15", "48", "2", "2", "65.72"       # Thymine                 C5H6N2O2
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "0", "3", "0", "1", "0", "0", "0", "0", "0", "0", "0", "0", "17", "4", "8", "1", "1", "35"             # Ammonia                 NH3
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "1", "4", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "16", "5", "8", "0", "0", "0"              # Methane                 CH4
    # C, H, O, N, Cl, CO, OH, COO, OO, NH2, CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA = "3", "6", "1", "0", "0", "0", "1", "0", "0", "0", "0", "0", "58", "10", "24", "1", "1", "20.23"        # Allyl alcohol           C3H5OH
                                                                                          # C,   H,   O,   N,   Cl, CO,  OH,  COO, OO,  NH2,  CN, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA

    data_columns = ['C', 'H', 'O', 'N', 'Cl', 'OH', 'CO', 'COO', 'OO', 'NH2', 'CN', 'NCO', 'Mass', 'N_atoms', 'ValE', 'HBA', 'HBD', 'TPSA']
    zipped_data = np.array([C, H, O, N, Cl, OH, CO, COO, OO, CN, NH2, NCO, Mass, N_atoms, ValE, HBA, HBD, TPSA]).reshape((1,-1))
    df = pd.DataFrame(zipped_data, columns=data_columns)

    df = scaler.transform(df)

    df = pd.DataFrame(df, columns=data_columns)

    # Here you choose what surface to predict the BE from. Type either ('carbon', 'metal', 'si', 'water')
    if LayerType == 'mono':
        df1 = pd.get_dummies(pd.DataFrame({'Surface': ['metal']}))

        columns = one_hot_encoded_dataframe.columns

        df2 = pd.concat([df, df1], axis=1, join='inner')

        df = df2.reindex(columns=columns, fill_value=0)

    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50, normalize_y=True)

    model.fit(one_hot_encoded_dataframe, Ebin)
    BE1, STD = model.predict(df, return_std=True)

    print(BE1, STD, 'K')
    print(BE1 * 8.61732814974056 * 10 ** (-5), STD * 8.61732814974056 * 10 ** (-5), 'eV')

end = time.time()

print("The time of execution of above program is :", round(end - start, 1), "s")
