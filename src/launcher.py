
import pandas, time
from tqdm import tqdm
from src import trials

if __name__ == "__main__":

    meta_data = pandas.read_csv("/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/metadata/ACHN_CL3_P1.csv")

    control = 'DMSO'

    control_data = meta_data[(meta_data['Drug'] == control) & (meta_data['Final_conc_uM'] == 367.)]
    control_ids = control_data['Row'].astype('str') + control_data['Column'].astype('str')

    drugs_data = meta_data[meta_data['Drug'] != control]
    drug_names = drugs_data['Drug'].dropna().unique()

    drug_info = {}
    for drug_name in tqdm(drug_names):
        print(drug_name, "is being processed")

        # create a dict for this drug
        drug_info[drug_name] = {}
        # add concentraions of this drug
        drug_data = drugs_data[drugs_data['Drug'] == drug_name]
        drug_info[drug_name]['cons'] = drug_data['Final_conc_uM'].dropna()
        # add ids of this drug
        drug_ids = drug_data['Row'].astype('str') + drug_data['Column'].astype('str')
        drug_info[drug_name]['ids'] = drug_ids

        trials.plot_correlation_distance_for_single_samples(drug_name, drug_info[drug_name], control_ids)
        # trials.plot_correlation_distance_for_averaged_samples(drug_name, drug_info[drug_name], control_ids)

        print()
