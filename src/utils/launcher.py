
import pandas, time
from tqdm import tqdm
from src import trials, constants, cluster_analysis

if __name__ == "__main__":

    if True:
        for drug in tqdm(constants.drugs):
            # for drug in ['Cladribine']:
            print('\nperforming umap for {}...'.format(drug))
            # only max concentrations are used there
            cluster_analysis.perform_umap_for_drug_and_plot_results(drug, time_point='all', n=5, metric='euclidean',
                                                   annotate_points=False,
                                                   save_to='/Users/andreidm/ETH/projects/pheno-ml/res/embeddings/drugs/')