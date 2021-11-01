
import pandas

if __name__ == "__main__":

     classification = pandas.read_csv('/Users/andreidm/ETH/projects/pheno-ml/res/comparison/classification/classification.csv')

     classification.loc[classification.method == 'SwAV', 'method'] = 'swav_resnet50'
     classification.loc[classification.method == 'ResNet-50', 'method'] = 'resnet50'

     classification.to_csv('/Users/andreidm/ETH/projects/pheno-ml/res/comparison/classification/classification.csv', index=False)
     print()