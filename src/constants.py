
version = 'v.0.2.5'
user = 'andreidm'

cropped_data_path = '/Users/{}/ETH/projects/pheno-ml/data/cropped/'.format(user)


cell_lines = ['ACHN', 'HT29', 'M14',  # batch 1
              'IGROV1', 'MDAMB231', 'SF539',   # batch 2
              'HS578T', 'SKMEL2', 'SW620',  # batch 3
              'EKVX', 'OVCAR4', 'UACC257',  # batch 4
              'BT549', 'LOXIMVI', 'MALME3M',  # batch 5
              'A498', 'COLO205', 'HOP62',  # batch 6
              'HCT15', 'OVCAR5', 'T47D']  # batch 7

drugs = ['Chlormethine', 'Clofarabine', 'Panzem-2-ME2', 'Pemetrexed', 'Asparaginase',
         'Irinotecan', 'Gemcitabine', '17-AAG', 'Docetaxel', 'Erlotinib',
         'UK5099', 'Fluorouracil', 'Everolimus', 'MEDICA 16', 'BPTES',
         'Oligomycin A', 'Trametinib', 'Oxaliplatin', 'Rapamycin', 'Etomoxir',
         'Lenvatinib', 'Oxfenicine', 'Mercaptopurine', 'Metformin', 'Omacetaxine',
         'Cladribine', 'Paclitaxel', 'Methotrexate', 'PBS', 'Topotecan',
         'YC-1', 'Decitabine']

markers = ('.', ',', 'o', 'v', '^',
           '<', '>', '1', '2', '3',
           '4', '8', 's', 'p', 'P',
           '*', 'h', 'H', '+', 'x',
           'X', 'D', 'd')