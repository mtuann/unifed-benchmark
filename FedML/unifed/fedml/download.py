import flbenchmark.datasets
import json

config = json.load(open('config.json', 'r'))

flbd = flbenchmark.datasets.FLBDatasets('../data')

print("Downloading Data...")

dataset_name = (
                'student_horizontal',
                'breast_horizontal',
                'default_credit_horizontal',
                'give_credit_horizontal',
                'vehicle_scale_horizontal'
                )



for x in dataset_name:
    if config["dataset"] == x:
        train_dataset, test_dataset = flbd.fateDatasets(x)
        flbenchmark.datasets.convert_to_csv(train_dataset, out_dir='../csv_data/{}_train'.format(x))
        if x != 'vehicle_scale_horizontal':
            flbenchmark.datasets.convert_to_csv(test_dataset, out_dir='../csv_data/{}_test'.format(x))


vertical = (
    'breast_vertical',
    'give_credit_vertical',
    'default_credit_vertical'
)

for x in vertical:
    if config["dataset"] == x:
        my_dataset = flbd.fateDatasets(x)
        flbenchmark.datasets.convert_to_csv(my_dataset[0], out_dir='../csv_data/{}'.format(x))
        if my_dataset[1] != None:
            flbenchmark.datasets.convert_to_csv(my_dataset[1], out_dir='../csv_data/{}'.format(x))


leaf = (
    'femnist',
    'reddit',
    'celeba'
)

for x in leaf:
    if config["dataset"] == x:
        my_dataset = flbd.leafDatasets(x)
