import numpy as np


def gen_scaled_dataset_repeat(dataset_name, scale_factor):
    scaled_dataset_name=f"{dataset_name}_{scale_factor}x_repeat"

    data_x = np.load(f"data/{dataset_name}_x_test.npy")
    data_y = np.load(f"data/{dataset_name}_y_test.npy")
    data_x_scaled = np.repeat(data_x, repeats=scale_factor, axis=0)
    data_y_scaled = np.repeat(data_y, repeats=scale_factor, axis=0)

    np.save(f"data/{scaled_dataset_name}_x_test.npy", data_x_scaled)
    np.save(f"data/{scaled_dataset_name}_y_test.npy", data_y_scaled)

def gen_scaled_dataset_tile(dataset_name, scale_factor):
    scaled_dataset_name=f"{dataset_name}_{scale_factor}x_tile"

    data_x = np.load(f"data/{dataset_name}_x_test.npy")
    data_y = np.load(f"data/{dataset_name}_y_test.npy")
    data_x_scaled = np.tile(data_x, [scale_factor, 1])
    data_y_scaled = np.tile(data_y, scale_factor)

    np.save(f"data/{scaled_dataset_name}_x_test.npy", data_x_scaled)
    np.save(f"data/{scaled_dataset_name}_y_test.npy", data_y_scaled)



# gen_scaled_dataset_tile("higgs1m", 100)
# gen_scaled_dataset_tile("letters", 1000)
# gen_scaled_dataset_tile("mortgage1Q", 10)
# gen_scaled_dataset_tile("plasticc", 100)

# gen_scaled_dataset_tile("abalone", 160000)
gen_scaled_dataset_tile("airline-ohe", 10)
# gen_scaled_dataset_tile("mlsr", 3)
# gen_scaled_dataset_tile("santander", 200)





