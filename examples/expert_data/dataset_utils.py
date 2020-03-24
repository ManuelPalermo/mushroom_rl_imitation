
import numpy as np

def quick_dataset_convert(data_path, save_path):
    # convert data from:
    # https://drive.google.com/drive/folders/1h3H4AY_ZBx08hz-Ct0Nxxus-V1melu1U
    # to common structure -> obs.npy / actions.npy
    expert_files = np.load(data_path, allow_pickle=True)
    print(expert_files.files)

    np.savez(save_path, "expert_dataset_hopper_processed.npz",
             obs=np.vstack(expert_files["obs"]),
             actions=np.vstack(expert_files["acs"]),
             ep_rets=np.stack(expert_files["ep_rets"]),
             rews=np.hstack(expert_files["rews"]))


if __name__ == "__main__":
    env_ids = ['Hopper-v2','HalfCheetah-v2', 'Walker2d-v2', 'Humanoid-v2']
    for env_id in env_ids:
        print(env_id)
        quick_dataset_convert(
                data_path="./expert_dataset_{}.npz".format(env_id),
                save_path="./expert_dataset_{}_processed.npz".format(env_id)
        )
        input()
