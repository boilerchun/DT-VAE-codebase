import os

# # State
# os.system(
#     "python3 run_dvae_gan.py  --epochs 1000 --lr 0.001 --ad_loss_weight 20 --data_dir ../../Dataset_Predictive_Score/State/increasing_and_late_peak/ "
#     "--data_name state_increasing_and_late_peak --experiment_name state_increasing_and_late_peak --parent_dir ./experiments")
# os.system(
#     "python3 ./utils/visual_utils.py --data_dir ./experiments/state_increasing_and_late_peak/")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15 "
#     "--save_dir='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_state_increasing_and_late_peak.npy' "
#     "--dataset_dir='../../Dataset_Predictive_Score/State/increasing_and_late_peak/' "
#     "--dataset_name='increasing_and_late_peak_test.npy' --num_series_per_path=5 "
#     "--model_dir='./experiments/state_increasing_and_late_peak/'")
# os.system(
#     "python3 cluster.py --data_path='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_state_increasing_and_late_peak.npy' "
#     "--save_path='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_state_increasing_and_late_peak.png' "
# )

# os.system(
#     "python3 run_dvae_gan.py  --epochs 1000 --lr 0.001 --ad_loss_weight 20 --data_dir ../../Dataset_Predictive_Score/State/decreasing_and_early_peak/ "
#     "--data_name state_decreasing_and_early_peak --experiment_name state_decreasing_and_early_peak --parent_dir ./experiments")
# os.system(
#     "python3 ./utils/visual_utils.py --data_dir ./experiments/state_decreasing_and_early_peak/")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15 "
#     "--save_dir='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_state_decreasing_and_early_peak.npy' "
#     "--dataset_dir='../../Dataset_Predictive_Score/State/decreasing_and_early_peak/' "
#     "--dataset_name='decreasing_and_early_peak_test.npy' --num_series_per_path=5 "
#     "--model_dir='./experiments/state_decreasing_and_early_peak/'")
# os.system(
#     "python3 cluster.py --data_path='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_state_decreasing_and_early_peak.npy' "
#     "--save_path='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_state_decreasing_and_early_peak.png' "
# )


# # County
# os.system(
#     "python3 run_dvae_gan.py  --epochs 1000 --lr 0.001 --ad_loss_weight 20 --data_dir ../../Dataset_Predictive_Score/County/increasing_and_late_peak/ "
#     "--data_name county_increasing_and_late_peak --experiment_name county_increasing_and_late_peak --parent_dir ./experiments")
# os.system(
#     "python3 ./utils/visual_utils.py --data_dir ./experiments/county_increasing_and_late_peak/")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15 "
#     "--save_dir='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_county_increasing_and_late_peak.npy' "
#     "--dataset_dir='../../Dataset_Predictive_Score/County/increasing_and_late_peak/' "
#     "--dataset_name='increasing_and_late_peak_test.npy' --num_series_per_path=5 "
#     "--model_dir='./experiments/county_increasing_and_late_peak/'")
# os.system(
#     "python3 cluster.py --data_path='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_county_increasing_and_late_peak.npy' "
#     "--save_path='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_county_increasing_and_late_peak.png' "
# )

# os.system(
#     "python3 run_dvae_gan.py  --epochs 1000 --lr 0.001 --ad_loss_weight 20 --data_dir ../../Dataset_Predictive_Score/County/decreasing_and_trough/ "
#     "--data_name county_decreasing_and_trough --experiment_name county_decreasing_and_trough --parent_dir ./experiments")
# os.system(
#     "python3 ./utils/visual_utils.py --data_dir ./experiments/county_decreasing_and_trough/")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15 "
#     "--save_dir='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_county_decreasing_and_trough.npy' "
#     "--dataset_dir='../../Dataset_Predictive_Score/County/decreasing_and_trough/' "
#     "--dataset_name='decreasing_and_trough_test.npy' --num_series_per_path=5 "
#     "--model_dir='./experiments/county_decreasing_and_trough/'")
# os.system(
#     "python3 cluster.py --data_path='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_county_decreasing_and_trough.npy' "
#     "--save_path='../../Generated_Data/DT-VAE-GAN/Predictive_Score/DT-VAE-GAN_county_decreasing_and_trough.png' "
# )

# Stock
# Increasing
# DT-VAE-GAN
# os.system(
#     "python3 run_dvae_gan.py  --epochs 700 --lr 0.001 --ad_loss_weight 30 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/ "
#     "--data_name stock_increasing --experiment_name DT-VAE-GAN_stock_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE-GAN_stock_increasing/DT-VAE-GAN_stock_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE-GAN_stock_increasing/'")

# DT-VAE
# os.system(
#     "python3 run_dvae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/ "
#     "--data_name stock_increasing --experiment_name DT-VAE_stock_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE_stock_increasing/DT-VAE_stock_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE_stock_increasing/'")

# T-VAE-GAN
# os.system(
#     "python3 run_vae_gan.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/ "
#     "--data_name stock_increasing --experiment_name T-VAE-GAN_stock_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' "
#     "--save_dir='./experiments/T-VAE-GAN_stock_increasing/T-VAE-GAN_stock_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE-GAN_stock_increasing/' ")

# T-VAE
# os.system(
#     "python3 run_vae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/ "
#     "--data_name stock_increasing --experiment_name T-VAE_stock_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' "
#     "--save_dir='./experiments/T-VAE_stock_increasing/T-VAE_stock_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE_stock_increasing/' ")

# Decreasing
# os.system(
#     "python3 run_dvae_gan.py  --epochs 500 --lr 0.001 --ad_loss_weight 15 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/ "
#     "--data_name stock_decreasing --experiment_name DT-VAE-GAN_stock_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE-GAN_stock_decreasing/DT-VAE-GAN_stock_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE-GAN_stock_decreasing/'")

# DT-VAE
# os.system(
#     "python3 run_dvae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/ "
#     "--data_name stock_decreasing --experiment_name DT-VAE_stock_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE_stock_decreasing/DT-VAE_stock_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE_stock_decreasing/'")

# T-VAE-GAN
# os.system(
#     "python3 run_vae_gan.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/ "
#     "--data_name stock_decreasing --experiment_name T-VAE-GAN_stock_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' "
#     "--save_dir='./experiments/T-VAE-GAN_stock_decreasing/T-VAE-GAN_stock_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE-GAN_stock_decreasing/' ")

# T-VAE
os.system(
    "python3 run_vae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/ "
    "--data_name stock_decreasing --experiment_name T-VAE_stock_decreasing --parent_dir ./experiments")
os.system(
    "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' "
    "--save_dir='./experiments/T-VAE_stock_decreasing/T-VAE_stock_decreasing.npy' "
    "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_49/train_test/' "
    "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
    "--model_dir='./experiments/T-VAE_stock_decreasing/' ")
