import os
# Stock_70
# 70 Increasing
# DT-VAE-GAN
# os.system(
#     "python3 run_dvae_gan.py  --epochs 700 --lr 0.001 --ad_loss_weight 30 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/ "
#     "--data_name stock_70_increasing --experiment_name DT-VAE-GAN_stock_70_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE-GAN_stock_70_increasing/DT-VAE-GAN_stock_70_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE-GAN_stock_70_increasing/'")

# # DT-VAE
# os.system(
#     "python3 run_dvae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/ "
#     "--data_name stock_70_increasing --experiment_name DT-VAE_stock_70_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE_stock_70_increasing/DT-VAE_stock_70_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE_stock_70_increasing/'")

# # T-VAE-GAN
# os.system(
#     "python3 run_vae_gan.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/ "
#     "--data_name stock_70_increasing --experiment_name T-VAE-GAN_stock_70_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' "
#     "--save_dir='./experiments/T-VAE-GAN_stock_70_increasing/T-VAE-GAN_stock_70_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE-GAN_stock_70_increasing/' ")

# # T-VAE
# os.system(
#     "python3 run_vae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/ "
#     "--data_name stock_70_increasing --experiment_name T-VAE_stock_70_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' "
#     "--save_dir='./experiments/T-VAE_stock_70_increasing/T-VAE_stock_70_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE_stock_70_increasing/' ")

# 70 Decreasing
# os.system(
#     "python3 run_dvae_gan.py  --epochs 700 --lr 0.001 --ad_loss_weight 15 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/ "
#     "--data_name stock_70_decreasing --experiment_name DT-VAE-GAN_stock_70_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE-GAN_stock_70_decreasing/DT-VAE-GAN_stock_70_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE-GAN_stock_70_decreasing/'")

# # DT-VAE
# os.system(
#     "python3 run_dvae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/ "
#     "--data_name stock_70_decreasing --experiment_name DT-VAE_stock_70_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE_stock_70_decreasing/DT-VAE_stock_70_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE_stock_70_decreasing/'")

# # T-VAE-GAN
# os.system(
#     "python3 run_vae_gan.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/ "
#     "--data_name stock_70_decreasing --experiment_name T-VAE-GAN_stock_70_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' "
#     "--save_dir='./experiments/T-VAE-GAN_stock_70_decreasing/T-VAE-GAN_stock_70_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE-GAN_stock_70_decreasing/' ")

# # T-VAE
# os.system(
#     "python3 run_vae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/ "
#     "--data_name stock_70_decreasing --experiment_name T-VAE_stock_70_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' "
#     "--save_dir='./experiments/T-VAE_stock_70_decreasing/T-VAE_stock_70_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_70/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE_stock_70_decreasing/' ")


# Stock_91
# 91 Increasing
# DT-VAE-GAN
# os.system(
#     "python3 run_dvae_gan.py  --epochs 700 --lr 0.001 --ad_loss_weight 30 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/ "
#     "--data_name stock_91_increasing --experiment_name DT-VAE-GAN_stock_91_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE-GAN_stock_91_increasing/DT-VAE-GAN_stock_91_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE-GAN_stock_91_increasing/'")

# DT-VAE
os.system(
    "python3 run_dvae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/ "
    "--data_name stock_91_increasing --experiment_name DT-VAE_stock_91_increasing --parent_dir ./experiments")
os.system(
    "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' --diff_first_day=True "
    "--save_dir='./experiments/DT-VAE_stock_91_increasing/DT-VAE_stock_91_increasing.npy' "
    "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/' "
    "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
    "--model_dir='./experiments/DT-VAE_stock_91_increasing/'")

# # T-VAE-GAN
# os.system(
#     "python3 run_vae_gan.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/ "
#     "--data_name stock_91_increasing --experiment_name T-VAE-GAN_stock_91_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' "
#     "--save_dir='./experiments/T-VAE-GAN_stock_91_increasing/T-VAE-GAN_stock_91_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE-GAN_stock_91_increasing/' ")

# # T-VAE
# os.system(
#     "python3 run_vae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/ "
#     "--data_name stock_91_increasing --experiment_name T-VAE_stock_91_increasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' "
#     "--save_dir='./experiments/T-VAE_stock_91_increasing/T-VAE_stock_91_increasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/' "
#     "--dataset_name='increasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE_stock_91_increasing/' ")

# 91 Decreasing
# os.system(
#     "python3 run_dvae_gan.py  --epochs 700 --lr 0.001 --ad_loss_weight 15 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/ "
#     "--data_name stock_91_decreasing --experiment_name DT-VAE-GAN_stock_91_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE-GAN_stock_91_decreasing/DT-VAE-GAN_stock_91_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE-GAN_stock_91_decreasing/'")

# # DT-VAE
# os.system(
#     "python3 run_dvae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/ "
#     "--data_name stock_91_decreasing --experiment_name DT-VAE_stock_91_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' --diff_first_day=True "
#     "--save_dir='./experiments/DT-VAE_stock_91_decreasing/DT-VAE_stock_91_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/DT-VAE_stock_91_decreasing/'")

# # T-VAE-GAN
# os.system(
#     "python3 run_vae_gan.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/ "
#     "--data_name stock_91_decreasing --experiment_name T-VAE-GAN_stock_91_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_gan_trained_model' "
#     "--save_dir='./experiments/T-VAE-GAN_stock_91_decreasing/T-VAE-GAN_stock_91_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE-GAN_stock_91_decreasing/' ")

# # T-VAE
# os.system(
#     "python3 run_vae.py  --epochs 700 --lr 0.001 --data_dir ./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/ "
#     "--data_name stock_91_decreasing --experiment_name T-VAE_stock_91_decreasing --parent_dir ./experiments")
# os.system(
#     "python3 sample_path_generation.py --noise_dim=15  --model_name='vae_trained_model' "
#     "--save_dir='./experiments/T-VAE_stock_91_decreasing/T-VAE_stock_91_decreasing.npy' "
#     "--dataset_dir='./datasets/Dataset_Dynamic_Trend_Preserve/Stock_91/train_test/' "
#     "--dataset_name='decreasing_test.npy' --num_series_per_path=2 "
#     "--model_dir='./experiments/T-VAE_stock_91_decreasing/' ")
