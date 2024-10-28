# train on Kitti
#python main.py --cfg cfgs/P2B_Car.yaml --epoch 50 --batch_size 32 --gpu 0 1 2
#python main.py --cfg cfgs/BAT_Car.yaml --epoch 50 --batch_size 32 --gpu 0 1 2
#python main.py --cfg cfgs/PVT_Car.yaml --epoch 50 --batch_size 64 --gpu 0 1 --category_name Car --base_scale 0.05
#python main.py --cfg cfgs/PVT_Car.yaml --epoch 50 --batch_size 64 --gpu 0 1 --category_name Car --base_scale 0.10
#python main.py --cfg cfgs/PVT_Car.yaml --epoch 50 --batch_size 64 --gpu 0 1 --category_name Car --base_scale 0.15
#python main.py --cfg cfgs/PVT_Car.yaml --epoch 50 --batch_size 64 --gpu 0 1 --category_name Car --base_scale 0.20
#python main.py --cfg cfgs/PVT_Car.yaml --epoch 50 --batch_size 64 --gpu 0 1 --category_name Car --base_scale 0.25
python main.py --cfg cfgs/PVT_Car.yaml --epoch 40 --batch_size 64 --gpu 0 1 --category_name Car --num_knn 4
python main.py --cfg cfgs/PVT_Car.yaml --epoch 40 --batch_size 64 --gpu 0 1 --category_name Car --num_knn 8
python main.py --cfg cfgs/PVT_Car.yaml --epoch 40 --batch_size 64 --gpu 0 1 --category_name Car --num_knn 12
python main.py --cfg cfgs/PVT_Car.yaml --epoch 40 --batch_size 64 --gpu 0 1 --category_name Car --num_knn 16
python main.py --cfg cfgs/PVT_Car.yaml --epoch 40 --batch_size 64 --gpu 0 1 --category_name Car --num_knn 20
#
# train on nuScence
#python main.py --cfg cfgs/P2B_Car_nuscenes.yaml --batch_size 64 --gpu 2 3 --epoch 50 --category_name nocar
#python main.py --cfg cfgs/P2B_Car_nuscenes.yaml --batch_size 64 --gpu 2 3 --epoch 50 --category_name noped
#python main.py --cfg cfgs/BAT_Car_nuscenes.yaml --batch_size 64 --gpu 0 1 --epoch 50 --category_name nocar
#python main.py --cfg cfgs/BAT_Car_nuscenes.yaml --batch_size 64 --gpu 0 1 --epoch 50 --category_name noped
#python main.py --cfg cfgs/PVT_Car_nuscenes.yaml --batch_size 64 --gpu 2 3 --epoch 50 --category_name nocar
#python main.py --cfg cfgs/PVT_Car_nuscenes.yaml --batch_size 64 --gpu 2 3 --epoch 50 --category_name noped
#
# test on nuScence
#python main.py --test --cfg cfgs/BAT_Car_nuscenes.yaml --checkpoint lightning_logs/version_18/checkpoints/last.ckpt --gpu 0 1 2 3
#python main.py --test --cfg cfgs/BAT_Car_nuscenes.yaml --checkpoint lightning_logs/version_2/checkpoints/last.ckpt --gpu 0 1 2 3
#
# bicycle->truck
# sed -i 's/category_name: bicycle/category_name: truck/g' ./cfgs/P2B_Car_nuscenes.yaml
# python main.py --gpu 0 1 --cfg cfgs/P2B_Car_nuscenes.yaml --checkpoint ./lighting_logs/version_2/checkpoint/last.ckpt --test
