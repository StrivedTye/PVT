# train on Kitti
#python main.py --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Car --log_dir ./baseline_logs
#python main.py --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Pedestrian --log_dir ./baseline_logs
#python main.py --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Van --log_dir ./baseline_logs
#python main.py --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Cyclist --log_dir ./baseline_logs
#python main.py --cfg cfgs/BAT_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Car --log_dir ./baseline_logs
#python main.py --cfg cfgs/BAT_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Pedestrian --log_dir ./baseline_logs
#python main.py --cfg cfgs/BAT_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Van --log_dir ./baseline_logs
#python main.py --cfg cfgs/BAT_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Cyclist --log_dir ./baseline_logs
#python main.py --cfg cfgs/M2Track_kitti.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Car --log_dir ./baseline_logs
#python main.py --cfg cfgs/M2Track_kitti.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Pedestrian --log_dir ./baseline_logs
#python main.py --cfg cfgs/M2Track_kitti.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Van --log_dir ./baseline_logs
#python main.py --cfg cfgs/M2Track_kitti.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Cyclist --log_dir ./baseline_logs
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Car --log_dir ./baseline_logs
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Pedestrian --log_dir ./baseline_logs
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Van --log_dir ./baseline_logs
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name Cyclist --log_dir ./baseline_logs
# train on nuScence
python main.py --cfg cfgs/STNet_Car_nuscenes.yaml --epoch 50 --workers 12 --batch_size 88 --gpu 0 1 --category_name Car --log_dir ./baseline_logs
python main.py --cfg cfgs/STNet_Car_nuscenes.yaml --epoch 50 --workers 12 --batch_size 88 --gpu 0 1 --category_name Pedestrian --log_dir ./baseline_logs
