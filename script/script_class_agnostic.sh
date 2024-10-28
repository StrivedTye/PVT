# train on Kitti
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCar --log_dir ./class_agnostic
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noPed --log_dir ./class_agnostic
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCyc --log_dir ./class_agnostic
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noVan --log_dir ./class_agnostic
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCar --log_dir ./class_agnostic --re_weight
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noPed --log_dir ./class_agnostic --re_weight
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCyc --log_dir ./class_agnostic --re_weight
#python main.py --cfg cfgs/STNet_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noVan --log_dir ./class_agnostic --re_weight
#
#python main.py --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCyc --log_dir ./class_agnostic
#python main.py --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noVan --log_dir ./class_agnostic
#python main.py --cfg cfgs/BAT_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCyc --log_dir ./class_agnostic
#python main.py --cfg cfgs/BAT_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noVan --log_dir ./class_agnostic
#python main.py --re_weight --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCyc --log_dir ./class_agnostic
#python main.py --re_weight --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noVan --log_dir ./class_agnostic
#python main.py --re_weight --cfg cfgs/BAT_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCyc --log_dir ./class_agnostic
#python main.py --re_weight --cfg cfgs/BAT_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noVan --log_dir ./class_agnostic
python main.py --cfg cfgs/P2B_Car.yaml --epoch 50 --workers 12 --batch_size 64 --gpu 0 1 --category_name noCar --log_dir ./class_agnostic
