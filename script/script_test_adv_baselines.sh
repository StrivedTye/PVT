#echo "=============================KITTI=============================="
#echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#echo "+++++++++++++++++++++++++++++BAT++++++++++++++++++++++++++"
#python main.py --test --cfg cfgs/BAT_Car.yaml --checkpoint pretrained_models/bat_car.ckpt --log_dir adv_output/ --gpu 1 --category_name Car --attack
#python main.py --test --cfg cfgs/BAT_Pedestrian.yaml --checkpoint pretrained_models/bat_ped.ckpt --log_dir adv_output/ --gpu 1 --category_name Pedestrian --attack
#python main.py --test --cfg cfgs/BAT_Car.yaml --checkpoint pretrained_models/bat_cyc_epoch\=44-step\=2114.ckpt --log_dir adv_output/ --gpu 1 --category_name Cyclist --attack
#python main.py --test --cfg cfgs/BAT_Van.yaml --checkpoint pretrained_models/bat_van_epoch\=21-step\=1363.ckpt --log_dir adv_output/ --gpu 1 --category_name Van --attack
#echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#echo "+++++++++++++++++++++++++++++P2B++++++++++++++++++++++++++"
#python main.py --test --cfg cfgs/P2B_Car.yaml --checkpoint pretrained_models/p2b_car_epoch\=38-step\=23789.ckpt --log_dir adv_output/ --gpu 0 --category_name Car --attack
#python main.py --test --cfg cfgs/P2B_Car.yaml --checkpoint pretrained_models/p2b_ped_epoch\=40-step\=5862.ckpt --log_dir adv_output/ --gpu 0 --category_name Pedestrian --attack
#python main.py --test --cfg cfgs/P2B_Car.yaml --checkpoint pretrained_models/p2b_cyc_epoch\=31-step\=1503.ckpt --log_dir adv_output/ --gpu 0 --category_name Cyclist --attack
#python main.py --test --cfg cfgs/P2B_Car.yaml --checkpoint pretrained_models/p2b_van_epoch\=47-step\=2975.ckpt --log_dir adv_output/ --gpu 0 --category_name Van --attack
#echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#echo "+++++++++++++++++++++++++++++M2track++++++++++++++++++++++++++"
#python main.py --test --cfg cfgs/M2Track_kitti.yaml --checkpoint pretrained_models/mmtrack_car.ckpt --log_dir adv_output/ --gpu 0 --category_name Car --attack
#python main.py --test --cfg cfgs/M2Track_kitti.yaml --checkpoint pretrained_models/mmtrack_ped.ckpt --log_dir adv_output/ --gpu 0 --category_name Pedestrian --attack
#python main.py --test --cfg cfgs/M2Track_kitti.yaml --checkpoint pretrained_models/mmtrack_cyc_epoch\=43-step\=2067.ckpt --log_dir adv_output/ --gpu 0 --category_name Cyclist --attack
#python main.py --test --cfg cfgs/M2Track_kitti.yaml --checkpoint pretrained_models/mmtrack_van_epoch\=19-step\=1239.ckpt --log_dir adv_output/ --gpu 0 --category_name Van --attack
#echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
#echo "+++++++++++++++++++++++++++++STNet++++++++++++++++++++++++++"
#python main.py --test --cfg cfgs/STNet_Car.yaml --checkpoint pretrained_models/stnet_car_eepoch\=13-step\=8553.ckpt --log_dir adv_output/ --gpu 1 --category_name Car --attack
#python main.py --test --cfg cfgs/STNet_Car.yaml --checkpoint pretrained_models/stnet_ped_epoch\=8-step\=1295.ckpt --log_dir adv_output/ --gpu 1 --category_name Pedestrian --attack
#python main.py --test --cfg cfgs/STNet_Car.yaml --checkpoint pretrained_models/stnet_cyc_epoch\=47-step\=2303.ckpt --log_dir adv_output/ --gpu 1 --category_name Cyclist --attack
#python main.py --test --cfg cfgs/STNet_Car.yaml --checkpoint pretrained_models/stnet_van_epoch\=5-step\=377.ckpt --log_dir adv_output/ --gpu 1 --category_name Van --attack
#echo "=============================Nuscenes=============================="
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "+++++++++++++++++++++++++++++M2track nuscenes++++++++++++++++++++++++++"
python main.py --test --cfg cfgs/M2Track_nuscene.yaml --checkpoint pretrained_models/nuscenes_mmtrack_car.ckpt --log_dir adv_output/ --gpu 1 --category_name Car --attack
python main.py --test --cfg cfgs/M2Track_nuscene.yaml --checkpoint pretrained_models/nuscenes_mmtrack_ped_epoch=49-step=147449.ckpt --log_dir adv_output/ --gpu 1 --category_name Pedestrian --attack
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "+++++++++++++++++++++++++++++STNet nuscenes++++++++++++++++++++++++++++"
python main.py --test --cfg cfgs/STNet_Car_nuscenes.yaml --checkpoint pretrained_models/nuscenes_stnet_car_epoch=30-step=142847.ckpt --log_dir adv_output/ --gpu 1 --category_name Car --attack
python main.py --test --cfg cfgs/STNet_Car_nuscenes.yaml --checkpoint pretrained_models/nuscenes_stnet_ped_epoch=8-step=19907.ckpt --log_dir adv_output/ --gpu 1 --category_name Pedestrian --attack
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "+++++++++++++++++++++++++++++BAT nuscenes++++++++++++++++++++++++++++++"
python main.py --test --cfg cfgs/BAT_Car_nuscenes.yaml --checkpoint pretrained_models/nuscenes_bat_car.ckpt --log_dir adv_output/ --gpu 1 --category_name car --attack
python main.py --test --cfg cfgs/BAT_Car_nuscenes.yaml --checkpoint pretrained_models/nuscenes_bat_ped_epoch\=14-step\=44234.ckpt --log_dir adv_output/ --gpu 1 --category_name Pedestrian --attack
echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
echo "+++++++++++++++++++++++++++++P2B nuscenes++++++++++++++++++++++++++++++"
python main.py --test --cfg cfgs/P2B_Car_nuscenes.yaml --checkpoint pretrained_models/nuscenes_p2b_car_epoch=46-step=288720.ckpt --log_dir adv_output/ --gpu 1 --category_name car --attack
python main.py --test --cfg cfgs/P2B_Car_nuscenes.yaml --checkpoint pretrained_models/nuscenes_p2b_ped_epoch=15-step=47183.ckpt --log_dir adv_output/ --gpu 1 --category_name Pedestrian --attack
