((device=0))
#for lr in 0.01 0.05 0.1 0.5 
for lr in 0.01 1 2.5
do
  for dim_p in 1 5  
  do 
    ((device=(device+1)%2+2))
    echo " lr $lr! device $device "
    CUDA_VISIBLE_DEVICES=$device python main_cifar.py --lr $lr --dim_p $dim_p --epochs 200 > resnet_dim/p_${dim_p}_lr_${lr} &
  done
  sleep 1
done

  

