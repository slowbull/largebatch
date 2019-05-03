((i=-1))
#for lr in 0.01 0.04 0.07 0.1 0.4 0.7 
for lr in 0.01 0.04 0.1 0.2 0.4 0.6 0.8 1
do
  for dim_p in 50
  do 
    ((i=i+1))
    ((device=i%4))
    echo " lr $lr! device $device "
    CUDA_VISIBLE_DEVICES=$device python main.py --lr $lr --cnn --bn --dim_p $dim_p > cnn_dim/p_${dim_p}_lr_${lr} &
  done
  sleep 1
done

  

