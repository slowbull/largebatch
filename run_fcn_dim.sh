((i=-1))
for lr in 0.01 0.05 0.1 0.5 1 5
do
  #for dim_p in 1 5 10 20 
  for dim_p in 50
  do 
    ((i=i+1))
    ((device=i%4))
    echo " lr $lr! device $device "
    CUDA_VISIBLE_DEVICES=$device python main.py --lr $lr --bn --dim_p $dim_p > fcn_dim/p_${dim_p}_lr_${lr} &
  done
  sleep 1
done

  

