((device=0))
for l in 1 2 3 4 5 6 7 8
do
  for i in  1 10 100 1000
  do 
    ((device=(device+1)%4))
    ((name=l-1))
    CUDA_VISIBLE_DEVICES=$device python main_cifar.py --ll --no_bn --ll_p $i --layer $l  --epochs 200 > resnet/layer$name/p_$i &
    echo $device
  done
  sleep 1
done

  

