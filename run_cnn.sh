for l in {1..5}
do
  echo " layer $l! "
  for i in  1 10 100 1000 10000 100000 1000000 10000000 
  do 
    ((device=l%4))
    CUDA_VISIBLE_DEVICES=$device python main.py --ll --cnn --ll_p $i --layer $l --epochs 100 > cnn_sigmoid/layer$l/p_$i &
  done
  sleep 1
done

  

