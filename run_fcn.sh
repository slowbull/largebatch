for l in {1..5}
do
  echo " layer $l! "
  #for i in 1 10 100 1000 10000 100000
  for i in 10  100000
  do 
    ((device=l%4))
    CUDA_VISIBLE_DEVICES=$device python main.py --ll --ll_p $i --layer $l --epochs 100 > layer$l/p_$i &
  done
  sleep 1
done

  

