for data in amazon_reviews-kitchen amazon_reviews-electronics amazon_reviews-dvd amazon_reviews-books
do
	for crossdomain in amazon_reviews/kitchen amazon_reviews/electronics amazon_reviews/dvd amazon_reviews/books #full
	    do
	    	alternate=${crossdomain/\//-}
	    	if [ $alternate == $data ]; then
	    		continue
	    	fi
	    	for model in /mnt/lerna/models/"$data"_aleatorics_M5_concrete /mnt/lerna/models/"$data"_aleatorics_M5 /mnt/lerna/models/"$data"_baseline_M5_concrete /mnt/lerna/models/"$data"_baseline_M5 /mnt/lerna/models/"$data"_nodropout_M5 
        	do
	        	for i in M0_ M1_ M2_ M3_ M4_
	        	do
				    if [ ! -f $model/$i$alternate"_fulleval.pickle" ] ; then
		        		python3 ../evaluate.py $model -i $i -d $crossdomain"_full"
		        	else
		        		echo "already evaluated"
				    fi    
	        	done
	        	if [ ! -f $model/$alternate"_fulleval.pickle" ] ; then
	        	#python3 ../compare.py  $model -i $alternate
	        		python3 ../compare.py  $model -i $alternate"_full"
	            #python3 ../ood.py $model -i $crossdomain -c -o #DETECTION setup
	        	fi
	        done

	  #       for model in /mnt/lerna/models/"$data"_aleatorics_concrete /mnt/lerna/models/"$data"_baseline_concrete /mnt/lerna/models/"$data"_aleatorics /mnt/lerna/models/"$data"_baseline /mnt/lerna/models/"$data"_nodropout
	  #       do
			# 	python3 ../evaluate.py $model -d $crossdomain"_full"
			# done
        #done
    done
	# python3 compare.py /mnt/lerna/models -d $data 
	# python3 compare.py /mnt/lerna/models -d $data -b
    # for model in /mnt/lerna/models/"$data"_aleatorics_M5_concrete /mnt/lerna/models/"$data"_aleatorics_M5 /mnt/lerna/models/"$data"_baseline_M5_concrete /mnt/lerna/models/"$data"_baseline_M5 /mnt/lerna/models/"$data"_nodropout_M5 
    # do
    #     if [ ! -d $model ] ; then
    #         echo "missing " $model
    #     fi
    # 	python3 get_single.py $model
    # done
done