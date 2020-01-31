for seed in 123864 294851 312559 409231 566710 600126 798102 882308 900001 999998
do
    #for domain in "positive"
    for domain in "real" "positive" "unit" "integer"
    do
        #for proc in 1 2 3 4
        #for proc in 5
        for proc in 1
        do
            echo $seed $domain $proc
            mseed=
            python simulate.py --out sim/dat/${domain}_proc${proc}_${seed} \
                --K 10 \
                --N 1000 \
                --M 20 \
                --seed $((seed+proc)) \
                --domain $domain \
                --proc $proc \
                > sim/dat/${domain}_proc${proc}_${seed}.out 2> sim/dat/${domain}_proc${proc}_${seed}.err
        done
    done
done
