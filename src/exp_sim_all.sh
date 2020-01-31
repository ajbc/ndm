for proc in 1 2 3 4 5
do
    for seed in 123864 294851 312559 409231 566710 600126 798102 882308 900001 999998
    do
        domain="positive"
        dist="gamma"
        echo "$proc $domain ($dist) $seed"
        (python main.py --data sim/dat/${domain}_proc${proc}_${seed}/data.hdf5 \
            --out sim/out3/${domain}_proc${proc}_${seed} \
            --K 10 --fix_K \
            --save_freq 1 \
            --min_iter 25 \
            --max_iter 200 \
            --seed $seed \
            --dist $dist \
            > sim/out3/${domain}_proc${proc}_${seed}.out \
            2> sim/out3/${domain}_proc${proc}_${seed}.err &)
        # --rho 100 \

        domain="unit"
        dist="beta"
        echo "$proc $domain ($dist) $seed"
        (python main.py --data sim/dat/${domain}_proc${proc}_${seed}/data.hdf5 \
            --out sim/out3/${domain}_proc${proc}_${seed} \
            --K 10 --fix_K \
            --save_freq 1 \
            --min_iter 25 \
            --max_iter 200 \
            --seed $seed \
            --dist $dist \
            > sim/out3/${domain}_proc${proc}_${seed}.out \
            2> sim/out3/${domain}_proc${proc}_${seed}.err &)

        domain="integer"
        dist="poisson"
        echo "$proc $domain ($dist) $seed"
        (python main.py --data sim/dat/${domain}_proc${proc}_${seed}/data.hdf5 \
            --out sim/out3/${domain}_proc${proc}_${seed} \
            --K 10 --fix_K \
            --save_freq 1 \
            --min_iter 25 \
            --max_iter 200 \
            --seed $seed \
            --dist $dist \
            > sim/out3/${domain}_proc${proc}_${seed}.out \
            2> sim/out3/${domain}_proc${proc}_${seed}.err &)

        domain="real"
        dist="normal"
        echo "$proc $domain ($dist) $seed"
        python main.py --data sim/dat/${domain}_proc${proc}_${seed}/data.hdf5 \
            --out sim/out3/${domain}_proc${proc}_${seed} \
            --K 10 --fix_K \
            --save_freq 1 \
            --min_iter 25 \
            --max_iter 200 \
            --seed $seed \
            --dist $dist \
            > sim/out3/${domain}_proc${proc}_${seed}.out \
            2> sim/out3/${domain}_proc${proc}_${seed}.err
    done
done
