for p in '' '-pm'; do
    for d in '' '-dd'; do
        for u in '' '-du'; do
            for i in '' '-di'; do
                for n in '' '-ny'; do
                    for m in '' '-ns 4'; do
                        echo "$p $d $u $i $n $m -wa 1000 -mt 1000"
                        python ./run_rainbow.py $p $d $u $i $n $m -wa 1000 -mt 500
                    done;
                done;
            done;
        done;
    done;
done