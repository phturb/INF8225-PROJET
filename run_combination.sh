# for p in '' '-pm'; do
#     for d in '' '-dd'; do
#         for u in '' '-du'; do
#             for i in '' '-di'; do
#                 for n in '' '-ny'; do
#                     for m in '' '-ns 4'; do
#                         echo "$p $d $u $i $n $m -wa 1000 -mt 1000"
#                         python ./run_rainbow.py $p $d $u $i $n $m -wa 2000 -mt 2500
#                     done;
#                 done;
#             done;
#         done;
#     done;
# done

WA="-wa 1000 -mt 2000"

python ./run_rainbow.py $WA
python ./run_rainbow.py -dd $WA
python ./run_rainbow.py -du $WA
python ./run_rainbow.py -di $WA
python ./run_rainbow.py -pm $WA
python ./run_rainbow.py -ny $WA
python ./run_rainbow.py -ns 4 $WA
python ./run_rainbow.py -dd -ns 4 $WA
python ./run_rainbow.py -dd -ns 4 -du $WA
python ./run_rainbow.py -dd -ns 4 -du -pm $WA
python ./run_rainbow.py -dd -ns 4 -du -pm -di $WA
python ./run_rainbow.py -dd -ns 4 -du -pm -di -ny $WA
python ./run_rainbow.py -dd -ns 4 -du -pm -ny $WA