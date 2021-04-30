WA="-ms 25000 -mt 2500 -wa 500"

# python ./run_rainbow.py $WA
# python ./run_rainbow.py -di $WA
# python ./run_rainbow.py -dd $WA
# python ./run_rainbow.py -du $WA
# python ./run_rainbow.py -pm $WA
# python ./run_rainbow.py -ny $WA
# python ./run_rainbow.py -ns 4 $WA

# python ./run_rainbow.py -dd -ns 4 -du -pm -di -ny $WA
# python ./run_rainbow.py -dd -ns 4 -du -pm -di $WA
python ./run_rainbow.py -dd -ns 4 -du -pm -ny $WA
python ./run_rainbow.py -dd -ns 4 -du -di -ny $WA
python ./run_rainbow.py -dd -ns 4 -du -pm -di -ny $WA
python ./run_rainbow.py -dd -ns 4 -pm -di -ny $WA
python ./run_rainbow.py -dd -du -pm -di -ny $WA
python ./run_rainbow.py -ns 4 -du -pm -di -ny $WA


# python ./run_rainbow.py -dd -ns 4 -du -pm -di -ny -e brick-breaker $WA
# python ./run_rainbow.py -dd -ns 4 $WA
# python ./run_rainbow.py -dd -ns 4 -du $WA
# python ./run_rainbow.py -dd -ns 4 -du -pm $WA
# python ./run_rainbow.py -dd -ns 4 -du -pm -ny $WA