./train  1 ./exps/heb-pos.train n
./decode 1 ./exps/heb-pos.test  heb-pos.train.param
./evaluate heb-pos.test.tagged ./exps/heb-pos.gold 1 n
mv heb-pos.test.tagged ./results/1/
mv heb-pos.test.eval   ./results/1/
mv heb-pos.train.param   ./exps/1/
