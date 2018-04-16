./train  2 ./exps/heb-pos.train n
./decode 2 ./exps/heb-pos.test  heb-pos.train.lex heb-pos.train.gram
./evaluate heb-pos.test.tagged ./exps/heb-pos.gold 2 n
mv heb-pos.test.tagged ./results/2_n/
mv heb-pos.test.eval   ./results/2_n/
mv heb-pos.train.lex   ./exps/2_n/
mv heb-pos.train.gram  ./exps/2_n/
