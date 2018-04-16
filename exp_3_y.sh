./train  3 ./exps/heb-pos.train y
./decode 3 ./exps/heb-pos.test  heb-pos.train.lex heb-pos.train.gram
./evaluate heb-pos.test.tagged ./exps/heb-pos.gold 3 y
mv heb-pos.test.tagged ./results/3_y/
mv heb-pos.test.eval   ./results/3_y/
mv heb-pos.train.lex   ./exps/3_y/
mv heb-pos.train.gram  ./exps/3_y/
