./train  2 ./exps/heb-pos.train y
./decode 2 ./exps/isha/isha.test heb-pos.train.lex heb-pos.train.gram
./evaluate isha.test.tagged ./exps/isha/isha.gold 2 y
mv isha.test.tagged ./results/isha/
mv isha.test.eval   ./results/isha/
mv heb-pos.train.lex   ./exps/isha/
mv heb-pos.train.gram  ./exps/isha/
