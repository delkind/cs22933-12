./train  2 ./exps/heb-pos.train y $1 
./decode 2 ./exps/heb-pos.test  heb-pos.train.lex heb-pos.train.gram
./evaluate heb-pos.test.tagged ./exps/heb-pos.gold 2 y
