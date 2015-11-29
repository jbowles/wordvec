package wordvec

/*
  VOCABUALRY SUPPORT IS NOT PROVIDED YET SO NO NEED FOR THESE

var emptyvocab int

func init() {
	flag.IntVar(&emptyvocab, "emptyvocab", 0, "pass 1 to run empty vocab search test")
}

var MxVocabSize ModelParams = VocabHashSizeOption(5)
var wordhashvocab = []struct {
	s    string
	hash int
}{
	{"cars", 6920873},
	{"bicycle", 15366747},
	{"bicycles", 19254094},
	{"dixon", 7326882},
	{"mountain", 9830891},
	{"considering", 20601749},
	{"martüa", 7438740},
	{"martua", 15767562},
	{"", 0},
	{" ", 32},
	{"  ", 289},
	{"8437289hfnkdj0owri3925yrheijfi9yr8932yhfbucndhfjioeqw", 3210550},
}

var emptyvocabtest = []struct {
	s    string
	hash int
}{
	{"this", 1},
	{"that", -1},
	{"those", -1},
}

// test if an empty vocab or missing word from a vocab will return -1 after it has exhausted search as large as the max vocabulary size. Need to skip this in normal testing as it will take a long time to search a large (about 21 million) pre-allocated vocabulary.
// Example: go test -emptyvocab=1 -v -run SearchVocab
func TestSearchEmptyVocab(t *testing.T) {
	flag.Parse()
	if emptyvocab == 0 {
		t.Skip("Skip emptyvocab test; cli flag set: 'emptyvocab=1; note it will take long time'")
	}
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
		//MxVocabSize,
	)

	log.Println("searching empty vocabulary, this will take a long time....")
	for _, e := range emptyvocabtest {
		res := mv.SearchVocab(e.s)
		if res != int(e.hash) {
			t.Errorf("SearchVocab(%s) = %d, want %d", e.s, res, e.hash)
		}
	}
}

func TestAddWordToVocab(t *testing.T) {
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
		MxVocabSize,
	)
}
*/
