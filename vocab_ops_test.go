package wordvec

import "testing"

var MxVocabSize ModelParams = VocabHashSizeOption(100)
var MinCountZero ModelParams = MinCountOption(0)
var testFileOneForLearnVocab string = "test_data/learn_vocab_training_one.txt"
var testFileTwoForLearnVocab string = "test_data/learn_vocab_training_two.txt"
var learnvocabtest = []struct {
	targetWord                string
	expectIndexForTestfileOne int
	expectIndexForTestfileTwo int
}{
	{"aaaaa", -1, -1},
	{"aardvark", -1, 2},
	{"aardwolves", 8, -1},
	{"aardwolf", 9, 3},
	{"abaci", 10, -1},
	{"abacus", 2, 4},
}

func TestLearnVocabFromTrainFileOne(t *testing.T) {
	mv, _ := NewWord2VecModel(
		testFileOneForLearnVocab,
		"word2vec_output.txt",
		MxVocabSize,
		MinCountZero,
	)
	mv.LearnVocabFromTrainFile()
	//fmt.Printf("%+v", mv.Vocab)

	if mv.TrainWords != 113 {
		t.Error("number of training words should be number of words in file (113), but got", mv.TrainWords)
	}
	if mv.VocabSize != 60 {
		t.Error("vocabulary size should be number of unique tokens (60), but got", mv.VocabSize)
	}

	for _, w := range learnvocabtest {
		wordPosition := mv.SearchVocab(w.targetWord)
		if wordPosition != w.expectIndexForTestfileOne {
			t.Errorf("SearchVocab() for target word %s returned %d, expected %d", w.targetWord, wordPosition, w.expectIndexForTestfileOne)
		}
	}
}

func TestLearnVocabFromTrainFileTwo(t *testing.T) {
	mv, _ := NewWord2VecModel(
		testFileTwoForLearnVocab,
		"word2vec_output.txt",
		MxVocabSize,
		MinCountZero,
	)
	mv.LearnVocabFromTrainFile()
	//fmt.Printf("%+v", mv.Vocab)

	if mv.TrainWords != 28 {
		t.Error("number of training words should be number of words in file (28), but got", mv.TrainWords)
	}
	if mv.VocabSize != 15 {
		t.Error("vocabulary size should be number of unique tokens (15), but got", mv.VocabSize)
	}

	for _, w := range learnvocabtest {
		wordPosition := mv.SearchVocab(w.targetWord)
		if wordPosition != w.expectIndexForTestfileTwo {
			t.Errorf("SearchVocab() for target word %s returned %d, expected %d", w.targetWord, wordPosition, w.expectIndexForTestfileTwo)
		}
	}
}

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
