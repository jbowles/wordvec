package wordvec

import (
	"flag"
	"testing"
)

var vocabrun int

func init() {
	flag.IntVar(&vocabrun, "vocabrun", 0, "pass 1 to run vocab tests: 'go test -vocabrun=1 -v -run VocabFromTrainFile' ")
}

var VocabSize100 ModelParams = VocabHashSizeOption(100)
var VocabSize200 ModelParams = VocabHashSizeOption(200)
var MinCountZero ModelParams = MinCountOption(0)
var VocabWriter ModelParams = VocabOutFileOption("testdata/vocab_write_sample.txt")

var testFileOneForLearnVocab string = "testdata/learn_vocab_training_one.txt"
var testFileTwoForLearnVocab string = "testdata/learn_vocab_training_two.txt"
var testFileThreeForLearnVocab string = "testdata/learn_vocab_training_three.txt"

//index counts are based on the vocabSort, which starts with the highest count token.
var learnvocabtest = []struct {
	targetWord                  string
	expectIndexForTestfileOne   int
	expectIndexForTestfileTwo   int
	expectIndexForTestfileThree int
}{
	{"abacushighcountfirstplace", 1, 1, 1},
	{"abacus", 4, 2, 12},
	{"aaaaa", -1, -1, -1},
	{"aardvark", -1, 14, -1},
	{"aardwolves", 10, -1, 8},
	{"aardwolf", 2, 15, 9},
	{"abaci", 3, -1, 10},
}

func TestLearnVocabFromTrainFileOne(t *testing.T) {
	flag.Parse()
	if vocabrun == 0 {
		t.Skip("Skipping vocabulary tests by default")
	}
	mv, _ := NewWord2VecModel(
		testFileOneForLearnVocab,
		"word2vec_output.txt",
		VocabSize100,
		MinCountZero,
	)
	mv.LearnVocabFromTrainFile()
	//fmt.Printf("%+v", mv.Vocab)

	if mv.TrainWords != 165 {
		t.Error("number of training words should be number of words in file (165), but got", mv.TrainWords)
	}
	if mv.VocabSize != 61 {
		t.Error("vocabulary size should be number of unique tokens (61), but got", mv.VocabSize)
	}

	for _, w := range learnvocabtest {
		wordPosition := mv.SearchVocab(w.targetWord)
		if wordPosition != w.expectIndexForTestfileOne {
			t.Errorf("SearchVocab() for target word %s returned %d, expected %d", w.targetWord, wordPosition, w.expectIndexForTestfileOne)
		}
	}
}

func TestLearnVocabFromTrainFileTwo(t *testing.T) {
	flag.Parse()
	if vocabrun == 0 {
		t.Skip("Skipping vocabulary tests by default")
	}
	mv, _ := NewWord2VecModel(
		testFileTwoForLearnVocab,
		"word2vec_output.txt",
		VocabSize100,
		MinCountZero,
	)
	mv.LearnVocabFromTrainFile()
	//fmt.Printf("%+v", mv.Vocab)

	if mv.TrainWords != 132 {
		t.Error("number of training words should be number of words in file (132), but got", mv.TrainWords)
	}
	if mv.VocabSize != 16 {
		t.Error("vocabulary size should be number of unique tokens (16), but got", mv.VocabSize)
	}

	for _, w := range learnvocabtest {
		wordPosition := mv.SearchVocab(w.targetWord)
		if wordPosition != w.expectIndexForTestfileTwo {
			t.Errorf("SearchVocab() for target word %s returned %d, expected %d", w.targetWord, wordPosition, w.expectIndexForTestfileTwo)
		}
	}
}

// Use file three which is much larger than the MaxVocabSize defined in this test, this will force a reduceVocab(). The test file has `wc -l testdata/learn_vocab_training_three.txt` => 1160, with lots of duplicates. Reducing will only keep words around that ocurr more the MIN_REDUCE (default = 1)... so its throwing out all words with only 1 instance
func TestLearnVocabFromTrainFileReduce(t *testing.T) {
	flag.Parse()
	if vocabrun == 0 {
		t.Skip("Skipping vocabulary tests by default")
	}
	//No longer need to keep MinCount to zero since this larger test file with many redundant tokens.
	mv, _ := NewWord2VecModel(
		testFileThreeForLearnVocab,
		"word2vec_output.txt",
		VocabSize200,
		VocabWriter,
	)
	mv.LearnVocabFromTrainFile()
	//fmt.Printf("%+v", mv.Vocab)

	if mv.TrainWords != 2641 {
		t.Error("number of training words should be number of words in file (2641), but got", mv.TrainWords)
	}
	if mv.VocabSize != 115 {
		t.Error("vocabulary size should be number of unique tokens (114), but got", mv.VocabSize)
	}

	for _, w := range learnvocabtest {
		wordPosition := mv.SearchVocab(w.targetWord)
		if wordPosition != w.expectIndexForTestfileThree {
			t.Errorf("SearchVocab() for target word %s returned %d, expected %d", w.targetWord, wordPosition, w.expectIndexForTestfileThree)
		}
	}
	mv.SaveVocab()
}

/*
TODO add this test

 SaveVocab()

 and this one too:

func TestAddWordToVocab(t *testing.T) {
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
		MxVocabSize,
	)
}
*/
