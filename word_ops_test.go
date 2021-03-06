//package unigram_table_test //do a blackbox test?
package wordvec

import (
	//"github.com/jbowles/word_vectors"
	"bufio"
	"os"
	"testing"
)

// TODO: if this file does not exist process will just hang forever.... need to handle the error case!
var testFileForReadWord string = "testdata/read_word_test.txt"

func BenchmarkReadWord(b *testing.B) {
	var reader *bufio.Reader
	f, _ := os.Open(testFileForReadWord)
	reader = bufio.NewReader(f)
	wvmB, _ := NewWord2VecModel("training_data.txt", "word2vec_output.txt")
	for i := 0; i < b.N; i++ {
		_, _ = wvmB.ReadWord(reader)

	}
}

var wordhashtests = []struct {
	s    string
	hash uint
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
	{"8437289hfnkdj0owri3925yrheijfi9yr8932yhfbucndhfjioeqw", 3210550},
}

func TestGetWordHash(t *testing.T) {
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
	)

	for _, w := range wordhashtests {
		hsh := mv.GetWordHash(w.s)
		if hsh != w.hash {
			t.Errorf("GetWordHash(%s) = %d, want %d", w.s, hsh, w.hash)
		}
	}
}

// expect the index to be -1 because it is not part of a vocabulary
var readwordtests = []struct {
	expectedWord string
}{
	{"line1word1"},
	{"line1word2"},
	{"line1word3"},
	{"</s>"},
	{"line2word1"},
	{"</s>"},
	{"</s>"},
	{"line4word1"},
	{"</s>"},
	{"jdkfhdskfhdhsfjdsfdhsuafrioeujidjsiojiofhueihsbdkljfadsiofvujdhksifoewuiihdijfwb--line5truncateword1"},
	{"</s>"},
	{"line6word1"},
	{"line6word2"},
}

func TestReadWord(t *testing.T) {
	var reader *bufio.Reader
	f, ferr := os.Open(testFileForReadWord)
	if ferr != nil {
		t.Error(ferr)
	}
	reader = bufio.NewReader(f)
	//make sure to handle error EVEN IN THE TEST, OR PROCESS WILL JUST HANG!!
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
	)
	for _, w := range readwordtests {
		wrd, _ := mv.ReadWord(reader)
		//t.Log("got word: ", wrd)  //output word text for debugging
		if wrd != w.expectedWord {
			t.Errorf("ReadWord(reader) = %s, expected %s", wrd, w.expectedWord)
		}
	}
}

/*
func TestReadWrdIndex(t *testing.T) {
	var reader *bufio.Reader
	f, ferr := os.Open(testFileForReadWord)
	if ferr != nil {
		t.Error(ferr)
	}
	reader = bufio.NewReader(f)
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
	)
	for _, w := range readwordtests {
		idx, _ := mv.ReadWordIndex(reader)
		if idx != w.expectedIndex {
			t.Errorf("ReadWord(reader) = %d, expected %d", idx, w.expectedIndex)
		}
	}
}
*/
