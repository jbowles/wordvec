//package unigram_table_test //do a blackbox test?
package wordvec

import (
	//"github.com/jbowles/word_vectors"
	"bufio"
	"os"
	"testing"
)

var testFileForReadWord string = "test_data/read_word_test.txt"

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
	{"cars", 19391274},
	{"bicycle", 26888540},
	{"bicycles", 20803279},
	{"dixon", 2219939},
	{"mountain", 11380076},
	{"considering", 20680214},
	{"martüa", 18960533},
	{"martua", 23283211},
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

var readwordtests = []struct {
	expected string
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
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
	)
	for _, w := range readwordtests {
		wrd, _ := mv.ReadWord(reader)
		if wrd != w.expected {
			t.Errorf("ReadWord(reader) = %s, expected %s", wrd, w.expected)
		}
	}
}
