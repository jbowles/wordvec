//package unigram_table_test //do a blackbox test?
package wordvec

import (
	//"github.com/jbowles/word_vectors"
	"testing"
)

func TestUnigramTable(t *testing.T) {
	var tableSizeExpect int = 100000000
	var vocabSizeExpect int = 1000

	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
	)

	mv.InitUnigramTable()

	tableSizeActual := len(mv.Table)
	vocabSizeActual := len(mv.Vocab)

	if tableSizeActual != tableSizeExpect {
		t.Error("expected table size to be '100000000', got: ", tableSizeActual)
	}

	if vocabSizeActual != vocabSizeExpect {
		t.Error("expected table size to be '1000', got: ", vocabSizeActual)
	}
}
