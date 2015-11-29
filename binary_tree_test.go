package wordvec

import "testing"

func TestCreateBinaryTree(t *testing.T) {
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
	)
	mv.CreateBinaryTree()
	//fmt.Printf("vocab: %v", len(mv.Vocab))
	if len(mv.Vocab) != 1000 {
		t.Error("vocab shoudl be 1000, got", len(mv.Vocab))
	}
}
