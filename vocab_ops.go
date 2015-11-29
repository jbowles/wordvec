package wordvec

import (
	"log"
	"sort"
)

/*
TODO these functions are related to using vocabulary files.
func SaveVocab()               {}
func ReadVocab()               {}
*/

// ResetVocabHashIndices resets all indices of the vocab hash to -1. This is done for querying later on so we can distinguish indexes with zero, that have not been touched since initializatio, from indexes that have been modified. See also RecomputeVocabHash, LearnVocabFromTrainFile().
func (v *VectorModel) ResetVocabHashIndices() {
	for i := 0; i < v.VocabHashSize; i++ {
		v.VocabHash[i] = -1
	}
}

// RecomputeVocabHash modifies the value of the hash by updating it for every hash index that is not -1; i.e., vocab indexes that have been previously modified. See also RecomputeVocabHash, LearnVocabFromTrainFile().
func (v *VectorModel) RecomputeVocabHash(hash uint) uint {
	log.Println("re-compute vocab hash")
	for v.VocabHash[hash] != -1 {
		hash = (hash + 1) % uint(v.VocabHashSize)
	}
	return hash
}

// SearchVocab returns the position of a single word in the vocabulary. If word is not found return -1.
func (v *VectorModel) SearchVocab(word string) int {
	var hash uint = v.GetWordHash(word)

	for {
		if v.VocabHash[hash] == -1 {
			return -1
		}
		if word == v.Vocab[v.VocabHash[hash]].Word {
			return v.VocabHash[hash]
		}
		hash = (hash + 1) % uint(v.VocabHashSize)
	}
	return -1
}

// AddWordToVocab adds a word to the vocabulary. Iterating over the vocabulary and looking for indices that are not -1 is possible becuase SortVocab() and ReduceVocab() set all indexes to -1; only vocab items with word will be be selected. See also RecomputeVocabHash, LearnVocabFromTrainFile().
func (v *VectorModel) AddWordToVocab(word string) int {
	var length int = len(word) + 1
	if length > v.MaxStringLen {
		length = v.MaxStringLen
	}

	v.Vocab[v.VocabSize].Word = word
	v.Vocab[v.VocabSize].Count = 0
	v.VocabSize++

	//reallocate memory if needed
	if v.VocabSize+2 > v.VocabMaxSize {
		//log.Println("realloc memory")
		v.VocabMaxSize += 1000
		v.Vocab = append(v.Vocab, make(VocabSlice, 1000)...)
	}

	hash := v.RecomputeVocabHash(v.GetWordHash(word))

	v.VocabHash[hash] = v.VocabSize - 1
	return v.VocabSize - 1
}

// ReduceVocab reduces the vocabulary by removing infrequent terms. See also ResetVocabHashIndices(), RecomputeVocabHash, LearnVocabFromTrainFile().
func (v *VectorModel) ReduceVocab() {
	log.Println("Reducing Vocabulary")
	var b int = 0
	var hash uint
	for a := 0; a < v.VocabSize; a++ {
		if v.Vocab[a].Count > v.MinReduce {
			v.Vocab[b].Count = v.Vocab[a].Count
			v.Vocab[b].Word = v.Vocab[a].Word
			b++
		} else {
			v.Vocab[a].Word = ""
		}
	}
	v.VocabSize = b
	v.ResetVocabHashIndices()

	for c := 0; c < v.VocabHashSize; c++ {
		//Hash will be re-computed; it is not actual
		hash = v.RecomputeVocabHash(v.GetWordHash(v.Vocab[c].Word))
		v.VocabHash[hash] = c
	}
	v.MinReduce++
}

// SortVocab sorts the vocabulary by frequency using word counts. See also ResetVocabHashIndices(), RecomputeVocabHash, LearnVocabFromTrainFile().
func (v *VectorModel) SortVocab() {
	log.Println("Sorting Vocabulary")
	var hash uint
	//var trainWords int64

	// Sort the vocabulary and keep </s> at the first position
	sort.Sort(v.Vocab[1:])
	// set up hash index for later querying
	v.ResetVocabHashIndices()

	size := v.VocabSize
	v.TrainWords = 0
	for b := 0; b < size; b++ {
		// Words occuring less than VectorModel.MinCount times will be discarded from the vocab
		if (v.Vocab[b].Count < v.MinCount) && (b != 0) {
			v.VocabSize--
			v.Vocab[b].Word = ""
		} else {
			// Hash will be re-computed, after the sorting it is not actual
			hash = v.RecomputeVocabHash(v.GetWordHash(v.Vocab[b].Word))
			v.VocabHash[hash] = b
			//trainWords += int64(v.Vocab[b].Count)
			v.TrainWords += int64(v.Vocab[b].Count)
		}
	}
	v.Vocab = v.Vocab[:v.VocabSize+1]
	// Allocate memory for the binary tree constuction
	for c := 0; c < v.VocabSize; c++ {
		v.Vocab[c].Code = make([]byte, v.MaxCodeLen)
		v.Vocab[c].Point = make([]int, v.MaxCodeLen)
	}
}
func LearnVocabFromTrainFile() {}
