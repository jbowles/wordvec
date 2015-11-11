package word_vectors

import (
	"bufio"
	"time"
)

const (
	//model params
	ALPHA float64 = 0.025
	// string, token, sentence, etc... max size
	MAX_STRING_WORD     int     = 100
	MAX_SENTENCE_WORD   int     = 1000
	MAX_STRING_PHRASE   int     = 60
	MAX_STRING_ACCURACY int     = 2000 // max string size for w2v accuracy
	MAX_STRING_ANALOGY  int     = 2000 // max string size for w2v analogy
	MAX_STRING_DISTANCE int     = 2000 // max string size for w2v distance
	MAX_CODE_LENGTH     int     = 40
	MAX_EXP             float64 = 6.0
	//position
	N_ACCURACY int = 1  // number of closest words for w2v accuracy
	N_ANALOGY  int = 40 // number of closest words for w2v analogy
	N_DISTANCE int = 40 // number of closest words for w2v distance
	//step through
	SEEK_SET           int    = 0
	NEXT_RANDOM_PHRASE uint64 = 1
	//tables
	EXP_TABLE_SIZE int = 1000
	//vocab size
	MAX_VOCAB_WORD     int = 1000  // max vocab size for w2v distance
	MAX_VOCAB_PHRASE   int = 10000 // max vocab size for w2v distance
	MAX_VOCAB_ANALOGY  int = 50    // max vocab size for w2v analogy query
	MAX_VOCAB_ACCURACY int = 50    // max vocab size for w2v accuracy
	MAX_VOCAB_DISTANCE int = 50    // max vocab size for w2v distance
	//vocab hash size
	VOCAB_HASH_SIZE_WORD   int = 30000000  // Maximum 30 * 0.7 = 21M words in the vocabulary for word model
	VOCAB_HASH_SIZE_PHRASE int = 500000000 // Maximum 500M entries in the vocabulary for phrase model
	//thresholds
	THRESHOLD_PHRASE float64 = 100
)

type VocabWord struct {
	Cn      int
	Point   []int
	Word    string
	Code    []byte
	Codelen byte
}

type VocabSlice []VocabWord

func (vs VocabSlice) Len() int {
	return len(vs)
}

func (vs VocabSlice) Less(i, j int) bool {
	return vs[i].Cn > vs[j].Cn
}

func (vs VocabSlice) Swap(i, j int) {
	tmp := vs[i]
	vs[i] = vs[j]
	vs[j] = tmp
}

// WordVecModel struct contains fields needs to train a wrod2vec or word2phrase model
type WordVecModel struct {
	Reader          *bufio.Reader
	TrainFile       string
	OutputFile      string
	OutVocabFile    string
	InVocabFile     string
	MaxStringLen    int
	MaxSentenceLen  int
	MaxCodeLen      int
	ExpTableSize    int
	MaxExp          float64
	Binaryf         int
	Cbow            bool
	DebugMode       int
	Window          int
	NumThreads      int
	MinCount        int
	MinReduce       int
	Vocab           VocabSlice
	VocabHash       []int
	VocabHashSize   int
	VocabMaxSize    int
	VocabSize       int
	Layer1Size      int
	TrainWords      int64
	WordCountActual int64
	Iter            int
	FileSize        int64
	Classes         int
	Alpha           float64
	StartingAlpha   float64
	Sample          float64
	NegSampling     int
	Syn0            []float64
	Syn1            []float64
	Syn1neg         []float64
	ExpTable        []float64
	Threshold       float64
	NextRandom      uint64
	Iterations      int
	Error           error
	Start           time.Time
}

func NewWordVecModelForWord(
	cbow bool,
	trainFile string,
	outFile string,
	inVocabFile string,
	outVocabFile string,
	debugMode,
	minCount,
	minReduce,
	negSample,
	window,
	numThread,
	iter int) *WordVecModel {
	return &WordVecModel{
		TrainFile:       trainFile,
		OutputFile:      outFile,
		OutVocabFile:    outVocabFile,
		InVocabFile:     inVocabFile,
		DebugMode:       debugMode,
		Binaryf:         0,
		Cbow:            cbow,
		Window:          window,
		NumThreads:      numThread,
		MinCount:        minCount,
		MinReduce:       minReduce,
		MaxStringLen:    MAX_STRING_WORD,
		MaxSentenceLen:  MAX_SENTENCE_WORD,
		MaxCodeLen:      MAX_CODE_LENGTH,
		Vocab:           VocabSlice{},
		VocabHash:       []int{},
		VocabHashSize:   VOCAB_HASH_SIZE_WORD,
		VocabMaxSize:    MAX_VOCAB_WORD,
		VocabSize:       0,
		Layer1Size:      100,
		TrainWords:      0,
		WordCountActual: 0,
		Iterations:      iter,
		FileSize:        0,
		Classes:         0,
		Alpha:           ALPHA,
		StartingAlpha:   0,
		Sample:          1e-3,
		NegSampling:     negSample,
		Syn0:            []float64{},
		Syn1:            []float64{},
		Syn1neg:         []float64{},
		ExpTable:        []float64{},
		Threshold:       THRESHOLD_PHRASE,
		NextRandom:      NEXT_RANDOM_PHRASE,
		Start:           time.Now(),
	}
}

func NewWordVecModelForPhrase(trainFile string, outFile string, debugMode, minCount, minReduce int) *WordVecModel {
	return &WordVecModel{
		TrainFile:     trainFile,
		OutputFile:    outFile,
		DebugMode:     debugMode,
		MinCount:      minCount,
		MinReduce:     minReduce,
		MaxStringLen:  MAX_STRING_PHRASE,
		Vocab:         VocabSlice{},
		VocabHash:     []int{},
		VocabHashSize: VOCAB_HASH_SIZE_PHRASE,
		VocabMaxSize:  MAX_VOCAB_PHRASE,
		VocabSize:     0,
		TrainWords:    0,
		Threshold:     THRESHOLD_PHRASE,
		NextRandom:    NEXT_RANDOM_PHRASE,
	}
}
