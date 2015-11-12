package word_vectors

import (
	"math"
	"time"
)

const (
	//model params
	// sane defaults for training parameters in the case that we need them, see similar named fields in WordVecModel struct. See also original project for where I got these defaults (https://code.google.com/p/word2vec/) or my github mirror: (https://github.com/jbowles/word2vec/blob/master/word2vec.c#L40).
	ALPHA           float64 = 0.025 // learning rate defualt for skip-gram
	BINARY_F        bool    = false
	BAG_OF_WORDS    bool    = true //default for cbow is 0.05; if false use alpha
	DEBUG_MODE      int     = 2
	IN_VOCAB_FILE   string  = ""
	ITER            int     = 5
	KMEANS_CLASSES  int     = 0
	LAYER1_VEC_SIZE int     = 100 //Set size of word vectors
	MIN_COUNT       int     = 5
	MIN_REDUCE      int     = 1
	NEG_SAMPLING    int     = 5 //common values are 3 - 10 (0 = not used)
	NUM_THREADS     int     = 12
	OUT_VOCAB_FILE  string  = ""
	SAMPLE          float64 = 1e-3  //useful range is (0, 1e-5)
	SOFTMAX         bool    = false //use Hierarchical Softmax
	WINDOW_SKIP_LEN int     = 5
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
	SEEK_SET    int    = 0
	NEXT_RANDOM uint64 = 1
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
	PHRASE_THRESHOLD float64 = 100
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
	//Error           error
	//ExpTableSize    int
	//MaxExp          float64
	Alpha           float64 //Set the starting learning rate; default is 0.025 for skip-gram,  and 0.05 for CBOW
	Binaryf         bool    //Save the resulting vectors in binary file; default is false (off)
	Cbow            bool    //Use the continuous bag of words model; default is true (use false for skip-gram model)
	DebugMode       int     //Set the debug mode (default = 2 = more info during training)
	ExpTable        []float64
	FileSize        int64
	InVocabFile     string //The vocabulary will be read from <file>, not constructed from the training data, if "" then will generate
	Iter            int
	KmeansClasses   int //Output word classes rather than word vectors; default number of classes is 0 (vectors are written)
	Layer1VecSize   int //Set size of word vectors; default is 100
	MaxCodeLen      int
	MaxSentenceLen  int
	MaxStringLen    int
	MinCount        int //This will discard words that appear less than n times; default is 5
	MinReduce       int
	NegSampling     int //Number of negative examples; default is 5, common values are 3 - 10 (0 = not used)
	NextRandom      uint64
	NumThreads      int
	OutVocabFile    string //The vocabulary will be saved to <file>; if no file name given, i.e. "", then it won't be saved
	OutputFile      string
	Sample          float64 //Set threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)
	SoftMax         bool    //Use Hierarchical Softmax; default is 0 (not used)
	Start           time.Time
	StartingAlpha   float64
	Syn0            []float64
	Syn1            []float64
	Syn1neg         []float64
	Table           []int
	Threshold       float64
	TrainFile       string //Use text data from <file> to train the model
	TrainWords      int64
	Vocab           VocabSlice
	VocabHash       []int
	VocabHashSize   int
	VocabMaxSize    int
	VocabSize       int
	WindowSkipLen   int // Set max skip length between words; default is 5
	WordCountActual int64
}

func PreComputeExpTable() (expTable []float64) {
	expTable = make([]float64, EXP_TABLE_SIZE+1)

	for i := 0; i < EXP_TABLE_SIZE; i++ {
		expTable[i] = math.Exp((float64(i) / float64(EXP_TABLE_SIZE) * (2 - 1)) * MAX_EXP) // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1)                                      // Precompute f(x) = x / (x + 1)
	}
	return
}

// NewWordVecModelForWord creates a new WordVecModel for the word2vec technique. It offers many arguments to modify paramters to the model. Many arguments will not need to be defined and can use the defaults.
// An example using all the defaults:
//	  func NewWordVecModelForWord(BAG_OF_WORDS, "training.txt", "word2vec_vectors", IN_VOCAB_FILE, OUT_VOCAB_FILE, SAMPLE, BINARY_F, SOFTMAX, DEBU_GMODE, ITER, KMEANS_CLASSES, LAYER1_VEC_SIZE, MIN_COUNT, MIN_REDUCE, NEG_SAMPLING, NUM_THREADS, WINDOW_SKIP_LEN)
func NewWordVecModelForWord(
	bagOfWords bool,
	trainFile string,
	outFile string,
	inVocabFile string,
	outVocabFile string,
	sample float64,
	binaryF bool,
	softMax bool,
	debugMode,
	iterations,
	kmeansClasses,
	layer1Size,
	minCount,
	minReduce,
	negSample,
	numThread,
	windowSkipLen int) *WordVecModel {
	return &WordVecModel{
		//VocabHashSize:  VOCAB_HASH_SIZE_WORD,
		//VocabMaxSize:    MAX_VOCAB_WORD,
		Alpha:           ALPHA,
		Binaryf:         binaryF,
		Cbow:            bagOfWords,
		DebugMode:       debugMode,
		ExpTable:        PreComputeExpTable(),
		FileSize:        0,
		InVocabFile:     inVocabFile,
		Iter:            iterations,
		KmeansClasses:   kmeansClasses,
		Layer1VecSize:   layer1Size,
		MaxCodeLen:      MAX_CODE_LENGTH,
		MaxSentenceLen:  MAX_SENTENCE_WORD,
		MaxStringLen:    MAX_STRING_WORD,
		MinCount:        minCount,
		MinReduce:       minReduce,
		NegSampling:     negSample,
		NextRandom:      NEXT_RANDOM,
		NumThreads:      numThread,
		OutVocabFile:    outVocabFile,
		OutputFile:      outFile,
		Sample:          sample,
		SoftMax:         softMax,
		Start:           time.Now(),
		StartingAlpha:   0,
		Syn0:            []float64{},
		Syn1:            []float64{},
		Syn1neg:         []float64{},
		Table:           []int{},
		TrainFile:       trainFile,
		TrainWords:      0,
		Vocab:           make(VocabSlice, MAX_VOCAB_WORD),
		VocabHash:       make([]int, VOCAB_HASH_SIZE_WORD),
		VocabSize:       0,
		WindowSkipLen:   windowSkipLen,
		WordCountActual: 0,
	}
}

func NewWordVecModelForPhrase(trainFile string, outFile string, threshold float64, debugMode, minCount, minReduce int) *WordVecModel {
	return &WordVecModel{
		TrainFile:    trainFile,
		OutputFile:   outFile,
		DebugMode:    debugMode,
		MinCount:     minCount,
		MinReduce:    minReduce,
		MaxStringLen: MAX_STRING_PHRASE,
		Vocab:        make(VocabSlice, MAX_VOCAB_PHRASE),
		VocabHash:    make([]int, VOCAB_HASH_SIZE_PHRASE),
		//VocabHashSize: VOCAB_HASH_SIZE_PHRASE,
		//VocabMaxSize:  MAX_VOCAB_PHRASE,
		VocabSize:  0,
		TrainWords: 0,
		Threshold:  threshold,
		NextRandom: NEXT_RANDOM,
	}
}

// DefaultWordVecModelForWord creates a new WordVecModel for the word2vec technique using all the sane defaults. This is just simply a convenience for having to use all the default parameters.
func DefaultWordVecModelForWord(trainFile, outFile string) *WordVecModel {
	return &WordVecModel{
		Alpha:           ALPHA,
		Binaryf:         BINARY_F,
		Cbow:            BAG_OF_WORDS,
		DebugMode:       DEBUG_MODE,
		ExpTable:        PreComputeExpTable(),
		FileSize:        0,
		InVocabFile:     IN_VOCAB_FILE,
		Iter:            ITER,
		KmeansClasses:   KMEANS_CLASSES,
		Layer1VecSize:   LAYER1_VEC_SIZE,
		MaxCodeLen:      MAX_CODE_LENGTH,
		MaxSentenceLen:  MAX_SENTENCE_WORD,
		MaxStringLen:    MAX_STRING_WORD,
		MinCount:        MIN_COUNT,
		MinReduce:       MIN_REDUCE,
		NegSampling:     NEG_SAMPLING,
		NextRandom:      NEXT_RANDOM,
		NumThreads:      NUM_THREADS,
		OutVocabFile:    IN_VOCAB_FILE,
		OutputFile:      outFile,
		Sample:          SAMPLE,
		SoftMax:         SOFTMAX,
		Start:           time.Now(),
		StartingAlpha:   0,
		Syn0:            []float64{},
		Syn1:            []float64{},
		Syn1neg:         []float64{},
		Table:           []int{},
		TrainFile:       trainFile,
		TrainWords:      0,
		Vocab:           make(VocabSlice, MAX_VOCAB_WORD),
		VocabHash:       make([]int, VOCAB_HASH_SIZE_WORD),
		VocabSize:       0,
		WindowSkipLen:   WINDOW_SKIP_LEN,
		WordCountActual: 0,
	}
}
