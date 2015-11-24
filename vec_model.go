package wordvec

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"time"
)

const (
	//model params
	// sane defaults for training parameters in the case that we need them, see similar named fields in WordVecModel struct. See also original project for where I got these defaults (https://code.google.com/p/word2vec/) or my github mirror: (https://github.com/jbowles/word2vec/blob/master/word2vec.c#L40).
	ALPHA_SKIP_GRAM float64 = 0.025 // learning rate defualt for skip-gram
	ALPHA_CBOW      float64 = 0.05  // learning rate defualt for continuous bag-of-words model
	BINARY_F        bool    = false
	BAG_OF_WORDS    bool    = true //default learning rate (alpha) for cbow is 0.05; if false use alpha
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
	MAX_STRING_WORD     int = 100
	MAX_SENTENCE_WORD   int = 1000
	MAX_STRING_PHRASE   int = 60
	MAX_STRING_ACCURACY int = 2000 // max string size for w2v accuracy
	MAX_STRING_ANALOGY  int = 2000 // max string size for w2v analogy
	MAX_STRING_DISTANCE int = 2000 // max string size for w2v distance
	MAX_CODE_LENGTH     int = 40
	//position
	N_ACCURACY int = 1  // number of closest words for w2v accuracy
	N_ANALOGY  int = 40 // number of closest words for w2v analogy
	N_DISTANCE int = 40 // number of closest words for w2v distance
	//step through
	SEEK_SET    int    = 0
	NEXT_RANDOM uint64 = 1.0
	//tables
	EXP_TABLE_SIZE float64 = 1000.0
	TABLE_SIZE     int     = 1e8
	MAX_EXP        float64 = 6.0
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
	PHRASE_THRESHOLD float64 = 100.0
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

/*
VectorModel struct contains fields needs to train a wrod2vec or word2phrase model:

Required:
	`OutputFile`      REQUIRED. Use <file> to save the resulting word vectors / word clusters.
	`TrainFile`		  REQUIRED. Use text data from <file> to train the model.

Optional (Option functions are supported):
	`Alpha`			  Sets the starting learning rate; default is 0.025 for skip-gram,  and 0.05 for CBOW.
	`Binaryf`		  Decides if the resulting vectors in binary file; default is false (off).
	`Cbow`			  Uses the continuous bag of words model; default is true (use false for skip-gram model).
	`DebugMode`		  Sets the debug mode (default = 2 = more info during training).
	`InVocabFile`	  The vocabulary will be read from <file>, not constructed from the training data, if "" then program will generate vocab. Default is "".
	`Iter`			  Is the number of iterations of training.
	`KmeansClasses`	  Will output word classes rather than word vectors; default number of classes is 0 (vectors are written).
	`Layer1VecSize`	  Sets size of word vectors; default is 100.
	`MinCount`		  This will discard words that appear less than n times; default is 5.
	`NegSampling`	  Number of negative examples; default is 5, common values are 3 - 10 (0 = not used).
	`OutVocabFile`	  The vocabulary will be saved to <file>; if no file name given, i.e. "", then it won't be saved.
	`Sample`		  Sets threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5).
	`SoftMax`		  Use Hierarchical Softmax; default is false (not used).
	`WindowSkipLen`	  Set max skip length between words; default is 5.
*/
type VectorModel struct {
	Alpha           float64
	Binaryf         bool
	Cbow            bool
	DebugMode       int
	ExpTable        []float64
	FileSize        int64
	InVocabFile     string
	Iter            int
	KmeansClasses   int
	Layer1VecSize   int
	MaxCodeLen      int
	MaxSentenceLen  int
	MaxStringLen    int
	MinCount        int
	MinReduce       int
	NegSampling     int
	NextRandom      uint64
	NumThreads      int
	OutVocabFile    string
	OutputFile      string
	Sample          float64
	SoftMax         bool
	Start           time.Time
	StartingAlpha   float64
	Syn0            []float64
	Syn1            []float64
	Syn1neg         []float64
	Table           []int
	Threshold       float64
	TrainFile       string
	TrainWords      int64
	Vocab           VocabSlice
	VocabHash       []int
	VocabHashSize   int
	VocabMaxSize    int
	VocabSize       int
	WindowSkipLen   int
	WordCountActual int64
}

// PrecomputeExpTable builds the computes an exponent table using EXP_TABLE_SIZE and MAX_EXP
func PreComputeExpTable() (expTable []float64) {
	//expTable = make([]float64, int(EXP_TABLE_SIZE+1))
	expTable = make([]float64, 2)

	for i := 0; i < int(EXP_TABLE_SIZE); i++ {
		expTable[i] = math.Exp((float64(i) / EXP_TABLE_SIZE * (2 - 1)) * MAX_EXP) // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1)                             // Precompute f(x) = x / (x + 1)
	}
	return
}

// NewPhraseVectorModel creates the word vector model struct for running word2phrase modelling. It does not requires the amount of parameters for word2vec and therefore the struct is much smaller and uses only a few of the fields. See NewWord2VecModel for how to use the ModelParams variadic arg.
func NewWord2PhraseModel(trainFile string, outFile string, threshold float64, modelParams ...ModelParams) (*VectorModel, error) {
	vm := &VectorModel{
		TrainFile:     trainFile,
		OutputFile:    outFile,
		DebugMode:     DEBUG_MODE,
		MinCount:      MIN_COUNT,
		MinReduce:     MIN_REDUCE,
		MaxStringLen:  MAX_STRING_PHRASE,
		Vocab:         make(VocabSlice, MAX_VOCAB_PHRASE),
		VocabHash:     make([]int, VOCAB_HASH_SIZE_PHRASE),
		VocabHashSize: VOCAB_HASH_SIZE_PHRASE,
		VocabSize:     0,
		TrainWords:    0,
		Threshold:     PHRASE_THRESHOLD,
		NextRandom:    NEXT_RANDOM,
	}

	for _, mp := range modelParams {
		err := mp(vm)
		if err != nil {
			return &VectorModel{}, err
		}
	}

	return vm, nil
}

/* NewWord2VecModel creates a new WordVecModel for the word2vec technique. It offers many arguments to modify paramters to the model. Many arguments will not need to be defined and can use the defaults. See vec_model_test for more examples.

An example with zero learning rate, no debugging output, doing 2o iterations and using a vocab input file:
  var ZeroLearningRate ModelParams = AlphaOption(0)
  var NoDebug ModelParams = DebugModeOption(0)
  var TwentyIters ModelParams = IterOption(20)
  var VocabFileInput ModelParams = InVocabFileOption("input_vocab_file.dat")

  wvm := NewWord2VecModel("training_data.txt", "word2vec_file.txt", ZeroLearningRate, NoDebug, TwentyIters, BinaryFileTrue, BagOfWordsFalse, VocabFileInput)


An example using k-means classes for words instead of vectors:

  wvm := NewWord2VecModel("training_data.txt", "word2vec_kmeans_file.txt", )

*/

func NewWord2VecModel(trainFile, outFile string, modelParams ...ModelParams) (*VectorModel, error) {
	vm := &VectorModel{
		Alpha:           ALPHA_CBOW,
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
		VocabHashSize:   VOCAB_HASH_SIZE_WORD,
		VocabMaxSize:    MAX_VOCAB_WORD,
		VocabSize:       0,
		WindowSkipLen:   WINDOW_SKIP_LEN,
		WordCountActual: 0,
	}

	for _, mp := range modelParams {
		err := mp(vm)
		if err != nil {
			return &VectorModel{}, err
		}
	}

	return vm, nil
}

// ModelParams is a typed function to support optional model params on creation of a word vec model struct
type ModelParams func(*VectorModel) error

// Alpha Sets the starting learning rate; default is 0.025 for skip-gram,  and 0.05 for CBOW.
// Note, if you wanna change the learning rate for skip-gram you should do it AFTER cbow option has been set.
func AlphaOption(alphaOption float64) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.Alpha = alphaOption
		return nil
	}
}

// BinaryFileOption Decides if the resulting vectors are written in binary file; default is false (off).
func BinaryFileTrue(v *VectorModel) error {
	fname := strings.Split(v.OutputFile, ".")
	if fname[len(fname)-1] != "bin" {
		msg := fmt.Sprintf("Specified binary output but output file has '.%s'", fname[len(fname)-1])
		return errors.New(msg)
	}
	v.Binaryf = true
	return nil
}

// BagOfWordsOption Uses the continuous bag of words model; default is true (use false for skip-gram model).
// Note, if you wanna change the learning rate for skip-gram you should do it after this option has been set.
func BagOfWordsFalse(v *VectorModel) error {
	v.Cbow = false
	v.Alpha = ALPHA_SKIP_GRAM
	return nil
}

// DebugModeOption Sets the debug mode (default = 2 = more info during training).
func DebugModeOption(debugModeOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.DebugMode = debugModeOption
		return nil
	}
}

// InVocabFileOption The vocabulary will be read from <file>, not constructed from the training data. if "" then program will generate vocab. Default is "".
func InVocabFileOption(inVocabFileOption string) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.InVocabFile = inVocabFileOption
		return nil
	}
}

// IterOption The vocabulary will be read from <file>, not constructed from the training data. if "" then program will generate vocab. Default is "".
func IterOption(iterOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.Iter = iterOption
		return nil
	}
}

// KmeansClassesOption Will output word classes rather than word vectors; default number of classes is 0 (vectors are written).
func KmeansClassesOption(kmeansClassesOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.KmeansClasses = kmeansClassesOption
		return nil
	}
}

// Layer1VecSize Sets size of word vectors; default is 100.
func Layer1VecSizeOption(layer1VecSizeOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.Layer1VecSize = layer1VecSizeOption
		return nil
	}
}

// MinCount This will discard words that appear less than n times; default is 5.
func MinCountOption(minCountOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.MinCount = minCountOption
		return nil
	}
}

// NegSampling Number of negative examples; default is 5, common values are 3 - 10 (0 = not used).
func NegSamplingOption(negSamplingOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.NegSampling = negSamplingOption
		return nil
	}
}

// OutVocabFile The vocabulary will be saved to <file>; if no file name given, i.e. "", then it won't be saved.
func OutVocabFileOption(outVocabFileOption string) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.OutVocabFile = outVocabFileOption
		return nil
	}
}

// Sample Sets threshold for occurrence of words. Those that appear with higher frequency in the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5).
func SampleOption(sampleOption float64) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.Sample = sampleOption
		return nil
	}
}

// SoftMax Uses Hierarchical Softmax; default is false (not used).
func SoftMaxOptionTrue(v *VectorModel) error {
	v.SoftMax = true
	return nil
}

// WindowSkipLen Set max skip length between words; default is 5.
func WindowSkipLenOption(windowSkipLenOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.WindowSkipLen = windowSkipLenOption
		return nil
	}
}
