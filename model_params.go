package wordvec

import (
	"errors"
	"fmt"
	"strings"
)

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

// VocabHashSizeOption Sets the size of a vocabulary hash and the hash table size
func VocabHashSizeOption(vocabHashSizeOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.VocabHashSize = vocabHashSizeOption
		v.VocabHash = make([]int, vocabHashSizeOption)
		return nil
	}
}

// WindowSkipLen Set max skip length between words; default is 5.
func WindowSkipLenOption(windowSkipLenOption int) func(v *VectorModel) error {
	return func(v *VectorModel) error {
		v.WindowSkipLen = windowSkipLenOption
		return nil
	}
}
