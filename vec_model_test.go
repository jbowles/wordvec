//package word_vec_model_test
package word_vectors

import (
	//"github.com/jbowles/word_vectors"
	"fmt"
	"testing"
)

var BZeroLearningRate ModelParams = AlphaOption(0)
var BNoDebug ModelParams = DebugModeOption(0)
var BIter20 ModelParams = IterOption(20)
var BLayer1VecSize200 ModelParams = Layer1VecSizeOption(200)
var BMinCount200 ModelParams = MinCountOption(200)
var BNegSample50 ModelParams = NegSamplingOption(50)
var BUseInputVocabFile ModelParams = InVocabFileOption("input_vocab_file.dat")
var BWriteOutVocabFile ModelParams = OutVocabFileOption("word2vec_output_vocab_file.dat")
var BSampleOverrideOption ModelParams = SampleOption(1e-5)
var BWindowSkipTen ModelParams = WindowSkipLenOption(10)

func BenchmarkPreComputeExpTable(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = PreComputeExpTable()
	}
}

func BenchmarkNewWord2VecDefaulte(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = NewWord2VecModel("training_data.txt", "word2vec_output.txt")

	}
}

func BenchmarkNewWord2VecOptions(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, _ = NewWord2VecModel(
			"training_data.txt",
			"word2vec_output.bin",
			BinaryFileTrue,
			BagOfWordsFalse,
			SoftMaxOptionTrue,
			BZeroLearningRate,
			BNoDebug,
			BIter20,
			BLayer1VecSize200,
			BMinCount200,
			BNegSample50,
			BUseInputVocabFile,
			BWriteOutVocabFile,
			BSampleOverrideOption,
			BWindowSkipTen,
		)

	}
}
func TestPreComputeExpTable(t *testing.T) {
	//expTable := word_vectors.PreComputeExpTable()
	expTable := PreComputeExpTable()
	tableLen := len(expTable)
	tableLenHalf := int(float64(tableLen) * 0.5)
	tableFirst := expTable[0]
	tableLast := expTable[tableLen-1]
	tableHalf := expTable[tableLen-tableLenHalf]

	if tableLen != int(1001) {
		t.Error("table length should be int `1001` but got:", tableLen)
	}

	if tableLenHalf != int(500) {
		t.Error("table length shoudl be int `500` but got:", tableLenHalf)
	}

	if tableFirst != float64(0.5) {
		t.Error("table length should be float64 `0.5` but got:", tableFirst)
	}

	if tableLast != float64(0) {
		t.Error("table length should be float64 `0` but got:", tableLast)
	}
	if tableHalf != float64(0.9528444519169822) {
		t.Error("table length should be float64 `0.9528444519169822` but got:", tableHalf)
	}
}

func TestNewWord2VecDefault(t *testing.T) {
	m, _ := NewWord2VecModel("training_data.txt", "word2vec_output.txt")

	if m.Cbow == false {
		t.Error("cbow should be true but was:", m.Cbow)
	}

	if m.Cbow == true && m.Alpha != ALPHA_CBOW {
		t.Error("alpha learning rate when using cbow should be float64 `0.05` but was:", m.Alpha)
	}
}

func TestNewWord2VecCbowLearningRate(t *testing.T) {
	mv, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
		BagOfWordsFalse,
		SoftMaxOptionTrue,
	)

	if mv.Cbow != false {
		t.Error("cbow should be false but was:", mv.Cbow)
	}

	if mv.Alpha != float64(0.025) {
		t.Error("alpha learning rate when using skip gram should be float64 `0.025` but was:", mv.Alpha)
	}
}

func TestNewWord2VecBinaryFileFormat(t *testing.T) {
	_, err := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.txt",
		BinaryFileTrue,
	)
	errMsg := fmt.Sprintf("%s", err)
	if errMsg != "Specified binary output but output file has '.txt'" {
		t.Error("error message when output file is not correct should be `Specified binary output but output file has '.txt'`, but was:", errMsg)
	}
}

func TestNewWord2VecOptionsMany(t *testing.T) {
	var ZeroLearningRate ModelParams = AlphaOption(0)
	var NoDebug ModelParams = DebugModeOption(0)
	var Iter20 ModelParams = IterOption(20)
	//var KmeansClasses100 ModelParams = KmeansClassesOption(100)
	var Layer1VecSize200 ModelParams = Layer1VecSizeOption(200)
	var MinCount200 ModelParams = MinCountOption(200)
	var NegSample50 ModelParams = NegSamplingOption(50)
	var UseInputVocabFile ModelParams = InVocabFileOption("input_vocab_file.dat")
	var WriteOutVocabFile ModelParams = OutVocabFileOption("word2vec_output_vocab_file.dat")
	var SampleOverrideOption ModelParams = SampleOption(1e-5)
	var WindowSkipTen ModelParams = WindowSkipLenOption(10)
	m, _ := NewWord2VecModel(
		"training_data.txt",
		"word2vec_output.bin",
		BinaryFileTrue,
		BagOfWordsFalse,
		ZeroLearningRate,
		NoDebug,
		Iter20,
		Layer1VecSize200,
		MinCount200,
		NegSample50,
		UseInputVocabFile,
		WriteOutVocabFile,
		SampleOverrideOption,
		WindowSkipTen,
	)

	if m.Binaryf != true {
		t.Error("binary file option should be 'true' but was:", m.Binaryf)
	}
	if m.Cbow != false {
		t.Error("cbow should be false but was:", m.Cbow)
	}

	if m.Alpha != float64(0) {
		t.Error("alpha learning rate option was float64 `0` but was:", m.Alpha)
	}

	if m.DebugMode != int(0) {
		t.Error("debug mode option should be 0, got", m.DebugMode)
	}

	if m.Iter != int(20) {
		t.Error("iterations option should be 20, got", m.Iter)
	}

	if m.Layer1VecSize != int(200) {
		t.Error("layer 1 vector size option should be 200, got", m.Layer1VecSize)
	}

	if m.MinCount != int(200) {
		t.Error("MinCount size option should be 200, got", m.MinCount)
	}
	if m.NegSampling != int(50) {
		t.Error("NegSampling option should be 50, got", m.NegSampling)
	}
	if m.InVocabFile != "input_vocab_file.dat" {
		t.Error("InVocabFile option should be 'input_vocab_file.dat', got", m.InVocabFile)
	}
	if m.OutVocabFile != "word2vec_output_vocab_file.dat" {
		t.Error("InVocabFile option should be 'word2vec_output_vocab_file.dat', got", m.OutVocabFile)
	}
	if m.Sample != float64(1e-5) {
		t.Error("Sample option should be 'float64(1e-5', got", m.Sample)
	}
	if m.WindowSkipLen != int(10) {
		t.Error("WindowSkipTen option should be 10, got", m.WindowSkipLen)
	}

}
