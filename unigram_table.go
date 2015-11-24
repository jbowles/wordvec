package wordvec

import (
	"log"
	"math"
)

// InitUnigramTable creates and seeds the 1-gram table
func (v *VectorModel) InitUnigramTable() {
	log.Println("Init UnigramTable")
	var trainWordPow float64
	var power float64 = 0.75
	var d1 float64

	v.Table = make([]int, TABLE_SIZE)
	for a := 0; a < v.VocabMaxSize; a++ {
		trainWordPow += math.Pow(float64(v.Vocab[a].Cn), power)
	}
	i := 0
	d1 = math.Pow(float64(v.Vocab[0].Cn), power) / trainWordPow
	for n := i; n < TABLE_SIZE; n++ {
		v.Table[n] = i
		if float64(n)/float64(TABLE_SIZE) > d1 {
			i++
			d1 += math.Pow(float64(v.Vocab[i].Cn), power) / trainWordPow
		}
		if i >= v.VocabMaxSize {
			i = v.VocabMaxSize - 1
		}
	}
}
