package wordvec

import "log"

/*
CreateBinaryTree creates a binary Huffman tree using word counts.

Huffman encoding is a type of variable-length encoding that is based on the actual character frequencies in a given document.

For Huffman trees: http://mathworld.wolfram.com/HuffmanCoding.html, http://www.fas.harvard.edu/~cscie119/lectures/trees.pdf, https://home.cse.ust.hk/~dekai/271/notes/L15/L15.pdf... and many more.

Frequent words will have short unique binary codes
*/
func (v *VectorModel) CreateBinaryTree() {
	log.Println("Create Binary Tree")
	var min1i, min2i, pos1, pos2 int
	var point []int = make([]int, MAX_CODE_LENGTH)
	var code []byte = make([]byte, MAX_CODE_LENGTH)
	var count []int64 = make([]int64, v.VocabSize*2+1)
	var binaryt []int = make([]int, v.VocabSize*2+1)
	var parentNode []int = make([]int, v.VocabSize*2+1)
	for a := 0; a < v.VocabSize; a++ {
		count[a] = int64(v.Vocab[a].Count)
	}
	for b := v.VocabSize; b < v.VocabSize*2; b++ {
		count[b] = 1e15
	}

	pos1 = v.VocabSize - 1
	pos2 = v.VocabSize
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for c := 0; c < v.VocabSize-1; c++ {
		// First, find the 2 smallest nodes: min1, min2
		if pos1 >= 0 {
			if count[pos1] < count[pos2] {
				min1i = pos1
				pos1--
			} else {
				min1i = pos2
				pos2++
			}
		} else {
			min1i = pos2
			pos2++
		}
		if pos1 >= 0 {
			if count[pos1] < count[pos2] {
				min2i = pos1
				pos1--
			} else {
				min2i = pos2
				pos2++
			}
		} else {
			min2i = pos2
			pos2++
		}
		count[v.VocabSize+c] = count[min1i] + count[min2i]
		parentNode[min1i] = v.VocabSize + c
		parentNode[min2i] = v.VocabSize + c
		binaryt[min2i] = 1
	}
	// Now assign binary code to each vocab word
	for d := 0; d < v.VocabSize; d++ {
		e := d
		i := 0
		for {
			code[i] = byte(binaryt[e])
			point[i] = e
			i++
			e = parentNode[e]
			if e == (v.VocabSize*2)-2 {
				break
			}
		}
		v.Vocab[d].Codelen = byte(i)
		v.Vocab[d].Point[0] = v.VocabSize - 2
		for j := 0; j < i; j++ {
			v.Vocab[d].Code[i-j-1] = code[j]
			v.Vocab[d].Point[i-j] = point[j] - v.VocabSize
		}
	}
}
