package wordvec

import (
	"bufio"
	"bytes"
	"io"
)

// ReadWord reads a single word from a file, assuming space OR tab OR EOL to be word boundaries
func (v *VectorModel) ReadWord(r *bufio.Reader) (word string, err error) {
	var scn int = 0
	var char byte
	var buf bytes.Buffer
	for {
		char, err = r.ReadByte()
		if err == io.EOF {
			break
		}
		//(carriage return,13)=>' '
		if char == 13 {
			continue
		}
		//(space,32)=>' ', (tab, 9)=>'\t', (linefeed, 10)=>'\n'
		if (char == 32) || (char == 9) || (char == 10) {
			if scn > 0 {
				if char == 10 {
					err = r.UnreadByte()
					if err != nil {
						return
					}
				}
				break
			}
			if char == 10 {
				word = "</s>"
				return
			} else {
				continue
			}
		}
		err = buf.WriteByte(char)
		if err != nil {
			return
		}
		scn++
	}
	if scn >= v.MaxStringLen { //truncate words that are too long
		buf.Truncate(v.MaxStringLen)
	}
	word = buf.String()
	return
}

// ReadWordIndex Reads a word and returns its index in the vocabulary
func (v *VectorModel) ReadWordIndex(buf *bufio.Reader) (int, error) {
	//var word string
	word, err := v.ReadWord(buf)
	if err == io.EOF {
		return -1, err
	}
	return v.SearchVocab(word), nil
}

// GetWordHash returns unisgned int hash value of a word
func (v *VectorModel) GetWordHash(word string) uint {
	var hash uint = 0
	for a := 0; a < len(word); a++ {
		hash = hash*257 + uint(word[a])
	}
	hash = hash % uint(v.VocabHashSize)
	return hash
}
