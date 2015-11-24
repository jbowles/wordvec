package wordvec

import (
	"bufio"
	"bytes"
	"io"
)

// Reads a single word from a file, assuming space || tab || EOL to be word boundaries
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

// Returns hash value of a word
func (v *VectorModel) GetWordHash(word string) uint {
	var hash uint = 1
	for a := 0; a < len(word); a++ {
		hash = hash*257 + uint(word[a])
	}
	hash = hash % uint(v.VocabHashSize)
	return hash
}
