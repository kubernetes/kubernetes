package tailfile

import (
	"bytes"
	"errors"
	"os"
)

const blockSize = 1024

var eol = []byte("\n")
var ErrNonPositiveLinesNumber = errors.New("Lines number must be positive")

//TailFile returns last n lines of file f
func TailFile(f *os.File, n int) ([][]byte, error) {
	if n <= 0 {
		return nil, ErrNonPositiveLinesNumber
	}
	size, err := f.Seek(0, os.SEEK_END)
	if err != nil {
		return nil, err
	}
	block := -1
	var data []byte
	var cnt int
	for {
		var b []byte
		step := int64(block * blockSize)
		left := size + step // how many bytes to beginning
		if left < 0 {
			if _, err := f.Seek(0, os.SEEK_SET); err != nil {
				return nil, err
			}
			b = make([]byte, blockSize+left)
			if _, err := f.Read(b); err != nil {
				return nil, err
			}
			data = append(b, data...)
			break
		} else {
			b = make([]byte, blockSize)
			if _, err := f.Seek(step, os.SEEK_END); err != nil {
				return nil, err
			}
			if _, err := f.Read(b); err != nil {
				return nil, err
			}
			data = append(b, data...)
		}
		cnt += bytes.Count(b, eol)
		if cnt > n {
			break
		}
		block--
	}
	lines := bytes.Split(data, eol)
	if n < len(lines) {
		return lines[len(lines)-n-1 : len(lines)-1], nil
	}
	return lines[:len(lines)-1], nil
}
