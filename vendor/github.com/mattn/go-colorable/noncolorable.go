package colorable

import (
	"bytes"
	"io"
)

// NonColorable hold writer but remove escape sequence.
type NonColorable struct {
	out io.Writer
}

// NewNonColorable return new instance of Writer which remove escape sequence from Writer.
func NewNonColorable(w io.Writer) io.Writer {
	return &NonColorable{out: w}
}

// Write write data on console
func (w *NonColorable) Write(data []byte) (n int, err error) {
	er := bytes.NewReader(data)
	var bw [1]byte
loop:
	for {
		c1, err := er.ReadByte()
		if err != nil {
			break loop
		}
		if c1 != 0x1b {
			bw[0] = c1
			w.out.Write(bw[:])
			continue
		}
		c2, err := er.ReadByte()
		if err != nil {
			break loop
		}
		if c2 != 0x5b {
			continue
		}

		var buf bytes.Buffer
		for {
			c, err := er.ReadByte()
			if err != nil {
				break loop
			}
			if ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '@' {
				break
			}
			buf.Write([]byte(string(c)))
		}
	}

	return len(data), nil
}
