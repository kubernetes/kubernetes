package colors

import (
	"bytes"
	"fmt"
	"io"
)

type noColors struct {
	out     io.Writer
	lastbuf bytes.Buffer
}

func Uncolored(w io.Writer) io.Writer {
	return &noColors{out: w}
}

func (w *noColors) Write(data []byte) (n int, err error) {
	er := bytes.NewBuffer(data)
loop:
	for {
		c1, _, err := er.ReadRune()
		if err != nil {
			break loop
		}
		if c1 != 0x1b {
			fmt.Fprint(w.out, string(c1))
			continue
		}
		c2, _, err := er.ReadRune()
		if err != nil {
			w.lastbuf.WriteRune(c1)
			break loop
		}
		if c2 != 0x5b {
			w.lastbuf.WriteRune(c1)
			w.lastbuf.WriteRune(c2)
			continue
		}

		var buf bytes.Buffer
		for {
			c, _, err := er.ReadRune()
			if err != nil {
				w.lastbuf.WriteRune(c1)
				w.lastbuf.WriteRune(c2)
				w.lastbuf.Write(buf.Bytes())
				break loop
			}
			if ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '@' {
				break
			}
			buf.Write([]byte(string(c)))
		}
	}
	return len(data) - w.lastbuf.Len(), nil
}
