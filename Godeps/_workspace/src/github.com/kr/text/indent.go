package text

import (
	"io"
)

// Indent inserts prefix at the beginning of each non-empty line of s. The
// end-of-line marker is NL.
func Indent(s, prefix string) string {
	return string(IndentBytes([]byte(s), []byte(prefix)))
}

// IndentBytes inserts prefix at the beginning of each non-empty line of b.
// The end-of-line marker is NL.
func IndentBytes(b, prefix []byte) []byte {
	var res []byte
	bol := true
	for _, c := range b {
		if bol && c != '\n' {
			res = append(res, prefix...)
		}
		res = append(res, c)
		bol = c == '\n'
	}
	return res
}

// Writer indents each line of its input.
type indentWriter struct {
	w   io.Writer
	bol bool
	pre [][]byte
	sel int
	off int
}

// NewIndentWriter makes a new write filter that indents the input
// lines. Each line is prefixed in order with the corresponding
// element of pre. If there are more lines than elements, the last
// element of pre is repeated for each subsequent line.
func NewIndentWriter(w io.Writer, pre ...[]byte) io.Writer {
	return &indentWriter{
		w:   w,
		pre: pre,
		bol: true,
	}
}

// The only errors returned are from the underlying indentWriter.
func (w *indentWriter) Write(p []byte) (n int, err error) {
	for _, c := range p {
		if w.bol {
			var i int
			i, err = w.w.Write(w.pre[w.sel][w.off:])
			w.off += i
			if err != nil {
				return n, err
			}
		}
		_, err = w.w.Write([]byte{c})
		if err != nil {
			return n, err
		}
		n++
		w.bol = c == '\n'
		if w.bol {
			w.off = 0
			if w.sel < len(w.pre)-1 {
				w.sel++
			}
		}
	}
	return n, nil
}
