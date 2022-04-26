/*Package dotwriter implements a buffered Writer for updating progress on the
terminal.
*/
package dotwriter

import (
	"bytes"
	"io"
)

// ESC is the ASCII code for escape character
const ESC = 27

// Writer buffers writes until Flush is called. Flush clears previously written
// lines before writing new lines from the buffer.
type Writer struct {
	out       io.Writer
	buf       bytes.Buffer
	lineCount int
}

// New returns a new Writer
func New(out io.Writer) *Writer {
	return &Writer{out: out}
}

// Flush the buffer, writing all buffered lines to out
func (w *Writer) Flush() error {
	if w.buf.Len() == 0 {
		return nil
	}
	w.clearLines(w.lineCount)
	w.lineCount = bytes.Count(w.buf.Bytes(), []byte{'\n'})
	_, err := w.out.Write(w.buf.Bytes())
	w.buf.Reset()
	return err
}

// Write saves buf to a buffer
func (w *Writer) Write(buf []byte) (int, error) {
	return w.buf.Write(buf)
}
