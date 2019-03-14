package recordio

import (
	"io"
	"strconv"
)

var lf = []byte{'\n'}

type Writer struct {
	out io.Writer
}

func NewWriter(out io.Writer) *Writer {
	return &Writer{out}
}

func (w *Writer) writeBuffer(b []byte, err error) error {
	if err != nil {
		return err
	}
	n, err := w.out.Write(b)
	if err == nil && n != len(b) {
		return io.ErrShortWrite
	}
	return err
}

func (w *Writer) WriteFrame(b []byte) (err error) {
	err = w.writeBuffer(([]byte)(strconv.Itoa(len(b))), err)
	err = w.writeBuffer(lf, err)
	err = w.writeBuffer(b, err)
	return
}
