package internal

import (
	"bytes"
	"fmt"
	"io"
	"sync"
)

type WriterMode uint

const (
	WriterModeStreamAndBuffer WriterMode = iota
	WriterModeBufferOnly
)

type WriterInterface interface {
	io.Writer

	Truncate()
	Bytes() []byte
}

//Writer impplements WriterInterface and GinkgoWriterInterface
type Writer struct {
	buffer    *bytes.Buffer
	outWriter io.Writer
	lock      *sync.Mutex
	mode      WriterMode

	teeWriters []io.Writer
}

func NewWriter(outWriter io.Writer) *Writer {
	return &Writer{
		buffer:    &bytes.Buffer{},
		lock:      &sync.Mutex{},
		outWriter: outWriter,
		mode:      WriterModeStreamAndBuffer,
	}
}

func (w *Writer) SetMode(mode WriterMode) {
	w.lock.Lock()
	defer w.lock.Unlock()
	w.mode = mode
}

func (w *Writer) Write(b []byte) (n int, err error) {
	w.lock.Lock()
	defer w.lock.Unlock()

	for _, teeWriter := range w.teeWriters {
		teeWriter.Write(b)
	}

	if w.mode == WriterModeStreamAndBuffer {
		w.outWriter.Write(b)
	}
	return w.buffer.Write(b)
}

func (w *Writer) Truncate() {
	w.lock.Lock()
	defer w.lock.Unlock()
	w.buffer.Reset()
}

func (w *Writer) Bytes() []byte {
	w.lock.Lock()
	defer w.lock.Unlock()
	b := w.buffer.Bytes()
	copied := make([]byte, len(b))
	copy(copied, b)
	return copied
}

//GinkgoWriterInterface
func (w *Writer) TeeTo(writer io.Writer) {
	w.lock.Lock()
	defer w.lock.Unlock()

	w.teeWriters = append(w.teeWriters, writer)
}

func (w *Writer) ClearTeeWriters() {
	w.lock.Lock()
	defer w.lock.Unlock()

	w.teeWriters = []io.Writer{}
}

func (w *Writer) Print(a ...interface{}) {
	fmt.Fprint(w, a...)
}

func (w *Writer) Printf(format string, a ...interface{}) {
	fmt.Fprintf(w, format, a...)
}

func (w *Writer) Println(a ...interface{}) {
	fmt.Fprintln(w, a...)
}
