package writer

import (
	"bytes"
	"io"
	"sync"
)

type WriterInterface interface {
	io.Writer

	Truncate()
	DumpOut()
	DumpOutWithHeader(header string)
}

type Writer struct {
	buffer    *bytes.Buffer
	outWriter io.Writer
	lock      *sync.Mutex
	stream    bool
}

func New(outWriter io.Writer) *Writer {
	return &Writer{
		buffer:    &bytes.Buffer{},
		lock:      &sync.Mutex{},
		outWriter: outWriter,
		stream:    true,
	}
}

func (w *Writer) SetStream(stream bool) {
	w.lock.Lock()
	defer w.lock.Unlock()
	w.stream = stream
}

func (w *Writer) Write(b []byte) (n int, err error) {
	w.lock.Lock()
	defer w.lock.Unlock()

	if w.stream {
		return w.outWriter.Write(b)
	} else {
		return w.buffer.Write(b)
	}
}

func (w *Writer) Truncate() {
	w.lock.Lock()
	defer w.lock.Unlock()
	w.buffer.Reset()
}

func (w *Writer) DumpOut() {
	w.lock.Lock()
	defer w.lock.Unlock()
	if !w.stream {
		w.buffer.WriteTo(w.outWriter)
	}
}

func (w *Writer) DumpOutWithHeader(header string) {
	w.lock.Lock()
	defer w.lock.Unlock()
	if !w.stream && w.buffer.Len() > 0 {
		w.outWriter.Write([]byte(header))
		w.buffer.WriteTo(w.outWriter)
	}
}
