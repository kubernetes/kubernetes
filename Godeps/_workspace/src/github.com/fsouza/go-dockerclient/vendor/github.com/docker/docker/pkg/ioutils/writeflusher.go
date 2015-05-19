package ioutils

import (
	"io"
	"net/http"
	"sync"
)

type WriteFlusher struct {
	sync.Mutex
	w       io.Writer
	flusher http.Flusher
	flushed bool
}

func (wf *WriteFlusher) Write(b []byte) (n int, err error) {
	wf.Lock()
	defer wf.Unlock()
	n, err = wf.w.Write(b)
	wf.flushed = true
	wf.flusher.Flush()
	return n, err
}

// Flush the stream immediately.
func (wf *WriteFlusher) Flush() {
	wf.Lock()
	defer wf.Unlock()
	wf.flushed = true
	wf.flusher.Flush()
}

func (wf *WriteFlusher) Flushed() bool {
	wf.Lock()
	defer wf.Unlock()
	return wf.flushed
}

func NewWriteFlusher(w io.Writer) *WriteFlusher {
	var flusher http.Flusher
	if f, ok := w.(http.Flusher); ok {
		flusher = f
	} else {
		flusher = &NopFlusher{}
	}
	return &WriteFlusher{w: w, flusher: flusher}
}
