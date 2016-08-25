/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package limitwriter

import (
	"errors"
	"io"
)

// New creates a writer that is limited to writing at most n bytes to w. This writer is not
// thread safe.
func New(w io.Writer, n int64) io.Writer {
	return &limitWriter{
		w: w,
		n: n,
	}
}

// ErrMaximumWrite is returned when all bytes have been written.
var ErrMaximumWrite = errors.New("maximum write")

type limitWriter struct {
	w io.Writer
	n int64
}

func (w *limitWriter) Write(p []byte) (n int, err error) {
	if int64(len(p)) > w.n {
		p = p[:w.n]
	}
	if w.n > 0 {
		n, err = w.w.Write(p)
		w.n -= int64(n)
	}
	if w.n == 0 {
		err = ErrMaximumWrite
	}
	return
}
