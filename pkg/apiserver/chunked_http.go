/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"bufio"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
)

// hijackHTTPChunked hijacks an http.ResponseWriter as a flushable, chunked writer and returns
// the connection so that write deadlines can be set. Clients MUST close the writer AND the
// connection.
func hijackHTTPChunked(w http.ResponseWriter) (io.WriteCloser, net.Conn, http.Flusher, error) {
	hijacker, ok := w.(http.Hijacker)
	if !ok {
		return nil, nil, nil, errors.New("does not implement http.Hijacker")
	}
	conn, rw, err := hijacker.Hijack()
	if err != nil {
		return nil, nil, nil, err
	}
	return httputil.NewChunkedWriter(rw.Writer), conn, noReturnFlusher{rw.Writer}, nil
}

// noReturnFlusher implements http.Flusher around a bufio.Writer
type noReturnFlusher struct {
	w *bufio.Writer
}

// Flush implements http.Flusher
func (f noReturnFlusher) Flush() {
	f.w.Flush()
}
