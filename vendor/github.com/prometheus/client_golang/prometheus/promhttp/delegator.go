// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package promhttp

import (
	"bufio"
	"io"
	"net"
	"net/http"
)

const (
	closeNotifier = 1 << iota
	flusher
	hijacker
	readerFrom
	pusher
)

type delegator interface {
	http.ResponseWriter

	Status() int
	Written() int64
}

type responseWriterDelegator struct {
	http.ResponseWriter

	handler, method    string
	status             int
	written            int64
	wroteHeader        bool
	observeWriteHeader func(int)
}

func (r *responseWriterDelegator) Status() int {
	return r.status
}

func (r *responseWriterDelegator) Written() int64 {
	return r.written
}

func (r *responseWriterDelegator) WriteHeader(code int) {
	r.status = code
	r.wroteHeader = true
	r.ResponseWriter.WriteHeader(code)
	if r.observeWriteHeader != nil {
		r.observeWriteHeader(code)
	}
}

func (r *responseWriterDelegator) Write(b []byte) (int, error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.Write(b)
	r.written += int64(n)
	return n, err
}

type closeNotifierDelegator struct{ *responseWriterDelegator }
type flusherDelegator struct{ *responseWriterDelegator }
type hijackerDelegator struct{ *responseWriterDelegator }
type readerFromDelegator struct{ *responseWriterDelegator }

func (d closeNotifierDelegator) CloseNotify() <-chan bool {
	return d.ResponseWriter.(http.CloseNotifier).CloseNotify()
}
func (d flusherDelegator) Flush() {
	d.ResponseWriter.(http.Flusher).Flush()
}
func (d hijackerDelegator) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return d.ResponseWriter.(http.Hijacker).Hijack()
}
func (d readerFromDelegator) ReadFrom(re io.Reader) (int64, error) {
	if !d.wroteHeader {
		d.WriteHeader(http.StatusOK)
	}
	n, err := d.ResponseWriter.(io.ReaderFrom).ReadFrom(re)
	d.written += n
	return n, err
}

var pickDelegator = make([]func(*responseWriterDelegator) delegator, 32)

func init() {
	// TODO(beorn7): Code generation would help here.
	pickDelegator[0] = func(d *responseWriterDelegator) delegator { // 0
		return d
	}
	pickDelegator[closeNotifier] = func(d *responseWriterDelegator) delegator { // 1
		return closeNotifierDelegator{d}
	}
	pickDelegator[flusher] = func(d *responseWriterDelegator) delegator { // 2
		return flusherDelegator{d}
	}
	pickDelegator[flusher+closeNotifier] = func(d *responseWriterDelegator) delegator { // 3
		return struct {
			*responseWriterDelegator
			http.Flusher
			http.CloseNotifier
		}{d, flusherDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[hijacker] = func(d *responseWriterDelegator) delegator { // 4
		return hijackerDelegator{d}
	}
	pickDelegator[hijacker+closeNotifier] = func(d *responseWriterDelegator) delegator { // 5
		return struct {
			*responseWriterDelegator
			http.Hijacker
			http.CloseNotifier
		}{d, hijackerDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[hijacker+flusher] = func(d *responseWriterDelegator) delegator { // 6
		return struct {
			*responseWriterDelegator
			http.Hijacker
			http.Flusher
		}{d, hijackerDelegator{d}, flusherDelegator{d}}
	}
	pickDelegator[hijacker+flusher+closeNotifier] = func(d *responseWriterDelegator) delegator { // 7
		return struct {
			*responseWriterDelegator
			http.Hijacker
			http.Flusher
			http.CloseNotifier
		}{d, hijackerDelegator{d}, flusherDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[readerFrom] = func(d *responseWriterDelegator) delegator { // 8
		return readerFromDelegator{d}
	}
	pickDelegator[readerFrom+closeNotifier] = func(d *responseWriterDelegator) delegator { // 9
		return struct {
			*responseWriterDelegator
			io.ReaderFrom
			http.CloseNotifier
		}{d, readerFromDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[readerFrom+flusher] = func(d *responseWriterDelegator) delegator { // 10
		return struct {
			*responseWriterDelegator
			io.ReaderFrom
			http.Flusher
		}{d, readerFromDelegator{d}, flusherDelegator{d}}
	}
	pickDelegator[readerFrom+flusher+closeNotifier] = func(d *responseWriterDelegator) delegator { // 11
		return struct {
			*responseWriterDelegator
			io.ReaderFrom
			http.Flusher
			http.CloseNotifier
		}{d, readerFromDelegator{d}, flusherDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[readerFrom+hijacker] = func(d *responseWriterDelegator) delegator { // 12
		return struct {
			*responseWriterDelegator
			io.ReaderFrom
			http.Hijacker
		}{d, readerFromDelegator{d}, hijackerDelegator{d}}
	}
	pickDelegator[readerFrom+hijacker+closeNotifier] = func(d *responseWriterDelegator) delegator { // 13
		return struct {
			*responseWriterDelegator
			io.ReaderFrom
			http.Hijacker
			http.CloseNotifier
		}{d, readerFromDelegator{d}, hijackerDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[readerFrom+hijacker+flusher] = func(d *responseWriterDelegator) delegator { // 14
		return struct {
			*responseWriterDelegator
			io.ReaderFrom
			http.Hijacker
			http.Flusher
		}{d, readerFromDelegator{d}, hijackerDelegator{d}, flusherDelegator{d}}
	}
	pickDelegator[readerFrom+hijacker+flusher+closeNotifier] = func(d *responseWriterDelegator) delegator { // 15
		return struct {
			*responseWriterDelegator
			io.ReaderFrom
			http.Hijacker
			http.Flusher
			http.CloseNotifier
		}{d, readerFromDelegator{d}, hijackerDelegator{d}, flusherDelegator{d}, closeNotifierDelegator{d}}
	}
}
