/*
Copyright 2021 The Kubernetes Authors.

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

package responsewriter

import (
	"bufio"
	"net"
	"net/http"
)

var _ http.ResponseWriter = &FakeResponseWriter{}

// FakeResponseWriter implements http.ResponseWriter,
// it is used for testing purpose only
type FakeResponseWriter struct{}

func (fw *FakeResponseWriter) Header() http.Header          { return http.Header{} }
func (fw *FakeResponseWriter) WriteHeader(code int)         {}
func (fw *FakeResponseWriter) Write(bs []byte) (int, error) { return len(bs), nil }

// For HTTP2 an http.ResponseWriter object implements
// http.Flusher and http.CloseNotifier.
// It is used for testing purpose only
type FakeResponseWriterFlusherCloseNotifier struct {
	*FakeResponseWriter
}

func (fw *FakeResponseWriterFlusherCloseNotifier) Flush()                   {}
func (fw *FakeResponseWriterFlusherCloseNotifier) CloseNotify() <-chan bool { return nil }

// For HTTP/1.x an http.ResponseWriter object implements
// http.Flusher, http.CloseNotifier and http.Hijacker.
// It is used for testing purpose only
type FakeResponseWriterFlusherCloseNotifierHijacker struct {
	*FakeResponseWriterFlusherCloseNotifier
}

func (fw *FakeResponseWriterFlusherCloseNotifierHijacker) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return nil, nil, nil
}
