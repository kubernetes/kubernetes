// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package otelhttp

import (
	"context"
	"io"
	"net/http"

	"go.opentelemetry.io/otel/propagation"
)

var _ io.ReadCloser = &bodyWrapper{}

// bodyWrapper wraps a http.Request.Body (an io.ReadCloser) to track the number
// of bytes read and the last error
type bodyWrapper struct {
	io.ReadCloser
	record func(n int64) // must not be nil

	read int64
	err  error
}

func (w *bodyWrapper) Read(b []byte) (int, error) {
	n, err := w.ReadCloser.Read(b)
	n1 := int64(n)
	w.read += n1
	w.err = err
	w.record(n1)
	return n, err
}

func (w *bodyWrapper) Close() error {
	return w.ReadCloser.Close()
}

var _ http.ResponseWriter = &respWriterWrapper{}

// respWriterWrapper wraps a http.ResponseWriter in order to track the number of
// bytes written, the last error, and to catch the returned statusCode
// TODO: The wrapped http.ResponseWriter doesn't implement any of the optional
// types (http.Hijacker, http.Pusher, http.CloseNotifier, http.Flusher, etc)
// that may be useful when using it in real life situations.
type respWriterWrapper struct {
	http.ResponseWriter
	record func(n int64) // must not be nil

	// used to inject the header
	ctx context.Context

	props propagation.TextMapPropagator

	written     int64
	statusCode  int
	err         error
	wroteHeader bool
}

func (w *respWriterWrapper) Header() http.Header {
	return w.ResponseWriter.Header()
}

func (w *respWriterWrapper) Write(p []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	n, err := w.ResponseWriter.Write(p)
	n1 := int64(n)
	w.record(n1)
	w.written += n1
	w.err = err
	return n, err
}

func (w *respWriterWrapper) WriteHeader(statusCode int) {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	w.statusCode = statusCode
	w.props.Inject(w.ctx, propagation.HeaderCarrier(w.Header()))
	w.ResponseWriter.WriteHeader(statusCode)
}
