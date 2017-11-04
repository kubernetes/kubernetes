// Copyright 2016 The Prometheus Authors
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

// Package promhttp provides tooling around HTTP servers and clients.
//
// First, the package allows the creation of http.Handler instances to expose
// Prometheus metrics via HTTP. promhttp.Handler acts on the
// prometheus.DefaultGatherer. With HandlerFor, you can create a handler for a
// custom registry or anything that implements the Gatherer interface. It also
// allows the creation of handlers that act differently on errors or allow to
// log errors.
//
// Second, the package provides tooling to instrument instances of http.Handler
// via middleware. Middleware wrappers follow the naming scheme
// InstrumentHandlerX, where X describes the intended use of the middleware.
// See each function's doc comment for specific details.
//
// Finally, the package allows for an http.RoundTripper to be instrumented via
// middleware. Middleware wrappers follow the naming scheme
// InstrumentRoundTripperX, where X describes the intended use of the
// middleware. See each function's doc comment for specific details.
package promhttp

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"

	"github.com/prometheus/common/expfmt"

	"github.com/prometheus/client_golang/prometheus"
)

const (
	contentTypeHeader     = "Content-Type"
	contentLengthHeader   = "Content-Length"
	contentEncodingHeader = "Content-Encoding"
	acceptEncodingHeader  = "Accept-Encoding"
)

var bufPool sync.Pool

func getBuf() *bytes.Buffer {
	buf := bufPool.Get()
	if buf == nil {
		return &bytes.Buffer{}
	}
	return buf.(*bytes.Buffer)
}

func giveBuf(buf *bytes.Buffer) {
	buf.Reset()
	bufPool.Put(buf)
}

// Handler returns an HTTP handler for the prometheus.DefaultGatherer. The
// Handler uses the default HandlerOpts, i.e. report the first error as an HTTP
// error, no error logging, and compression if requested by the client.
//
// If you want to create a Handler for the DefaultGatherer with different
// HandlerOpts, create it with HandlerFor with prometheus.DefaultGatherer and
// your desired HandlerOpts.
func Handler() http.Handler {
	return HandlerFor(prometheus.DefaultGatherer, HandlerOpts{})
}

// HandlerFor returns an http.Handler for the provided Gatherer. The behavior
// of the Handler is defined by the provided HandlerOpts.
func HandlerFor(reg prometheus.Gatherer, opts HandlerOpts) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		mfs, err := reg.Gather()
		if err != nil {
			if opts.ErrorLog != nil {
				opts.ErrorLog.Println("error gathering metrics:", err)
			}
			switch opts.ErrorHandling {
			case PanicOnError:
				panic(err)
			case ContinueOnError:
				if len(mfs) == 0 {
					http.Error(w, "No metrics gathered, last error:\n\n"+err.Error(), http.StatusInternalServerError)
					return
				}
			case HTTPErrorOnError:
				http.Error(w, "An error has occurred during metrics gathering:\n\n"+err.Error(), http.StatusInternalServerError)
				return
			}
		}

		contentType := expfmt.Negotiate(req.Header)
		buf := getBuf()
		defer giveBuf(buf)
		writer, encoding := decorateWriter(req, buf, opts.DisableCompression)
		enc := expfmt.NewEncoder(writer, contentType)
		var lastErr error
		for _, mf := range mfs {
			if err := enc.Encode(mf); err != nil {
				lastErr = err
				if opts.ErrorLog != nil {
					opts.ErrorLog.Println("error encoding metric family:", err)
				}
				switch opts.ErrorHandling {
				case PanicOnError:
					panic(err)
				case ContinueOnError:
					// Handled later.
				case HTTPErrorOnError:
					http.Error(w, "An error has occurred during metrics encoding:\n\n"+err.Error(), http.StatusInternalServerError)
					return
				}
			}
		}
		if closer, ok := writer.(io.Closer); ok {
			closer.Close()
		}
		if lastErr != nil && buf.Len() == 0 {
			http.Error(w, "No metrics encoded, last error:\n\n"+err.Error(), http.StatusInternalServerError)
			return
		}
		header := w.Header()
		header.Set(contentTypeHeader, string(contentType))
		header.Set(contentLengthHeader, fmt.Sprint(buf.Len()))
		if encoding != "" {
			header.Set(contentEncodingHeader, encoding)
		}
		w.Write(buf.Bytes())
		// TODO(beorn7): Consider streaming serving of metrics.
	})
}

// HandlerErrorHandling defines how a Handler serving metrics will handle
// errors.
type HandlerErrorHandling int

// These constants cause handlers serving metrics to behave as described if
// errors are encountered.
const (
	// Serve an HTTP status code 500 upon the first error
	// encountered. Report the error message in the body.
	HTTPErrorOnError HandlerErrorHandling = iota
	// Ignore errors and try to serve as many metrics as possible.  However,
	// if no metrics can be served, serve an HTTP status code 500 and the
	// last error message in the body. Only use this in deliberate "best
	// effort" metrics collection scenarios. It is recommended to at least
	// log errors (by providing an ErrorLog in HandlerOpts) to not mask
	// errors completely.
	ContinueOnError
	// Panic upon the first error encountered (useful for "crash only" apps).
	PanicOnError
)

// Logger is the minimal interface HandlerOpts needs for logging. Note that
// log.Logger from the standard library implements this interface, and it is
// easy to implement by custom loggers, if they don't do so already anyway.
type Logger interface {
	Println(v ...interface{})
}

// HandlerOpts specifies options how to serve metrics via an http.Handler. The
// zero value of HandlerOpts is a reasonable default.
type HandlerOpts struct {
	// ErrorLog specifies an optional logger for errors collecting and
	// serving metrics. If nil, errors are not logged at all.
	ErrorLog Logger
	// ErrorHandling defines how errors are handled. Note that errors are
	// logged regardless of the configured ErrorHandling provided ErrorLog
	// is not nil.
	ErrorHandling HandlerErrorHandling
	// If DisableCompression is true, the handler will never compress the
	// response, even if requested by the client.
	DisableCompression bool
}

// decorateWriter wraps a writer to handle gzip compression if requested.  It
// returns the decorated writer and the appropriate "Content-Encoding" header
// (which is empty if no compression is enabled).
func decorateWriter(request *http.Request, writer io.Writer, compressionDisabled bool) (io.Writer, string) {
	if compressionDisabled {
		return writer, ""
	}
	header := request.Header.Get(acceptEncodingHeader)
	parts := strings.Split(header, ",")
	for _, part := range parts {
		part := strings.TrimSpace(part)
		if part == "gzip" || strings.HasPrefix(part, "gzip;") {
			return gzip.NewWriter(writer), "gzip"
		}
	}
	return writer, ""
}
