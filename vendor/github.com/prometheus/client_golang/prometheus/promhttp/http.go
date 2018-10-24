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
	"time"

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

// Handler returns an http.Handler for the prometheus.DefaultGatherer, using
// default HandlerOpts, i.e. it reports the first error as an HTTP error, it has
// no error logging, and it applies compression if requested by the client.
//
// The returned http.Handler is already instrumented using the
// InstrumentMetricHandler function and the prometheus.DefaultRegisterer. If you
// create multiple http.Handlers by separate calls of the Handler function, the
// metrics used for instrumentation will be shared between them, providing
// global scrape counts.
//
// This function is meant to cover the bulk of basic use cases. If you are doing
// anything that requires more customization (including using a non-default
// Gatherer, different instrumentation, and non-default HandlerOpts), use the
// HandlerFor function. See there for details.
func Handler() http.Handler {
	return InstrumentMetricHandler(
		prometheus.DefaultRegisterer, HandlerFor(prometheus.DefaultGatherer, HandlerOpts{}),
	)
}

// HandlerFor returns an uninstrumented http.Handler for the provided
// Gatherer. The behavior of the Handler is defined by the provided
// HandlerOpts. Thus, HandlerFor is useful to create http.Handlers for custom
// Gatherers, with non-default HandlerOpts, and/or with custom (or no)
// instrumentation. Use the InstrumentMetricHandler function to apply the same
// kind of instrumentation as it is used by the Handler function.
func HandlerFor(reg prometheus.Gatherer, opts HandlerOpts) http.Handler {
	var inFlightSem chan struct{}
	if opts.MaxRequestsInFlight > 0 {
		inFlightSem = make(chan struct{}, opts.MaxRequestsInFlight)
	}

	h := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if inFlightSem != nil {
			select {
			case inFlightSem <- struct{}{}: // All good, carry on.
				defer func() { <-inFlightSem }()
			default:
				http.Error(w, fmt.Sprintf(
					"Limit of concurrent requests reached (%d), try again later.", opts.MaxRequestsInFlight,
				), http.StatusServiceUnavailable)
				return
			}
		}

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
			http.Error(w, "No metrics encoded, last error:\n\n"+lastErr.Error(), http.StatusInternalServerError)
			return
		}
		header := w.Header()
		header.Set(contentTypeHeader, string(contentType))
		header.Set(contentLengthHeader, fmt.Sprint(buf.Len()))
		if encoding != "" {
			header.Set(contentEncodingHeader, encoding)
		}
		if _, err := w.Write(buf.Bytes()); err != nil && opts.ErrorLog != nil {
			opts.ErrorLog.Println("error while sending encoded metrics:", err)
		}
		// TODO(beorn7): Consider streaming serving of metrics.
	})

	if opts.Timeout <= 0 {
		return h
	}
	return http.TimeoutHandler(h, opts.Timeout, fmt.Sprintf(
		"Exceeded configured timeout of %v.\n",
		opts.Timeout,
	))
}

// InstrumentMetricHandler is usually used with an http.Handler returned by the
// HandlerFor function. It instruments the provided http.Handler with two
// metrics: A counter vector "promhttp_metric_handler_requests_total" to count
// scrapes partitioned by HTTP status code, and a gauge
// "promhttp_metric_handler_requests_in_flight" to track the number of
// simultaneous scrapes. This function idempotently registers collectors for
// both metrics with the provided Registerer. It panics if the registration
// fails. The provided metrics are useful to see how many scrapes hit the
// monitored target (which could be from different Prometheus servers or other
// scrapers), and how often they overlap (which would result in more than one
// scrape in flight at the same time). Note that the scrapes-in-flight gauge
// will contain the scrape by which it is exposed, while the scrape counter will
// only get incremented after the scrape is complete (as only then the status
// code is known). For tracking scrape durations, use the
// "scrape_duration_seconds" gauge created by the Prometheus server upon each
// scrape.
func InstrumentMetricHandler(reg prometheus.Registerer, handler http.Handler) http.Handler {
	cnt := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "promhttp_metric_handler_requests_total",
			Help: "Total number of scrapes by HTTP status code.",
		},
		[]string{"code"},
	)
	// Initialize the most likely HTTP status codes.
	cnt.WithLabelValues("200")
	cnt.WithLabelValues("500")
	cnt.WithLabelValues("503")
	if err := reg.Register(cnt); err != nil {
		if are, ok := err.(prometheus.AlreadyRegisteredError); ok {
			cnt = are.ExistingCollector.(*prometheus.CounterVec)
		} else {
			panic(err)
		}
	}

	gge := prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "promhttp_metric_handler_requests_in_flight",
		Help: "Current number of scrapes being served.",
	})
	if err := reg.Register(gge); err != nil {
		if are, ok := err.(prometheus.AlreadyRegisteredError); ok {
			gge = are.ExistingCollector.(prometheus.Gauge)
		} else {
			panic(err)
		}
	}

	return InstrumentHandlerCounter(cnt, InstrumentHandlerInFlight(gge, handler))
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
	// The number of concurrent HTTP requests is limited to
	// MaxRequestsInFlight. Additional requests are responded to with 503
	// Service Unavailable and a suitable message in the body. If
	// MaxRequestsInFlight is 0 or negative, no limit is applied.
	MaxRequestsInFlight int
	// If handling a request takes longer than Timeout, it is responded to
	// with 503 ServiceUnavailable and a suitable Message. No timeout is
	// applied if Timeout is 0 or negative. Note that with the current
	// implementation, reaching the timeout simply ends the HTTP requests as
	// described above (and even that only if sending of the body hasn't
	// started yet), while the bulk work of gathering all the metrics keeps
	// running in the background (with the eventual result to be thrown
	// away). Until the implementation is improved, it is recommended to
	// implement a separate timeout in potentially slow Collectors.
	Timeout time.Duration
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
		part = strings.TrimSpace(part)
		if part == "gzip" || strings.HasPrefix(part, "gzip;") {
			return gzip.NewWriter(writer), "gzip"
		}
	}
	return writer, ""
}
