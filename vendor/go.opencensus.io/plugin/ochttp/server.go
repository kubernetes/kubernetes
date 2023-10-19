// Copyright 2018, OpenCensus Authors
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

package ochttp

import (
	"context"
	"io"
	"net/http"
	"strconv"
	"sync"
	"time"

	"go.opencensus.io/stats"
	"go.opencensus.io/tag"
	"go.opencensus.io/trace"
	"go.opencensus.io/trace/propagation"
)

// Handler is an http.Handler wrapper to instrument your HTTP server with
// OpenCensus. It supports both stats and tracing.
//
// # Tracing
//
// This handler is aware of the incoming request's span, reading it from request
// headers as configured using the Propagation field.
// The extracted span can be accessed from the incoming request's
// context.
//
//	span := trace.FromContext(r.Context())
//
// The server span will be automatically ended at the end of ServeHTTP.
type Handler struct {
	// Propagation defines how traces are propagated. If unspecified,
	// B3 propagation will be used.
	Propagation propagation.HTTPFormat

	// Handler is the handler used to handle the incoming request.
	Handler http.Handler

	// StartOptions are applied to the span started by this Handler around each
	// request.
	//
	// StartOptions.SpanKind will always be set to trace.SpanKindServer
	// for spans started by this transport.
	StartOptions trace.StartOptions

	// GetStartOptions allows to set start options per request. If set,
	// StartOptions is going to be ignored.
	GetStartOptions func(*http.Request) trace.StartOptions

	// IsPublicEndpoint should be set to true for publicly accessible HTTP(S)
	// servers. If true, any trace metadata set on the incoming request will
	// be added as a linked trace instead of being added as a parent of the
	// current trace.
	IsPublicEndpoint bool

	// FormatSpanName holds the function to use for generating the span name
	// from the information found in the incoming HTTP Request. By default the
	// name equals the URL Path.
	FormatSpanName func(*http.Request) string

	// IsHealthEndpoint holds the function to use for determining if the
	// incoming HTTP request should be considered a health check. This is in
	// addition to the private isHealthEndpoint func which may also indicate
	// tracing should be skipped.
	IsHealthEndpoint func(*http.Request) bool
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	var tags addedTags
	r, traceEnd := h.startTrace(w, r)
	defer traceEnd()
	w, statsEnd := h.startStats(w, r)
	defer statsEnd(&tags)
	handler := h.Handler
	if handler == nil {
		handler = http.DefaultServeMux
	}
	r = r.WithContext(context.WithValue(r.Context(), addedTagsKey{}, &tags))
	handler.ServeHTTP(w, r)
}

func (h *Handler) startTrace(w http.ResponseWriter, r *http.Request) (*http.Request, func()) {
	if h.IsHealthEndpoint != nil && h.IsHealthEndpoint(r) || isHealthEndpoint(r.URL.Path) {
		return r, func() {}
	}
	var name string
	if h.FormatSpanName == nil {
		name = spanNameFromURL(r)
	} else {
		name = h.FormatSpanName(r)
	}
	ctx := r.Context()

	startOpts := h.StartOptions
	if h.GetStartOptions != nil {
		startOpts = h.GetStartOptions(r)
	}

	var span *trace.Span
	sc, ok := h.extractSpanContext(r)
	if ok && !h.IsPublicEndpoint {
		ctx, span = trace.StartSpanWithRemoteParent(ctx, name, sc,
			trace.WithSampler(startOpts.Sampler),
			trace.WithSpanKind(trace.SpanKindServer))
	} else {
		ctx, span = trace.StartSpan(ctx, name,
			trace.WithSampler(startOpts.Sampler),
			trace.WithSpanKind(trace.SpanKindServer),
		)
		if ok {
			span.AddLink(trace.Link{
				TraceID:    sc.TraceID,
				SpanID:     sc.SpanID,
				Type:       trace.LinkTypeParent,
				Attributes: nil,
			})
		}
	}
	span.AddAttributes(requestAttrs(r)...)
	if r.Body == nil {
		// TODO: Handle cases where ContentLength is not set.
	} else if r.ContentLength > 0 {
		span.AddMessageReceiveEvent(0, /* TODO: messageID */
			r.ContentLength, -1)
	}
	return r.WithContext(ctx), span.End
}

func (h *Handler) extractSpanContext(r *http.Request) (trace.SpanContext, bool) {
	if h.Propagation == nil {
		return defaultFormat.SpanContextFromRequest(r)
	}
	return h.Propagation.SpanContextFromRequest(r)
}

func (h *Handler) startStats(w http.ResponseWriter, r *http.Request) (http.ResponseWriter, func(tags *addedTags)) {
	ctx, _ := tag.New(r.Context(),
		tag.Upsert(Host, r.Host),
		tag.Upsert(Path, r.URL.Path),
		tag.Upsert(Method, r.Method))
	track := &trackingResponseWriter{
		start:  time.Now(),
		ctx:    ctx,
		writer: w,
	}
	if r.Body == nil {
		// TODO: Handle cases where ContentLength is not set.
		track.reqSize = -1
	} else if r.ContentLength > 0 {
		track.reqSize = r.ContentLength
	}
	stats.Record(ctx, ServerRequestCount.M(1))
	return track.wrappedResponseWriter(), track.end
}

type trackingResponseWriter struct {
	ctx        context.Context
	reqSize    int64
	respSize   int64
	start      time.Time
	statusCode int
	statusLine string
	endOnce    sync.Once
	writer     http.ResponseWriter
}

// Compile time assertion for ResponseWriter interface
var _ http.ResponseWriter = (*trackingResponseWriter)(nil)

func (t *trackingResponseWriter) end(tags *addedTags) {
	t.endOnce.Do(func() {
		if t.statusCode == 0 {
			t.statusCode = 200
		}

		span := trace.FromContext(t.ctx)
		span.SetStatus(TraceStatus(t.statusCode, t.statusLine))
		span.AddAttributes(trace.Int64Attribute(StatusCodeAttribute, int64(t.statusCode)))

		m := []stats.Measurement{
			ServerLatency.M(float64(time.Since(t.start)) / float64(time.Millisecond)),
			ServerResponseBytes.M(t.respSize),
		}
		if t.reqSize >= 0 {
			m = append(m, ServerRequestBytes.M(t.reqSize))
		}
		allTags := make([]tag.Mutator, len(tags.t)+1)
		allTags[0] = tag.Upsert(StatusCode, strconv.Itoa(t.statusCode))
		copy(allTags[1:], tags.t)
		stats.RecordWithTags(t.ctx, allTags, m...)
	})
}

func (t *trackingResponseWriter) Header() http.Header {
	return t.writer.Header()
}

func (t *trackingResponseWriter) Write(data []byte) (int, error) {
	n, err := t.writer.Write(data)
	t.respSize += int64(n)
	// Add message event for request bytes sent.
	span := trace.FromContext(t.ctx)
	span.AddMessageSendEvent(0 /* TODO: messageID */, int64(n), -1)
	return n, err
}

func (t *trackingResponseWriter) WriteHeader(statusCode int) {
	t.writer.WriteHeader(statusCode)
	t.statusCode = statusCode
	t.statusLine = http.StatusText(t.statusCode)
}

// wrappedResponseWriter returns a wrapped version of the original
//
//	ResponseWriter and only implements the same combination of additional
//
// interfaces as the original.
// This implementation is based on https://github.com/felixge/httpsnoop.
func (t *trackingResponseWriter) wrappedResponseWriter() http.ResponseWriter {
	var (
		hj, i0 = t.writer.(http.Hijacker)
		cn, i1 = t.writer.(http.CloseNotifier)
		pu, i2 = t.writer.(http.Pusher)
		fl, i3 = t.writer.(http.Flusher)
		rf, i4 = t.writer.(io.ReaderFrom)
	)

	switch {
	case !i0 && !i1 && !i2 && !i3 && !i4:
		return struct {
			http.ResponseWriter
		}{t}
	case !i0 && !i1 && !i2 && !i3 && i4:
		return struct {
			http.ResponseWriter
			io.ReaderFrom
		}{t, rf}
	case !i0 && !i1 && !i2 && i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Flusher
		}{t, fl}
	case !i0 && !i1 && !i2 && i3 && i4:
		return struct {
			http.ResponseWriter
			http.Flusher
			io.ReaderFrom
		}{t, fl, rf}
	case !i0 && !i1 && i2 && !i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Pusher
		}{t, pu}
	case !i0 && !i1 && i2 && !i3 && i4:
		return struct {
			http.ResponseWriter
			http.Pusher
			io.ReaderFrom
		}{t, pu, rf}
	case !i0 && !i1 && i2 && i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Pusher
			http.Flusher
		}{t, pu, fl}
	case !i0 && !i1 && i2 && i3 && i4:
		return struct {
			http.ResponseWriter
			http.Pusher
			http.Flusher
			io.ReaderFrom
		}{t, pu, fl, rf}
	case !i0 && i1 && !i2 && !i3 && !i4:
		return struct {
			http.ResponseWriter
			http.CloseNotifier
		}{t, cn}
	case !i0 && i1 && !i2 && !i3 && i4:
		return struct {
			http.ResponseWriter
			http.CloseNotifier
			io.ReaderFrom
		}{t, cn, rf}
	case !i0 && i1 && !i2 && i3 && !i4:
		return struct {
			http.ResponseWriter
			http.CloseNotifier
			http.Flusher
		}{t, cn, fl}
	case !i0 && i1 && !i2 && i3 && i4:
		return struct {
			http.ResponseWriter
			http.CloseNotifier
			http.Flusher
			io.ReaderFrom
		}{t, cn, fl, rf}
	case !i0 && i1 && i2 && !i3 && !i4:
		return struct {
			http.ResponseWriter
			http.CloseNotifier
			http.Pusher
		}{t, cn, pu}
	case !i0 && i1 && i2 && !i3 && i4:
		return struct {
			http.ResponseWriter
			http.CloseNotifier
			http.Pusher
			io.ReaderFrom
		}{t, cn, pu, rf}
	case !i0 && i1 && i2 && i3 && !i4:
		return struct {
			http.ResponseWriter
			http.CloseNotifier
			http.Pusher
			http.Flusher
		}{t, cn, pu, fl}
	case !i0 && i1 && i2 && i3 && i4:
		return struct {
			http.ResponseWriter
			http.CloseNotifier
			http.Pusher
			http.Flusher
			io.ReaderFrom
		}{t, cn, pu, fl, rf}
	case i0 && !i1 && !i2 && !i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
		}{t, hj}
	case i0 && !i1 && !i2 && !i3 && i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			io.ReaderFrom
		}{t, hj, rf}
	case i0 && !i1 && !i2 && i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.Flusher
		}{t, hj, fl}
	case i0 && !i1 && !i2 && i3 && i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.Flusher
			io.ReaderFrom
		}{t, hj, fl, rf}
	case i0 && !i1 && i2 && !i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.Pusher
		}{t, hj, pu}
	case i0 && !i1 && i2 && !i3 && i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.Pusher
			io.ReaderFrom
		}{t, hj, pu, rf}
	case i0 && !i1 && i2 && i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.Pusher
			http.Flusher
		}{t, hj, pu, fl}
	case i0 && !i1 && i2 && i3 && i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.Pusher
			http.Flusher
			io.ReaderFrom
		}{t, hj, pu, fl, rf}
	case i0 && i1 && !i2 && !i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.CloseNotifier
		}{t, hj, cn}
	case i0 && i1 && !i2 && !i3 && i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.CloseNotifier
			io.ReaderFrom
		}{t, hj, cn, rf}
	case i0 && i1 && !i2 && i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.CloseNotifier
			http.Flusher
		}{t, hj, cn, fl}
	case i0 && i1 && !i2 && i3 && i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.CloseNotifier
			http.Flusher
			io.ReaderFrom
		}{t, hj, cn, fl, rf}
	case i0 && i1 && i2 && !i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.CloseNotifier
			http.Pusher
		}{t, hj, cn, pu}
	case i0 && i1 && i2 && !i3 && i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.CloseNotifier
			http.Pusher
			io.ReaderFrom
		}{t, hj, cn, pu, rf}
	case i0 && i1 && i2 && i3 && !i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.CloseNotifier
			http.Pusher
			http.Flusher
		}{t, hj, cn, pu, fl}
	case i0 && i1 && i2 && i3 && i4:
		return struct {
			http.ResponseWriter
			http.Hijacker
			http.CloseNotifier
			http.Pusher
			http.Flusher
			io.ReaderFrom
		}{t, hj, cn, pu, fl, rf}
	default:
		return struct {
			http.ResponseWriter
		}{t}
	}
}
