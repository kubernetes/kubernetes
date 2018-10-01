// Copyright 2017, OpenCensus Authors
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

package stackdriver

import (
	"fmt"
	"log"
	"sync"
	"time"

	tracingclient "cloud.google.com/go/trace/apiv2"
	"go.opencensus.io/trace"
	"google.golang.org/api/support/bundler"
	tracepb "google.golang.org/genproto/googleapis/devtools/cloudtrace/v2"
)

// traceExporter is an implementation of trace.Exporter that uploads spans to
// Stackdriver.
//
type traceExporter struct {
	o         Options
	projectID string
	bundler   *bundler.Bundler
	// uploadFn defaults to uploadSpans; it can be replaced for tests.
	uploadFn func(spans []*trace.SpanData)
	overflowLogger
	client *tracingclient.Client
}

var _ trace.Exporter = (*traceExporter)(nil)

func newTraceExporter(o Options) (*traceExporter, error) {
	client, err := tracingclient.NewClient(o.Context, o.TraceClientOptions...)
	if err != nil {
		return nil, fmt.Errorf("stackdriver: couldn't initialize trace client: %v", err)
	}
	return newTraceExporterWithClient(o, client), nil
}

func newTraceExporterWithClient(o Options, c *tracingclient.Client) *traceExporter {
	e := &traceExporter{
		projectID: o.ProjectID,
		client:    c,
		o:         o,
	}
	bundler := bundler.NewBundler((*trace.SpanData)(nil), func(bundle interface{}) {
		e.uploadFn(bundle.([]*trace.SpanData))
	})
	if o.BundleDelayThreshold > 0 {
		bundler.DelayThreshold = o.BundleDelayThreshold
	} else {
		bundler.DelayThreshold = 2 * time.Second
	}
	if o.BundleCountThreshold > 0 {
		bundler.BundleCountThreshold = o.BundleCountThreshold
	} else {
		bundler.BundleCountThreshold = 50
	}
	// The measured "bytes" are not really bytes, see exportReceiver.
	bundler.BundleByteThreshold = bundler.BundleCountThreshold * 200
	bundler.BundleByteLimit = bundler.BundleCountThreshold * 1000
	bundler.BufferedByteLimit = bundler.BundleCountThreshold * 2000

	e.bundler = bundler
	e.uploadFn = e.uploadSpans
	return e
}

// ExportSpan exports a SpanData to Stackdriver Trace.
func (e *traceExporter) ExportSpan(s *trace.SpanData) {
	// n is a length heuristic.
	n := 1
	n += len(s.Attributes)
	n += len(s.Annotations)
	n += len(s.MessageEvents)
	err := e.bundler.Add(s, n)
	switch err {
	case nil:
		return
	case bundler.ErrOversizedItem:
		go e.uploadFn([]*trace.SpanData{s})
	case bundler.ErrOverflow:
		e.overflowLogger.log()
	default:
		e.o.handleError(err)
	}
}

// Flush waits for exported trace spans to be uploaded.
//
// This is useful if your program is ending and you do not want to lose recent
// spans.
func (e *traceExporter) Flush() {
	e.bundler.Flush()
}

// uploadSpans uploads a set of spans to Stackdriver.
func (e *traceExporter) uploadSpans(spans []*trace.SpanData) {
	req := tracepb.BatchWriteSpansRequest{
		Name:  "projects/" + e.projectID,
		Spans: make([]*tracepb.Span, 0, len(spans)),
	}
	for _, span := range spans {
		req.Spans = append(req.Spans, protoFromSpanData(span, e.projectID, e.o.Resource))
	}
	// Create a never-sampled span to prevent traces associated with exporter.
	ctx, span := trace.StartSpan( // TODO: add timeouts
		e.o.Context,
		"contrib.go.opencensus.io/exporter/stackdriver.uploadSpans",
		trace.WithSampler(trace.NeverSample()),
	)
	defer span.End()
	span.AddAttributes(trace.Int64Attribute("num_spans", int64(len(spans))))

	err := e.client.BatchWriteSpans(ctx, &req)
	if err != nil {
		span.SetStatus(trace.Status{Code: 2, Message: err.Error()})
		e.o.handleError(err)
	}
}

// overflowLogger ensures that at most one overflow error log message is
// written every 5 seconds.
type overflowLogger struct {
	mu    sync.Mutex
	pause bool
	accum int
}

func (o *overflowLogger) delay() {
	o.pause = true
	time.AfterFunc(5*time.Second, func() {
		o.mu.Lock()
		defer o.mu.Unlock()
		switch {
		case o.accum == 0:
			o.pause = false
		case o.accum == 1:
			log.Println("OpenCensus Stackdriver exporter: failed to upload span: buffer full")
			o.accum = 0
			o.delay()
		default:
			log.Printf("OpenCensus Stackdriver exporter: failed to upload %d spans: buffer full", o.accum)
			o.accum = 0
			o.delay()
		}
	})
}

func (o *overflowLogger) log() {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.pause {
		log.Println("OpenCensus Stackdriver exporter: failed to upload span: buffer full")
		o.delay()
	} else {
		o.accum++
	}
}
