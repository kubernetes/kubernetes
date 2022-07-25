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

package trace // import "go.opentelemetry.io/otel/sdk/trace"

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel"
)

// simpleSpanProcessor is a SpanProcessor that synchronously sends all
// completed Spans to a trace.Exporter immediately.
type simpleSpanProcessor struct {
	exporterMu sync.RWMutex
	exporter   SpanExporter
	stopOnce   sync.Once
}

var _ SpanProcessor = (*simpleSpanProcessor)(nil)

// NewSimpleSpanProcessor returns a new SpanProcessor that will synchronously
// send completed spans to the exporter immediately.
func NewSimpleSpanProcessor(exporter SpanExporter) SpanProcessor {
	ssp := &simpleSpanProcessor{
		exporter: exporter,
	}
	return ssp
}

// OnStart does nothing.
func (ssp *simpleSpanProcessor) OnStart(context.Context, ReadWriteSpan) {}

// OnEnd immediately exports a ReadOnlySpan.
func (ssp *simpleSpanProcessor) OnEnd(s ReadOnlySpan) {
	ssp.exporterMu.RLock()
	defer ssp.exporterMu.RUnlock()

	if ssp.exporter != nil && s.SpanContext().TraceFlags().IsSampled() {
		ss := s.Snapshot()
		if err := ssp.exporter.ExportSpans(context.Background(), []*SpanSnapshot{ss}); err != nil {
			otel.Handle(err)
		}
	}
}

// Shutdown shuts down the exporter this SimpleSpanProcessor exports to.
func (ssp *simpleSpanProcessor) Shutdown(ctx context.Context) error {
	var err error
	ssp.stopOnce.Do(func() {
		ssp.exporterMu.Lock()
		exporter := ssp.exporter
		// Set exporter to nil so subsequent calls to OnEnd are ignored
		// gracefully.
		ssp.exporter = nil
		ssp.exporterMu.Unlock()

		// Clear the ssp.exporter prior to shutting it down so if that creates
		// a span that needs to be exported there is no deadlock.
		err = exporter.Shutdown(ctx)
	})
	return err
}

// ForceFlush does nothing as there is no data to flush.
func (ssp *simpleSpanProcessor) ForceFlush(context.Context) error {
	return nil
}
