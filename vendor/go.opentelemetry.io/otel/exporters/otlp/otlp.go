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

package otlp // import "go.opentelemetry.io/otel/exporters/otlp"

import (
	"context"
	"errors"
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/metric"
	metricsdk "go.opentelemetry.io/otel/sdk/export/metric"
	"go.opentelemetry.io/otel/sdk/export/metric/aggregation"
	"go.opentelemetry.io/otel/sdk/metric/selector/simple"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"go.opentelemetry.io/otel/sdk/metric/controller/basic"
	processor "go.opentelemetry.io/otel/sdk/metric/processor/basic"
	tracesdk "go.opentelemetry.io/otel/sdk/trace"
)

// Exporter is an OpenTelemetry exporter. It exports both traces and metrics
// from OpenTelemetry instrumented to code using OpenTelemetry protocol
// buffers to a configurable receiver.
type Exporter struct {
	cfg    config
	driver ProtocolDriver

	mu      sync.RWMutex
	started bool

	startOnce sync.Once
	stopOnce  sync.Once
}

var _ tracesdk.SpanExporter = (*Exporter)(nil)
var _ metricsdk.Exporter = (*Exporter)(nil)

// NewExporter constructs a new Exporter and starts it.
func NewExporter(ctx context.Context, driver ProtocolDriver, opts ...ExporterOption) (*Exporter, error) {
	exp := NewUnstartedExporter(driver, opts...)
	if err := exp.Start(ctx); err != nil {
		return nil, err
	}
	return exp, nil
}

// NewUnstartedExporter constructs a new Exporter and does not start it.
func NewUnstartedExporter(driver ProtocolDriver, opts ...ExporterOption) *Exporter {
	cfg := config{
		// Note: the default ExportKindSelector is specified
		// as Cumulative:
		// https://github.com/open-telemetry/opentelemetry-specification/issues/731
		exportKindSelector: metricsdk.CumulativeExportKindSelector(),
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return &Exporter{
		cfg:    cfg,
		driver: driver,
	}
}

var (
	errAlreadyStarted = errors.New("already started")
)

// Start establishes connections to the OpenTelemetry collector. Starting an
// already started exporter returns an error.
func (e *Exporter) Start(ctx context.Context) error {
	var err = errAlreadyStarted
	e.startOnce.Do(func() {
		e.mu.Lock()
		e.started = true
		e.mu.Unlock()
		err = e.driver.Start(ctx)
	})

	return err
}

// Shutdown closes all connections and releases resources currently being used
// by the exporter. If the exporter is not started this does nothing. A shut
// down exporter can't be started again. Shutting down an already shut down
// exporter does nothing.
func (e *Exporter) Shutdown(ctx context.Context) error {
	e.mu.RLock()
	started := e.started
	e.mu.RUnlock()

	if !started {
		return nil
	}

	var err error

	e.stopOnce.Do(func() {
		err = e.driver.Stop(ctx)
		e.mu.Lock()
		e.started = false
		e.mu.Unlock()
	})

	return err
}

// Export transforms and batches metric Records into OTLP Metrics and
// transmits them to the configured collector.
func (e *Exporter) Export(parent context.Context, cps metricsdk.CheckpointSet) error {
	return e.driver.ExportMetrics(parent, cps, e.cfg.exportKindSelector)
}

// ExportKindFor reports back to the OpenTelemetry SDK sending this Exporter
// metric telemetry that it needs to be provided in a configured format.
func (e *Exporter) ExportKindFor(desc *metric.Descriptor, kind aggregation.Kind) metricsdk.ExportKind {
	return e.cfg.exportKindSelector.ExportKindFor(desc, kind)
}

// ExportSpans transforms and batches trace SpanSnapshots into OTLP Trace and
// transmits them to the configured collector.
func (e *Exporter) ExportSpans(ctx context.Context, ss []*tracesdk.SpanSnapshot) error {
	return e.driver.ExportTraces(ctx, ss)
}

// NewExportPipeline sets up a complete export pipeline
// with the recommended TracerProvider setup.
func NewExportPipeline(ctx context.Context, driver ProtocolDriver, exporterOpts ...ExporterOption) (*Exporter,
	*sdktrace.TracerProvider, *basic.Controller, error) {

	exp, err := NewExporter(ctx, driver, exporterOpts...)
	if err != nil {
		return nil, nil, nil, err
	}

	tracerProvider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
	)

	cntr := basic.New(
		processor.New(
			simple.NewWithInexpensiveDistribution(),
			exp,
		),
	)

	return exp, tracerProvider, cntr, nil
}

// InstallNewPipeline instantiates a NewExportPipeline with the
// recommended configuration and registers it globally.
func InstallNewPipeline(ctx context.Context, driver ProtocolDriver, exporterOpts ...ExporterOption) (*Exporter,
	*sdktrace.TracerProvider, *basic.Controller, error) {

	exp, tp, cntr, err := NewExportPipeline(ctx, driver, exporterOpts...)
	if err != nil {
		return nil, nil, nil, err
	}

	otel.SetTracerProvider(tp)
	err = cntr.Start(ctx)
	if err != nil {
		return nil, nil, nil, err
	}

	return exp, tp, cntr, err
}
