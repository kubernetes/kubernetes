// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package observ // import "go.opentelemetry.io/otel/sdk/trace/internal/observ"

import (
	"context"
	"fmt"
	"sync"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/sdk"
	"go.opentelemetry.io/otel/sdk/internal/x"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
	"go.opentelemetry.io/otel/semconv/v1.37.0/otelconv"
)

var measureAttrsPool = sync.Pool{
	New: func() any {
		// "component.name" + "component.type" + "error.type"
		const n = 1 + 1 + 1
		s := make([]attribute.KeyValue, 0, n)
		// Return a pointer to a slice instead of a slice itself
		// to avoid allocations on every call.
		return &s
	},
}

// SSP is the instrumentation for an OTel SDK SimpleSpanProcessor.
type SSP struct {
	spansProcessedCounter metric.Int64Counter
	addOpts               []metric.AddOption
	attrs                 []attribute.KeyValue
}

// SSPComponentName returns the component name attribute for a
// SimpleSpanProcessor with the given ID.
func SSPComponentName(id int64) attribute.KeyValue {
	t := otelconv.ComponentTypeSimpleSpanProcessor
	name := fmt.Sprintf("%s/%d", t, id)
	return semconv.OTelComponentName(name)
}

// NewSSP returns instrumentation for an OTel SDK SimpleSpanProcessor with the
// provided ID.
//
// If the experimental observability is disabled, nil is returned.
func NewSSP(id int64) (*SSP, error) {
	if !x.Observability.Enabled() {
		return nil, nil
	}

	meter := otel.GetMeterProvider().Meter(
		ScopeName,
		metric.WithInstrumentationVersion(sdk.Version()),
		metric.WithSchemaURL(SchemaURL),
	)
	spansProcessedCounter, err := otelconv.NewSDKProcessorSpanProcessed(meter)
	if err != nil {
		err = fmt.Errorf("failed to create SSP processed spans metric: %w", err)
	}

	componentName := SSPComponentName(id)
	componentType := spansProcessedCounter.AttrComponentType(otelconv.ComponentTypeSimpleSpanProcessor)
	attrs := []attribute.KeyValue{componentName, componentType}
	addOpts := []metric.AddOption{metric.WithAttributeSet(attribute.NewSet(attrs...))}

	return &SSP{
		spansProcessedCounter: spansProcessedCounter.Inst(),
		addOpts:               addOpts,
		attrs:                 attrs,
	}, err
}

// SpanProcessed records that a span has been processed by the SimpleSpanProcessor.
// If err is non-nil, it records the processing error as an attribute.
func (ssp *SSP) SpanProcessed(ctx context.Context, err error) {
	ssp.spansProcessedCounter.Add(ctx, 1, ssp.addOption(err)...)
}

func (ssp *SSP) addOption(err error) []metric.AddOption {
	if err == nil {
		return ssp.addOpts
	}
	attrs := measureAttrsPool.Get().(*[]attribute.KeyValue)
	defer func() {
		*attrs = (*attrs)[:0] // reset the slice for reuse
		measureAttrsPool.Put(attrs)
	}()
	*attrs = append(*attrs, ssp.attrs...)
	*attrs = append(*attrs, semconv.ErrorType(err))
	// Do not inefficiently make a copy of attrs by using
	// WithAttributes instead of WithAttributeSet.
	return []metric.AddOption{metric.WithAttributeSet(attribute.NewSet(*attrs...))}
}
