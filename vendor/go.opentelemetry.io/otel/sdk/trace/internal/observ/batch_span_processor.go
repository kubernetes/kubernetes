// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package observ // import "go.opentelemetry.io/otel/sdk/trace/internal/observ"

import (
	"context"
	"errors"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/sdk"
	"go.opentelemetry.io/otel/sdk/internal/x"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
	"go.opentelemetry.io/otel/semconv/v1.37.0/otelconv"
)

const (
	// ScopeName is the name of the instrumentation scope.
	ScopeName = "go.opentelemetry.io/otel/sdk/trace/internal/observ"

	// SchemaURL is the schema URL of the instrumentation.
	SchemaURL = semconv.SchemaURL
)

// ErrQueueFull is the attribute value for the "queue_full" error type.
var ErrQueueFull = otelconv.SDKProcessorSpanProcessed{}.AttrErrorType(
	otelconv.ErrorTypeAttr("queue_full"),
)

// BSPComponentName returns the component name attribute for a
// BatchSpanProcessor with the given ID.
func BSPComponentName(id int64) attribute.KeyValue {
	t := otelconv.ComponentTypeBatchingSpanProcessor
	name := fmt.Sprintf("%s/%d", t, id)
	return semconv.OTelComponentName(name)
}

// BSP is the instrumentation for an OTel SDK BatchSpanProcessor.
type BSP struct {
	reg metric.Registration

	processed              metric.Int64Counter
	processedOpts          []metric.AddOption
	processedQueueFullOpts []metric.AddOption
}

func NewBSP(id int64, qLen func() int64, qMax int64) (*BSP, error) {
	if !x.Observability.Enabled() {
		return nil, nil
	}

	meter := otel.GetMeterProvider().Meter(
		ScopeName,
		metric.WithInstrumentationVersion(sdk.Version()),
		metric.WithSchemaURL(SchemaURL),
	)

	qCap, err := otelconv.NewSDKProcessorSpanQueueCapacity(meter)
	if err != nil {
		err = fmt.Errorf("failed to create BSP queue capacity metric: %w", err)
	}
	qCapInst := qCap.Inst()

	qSize, e := otelconv.NewSDKProcessorSpanQueueSize(meter)
	if e != nil {
		e := fmt.Errorf("failed to create BSP queue size metric: %w", e)
		err = errors.Join(err, e)
	}
	qSizeInst := qSize.Inst()

	cmpntT := semconv.OTelComponentTypeBatchingSpanProcessor
	cmpnt := BSPComponentName(id)
	set := attribute.NewSet(cmpnt, cmpntT)

	obsOpts := []metric.ObserveOption{metric.WithAttributeSet(set)}
	reg, e := meter.RegisterCallback(
		func(_ context.Context, o metric.Observer) error {
			o.ObserveInt64(qSizeInst, qLen(), obsOpts...)
			o.ObserveInt64(qCapInst, qMax, obsOpts...)
			return nil
		},
		qSizeInst,
		qCapInst,
	)
	if e != nil {
		e := fmt.Errorf("failed to register BSP queue size/capacity callback: %w", e)
		err = errors.Join(err, e)
	}

	processed, e := otelconv.NewSDKProcessorSpanProcessed(meter)
	if e != nil {
		e := fmt.Errorf("failed to create BSP processed spans metric: %w", e)
		err = errors.Join(err, e)
	}
	processedOpts := []metric.AddOption{metric.WithAttributeSet(set)}

	set = attribute.NewSet(cmpnt, cmpntT, ErrQueueFull)
	processedQueueFullOpts := []metric.AddOption{metric.WithAttributeSet(set)}

	return &BSP{
		reg:                    reg,
		processed:              processed.Inst(),
		processedOpts:          processedOpts,
		processedQueueFullOpts: processedQueueFullOpts,
	}, err
}

func (b *BSP) Shutdown() error { return b.reg.Unregister() }

func (b *BSP) Processed(ctx context.Context, n int64) {
	b.processed.Add(ctx, n, b.processedOpts...)
}

func (b *BSP) ProcessedQueueFull(ctx context.Context, n int64) {
	b.processed.Add(ctx, n, b.processedQueueFullOpts...)
}
