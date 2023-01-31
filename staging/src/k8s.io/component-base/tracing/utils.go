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

package tracing

import (
	"context"
	"net/http"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	oteltrace "go.opentelemetry.io/otel/trace"

	"k8s.io/client-go/transport"
	"k8s.io/component-base/tracing/api/v1"
)

// TracerProvider is an OpenTelemetry TracerProvider which can be shut down
type TracerProvider interface {
	oteltrace.TracerProvider
	Shutdown(context.Context) error
}

type noopTracerProvider struct {
	oteltrace.TracerProvider
}

func (n *noopTracerProvider) Shutdown(context.Context) error {
	return nil
}

func NewNoopTracerProvider() TracerProvider {
	return &noopTracerProvider{TracerProvider: oteltrace.NewNoopTracerProvider()}
}

// NewProvider creates a TracerProvider in a component, and enforces recommended tracing behavior
func NewProvider(ctx context.Context,
	tracingConfig *v1.TracingConfiguration,
	addedOpts []otlptracegrpc.Option,
	resourceOpts []resource.Option,
) (TracerProvider, error) {
	if tracingConfig == nil {
		return NewNoopTracerProvider(), nil
	}
	opts := append([]otlptracegrpc.Option{}, addedOpts...)
	if tracingConfig.Endpoint != nil {
		opts = append(opts, otlptracegrpc.WithEndpoint(*tracingConfig.Endpoint))
	}
	opts = append(opts, otlptracegrpc.WithInsecure())
	exporter, err := otlptracegrpc.New(ctx, opts...)
	if err != nil {
		return nil, err
	}
	res, err := resource.New(ctx, resourceOpts...)
	if err != nil {
		return nil, err
	}

	// sampler respects parent span's sampling rate or
	// otherwise never samples.
	sampler := sdktrace.NeverSample()
	// Or, emit spans for a fraction of transactions
	if tracingConfig.SamplingRatePerMillion != nil && *tracingConfig.SamplingRatePerMillion > 0 {
		sampler = sdktrace.TraceIDRatioBased(float64(*tracingConfig.SamplingRatePerMillion) / float64(1000000))
	}
	// batch span processor to aggregate spans before export.
	bsp := sdktrace.NewBatchSpanProcessor(exporter)
	tp := sdktrace.NewTracerProvider(
		sdktrace.WithSampler(sdktrace.ParentBased(sampler)),
		sdktrace.WithSpanProcessor(bsp),
		sdktrace.WithResource(res),
	)
	return tp, nil
}

// WithTracing adds tracing to requests if the incoming request is sampled
func WithTracing(handler http.Handler, tp oteltrace.TracerProvider, serviceName string) http.Handler {
	opts := []otelhttp.Option{
		otelhttp.WithPropagators(Propagators()),
		otelhttp.WithTracerProvider(tp),
	}
	// With Noop TracerProvider, the otelhttp still handles context propagation.
	// See https://github.com/open-telemetry/opentelemetry-go/tree/main/example/passthrough
	return otelhttp.NewHandler(handler, serviceName, opts...)
}

// WrapperFor can be used to add tracing to a *rest.Config.
// Example usage:
// tp := NewProvider(...)
// config, _ := rest.InClusterConfig()
// config.Wrap(WrapperFor(tp))
// kubeclient, _ := clientset.NewForConfig(config)
func WrapperFor(tp oteltrace.TracerProvider) transport.WrapperFunc {
	return func(rt http.RoundTripper) http.RoundTripper {
		opts := []otelhttp.Option{
			otelhttp.WithPropagators(Propagators()),
			otelhttp.WithTracerProvider(tp),
		}
		// With Noop TracerProvider, the otelhttp still handles context propagation.
		// See https://github.com/open-telemetry/opentelemetry-go/tree/main/example/passthrough
		return otelhttp.NewTransport(rt, opts...)
	}
}

// Propagators returns the recommended set of propagators.
func Propagators() propagation.TextMapPropagator {
	return propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{})
}
