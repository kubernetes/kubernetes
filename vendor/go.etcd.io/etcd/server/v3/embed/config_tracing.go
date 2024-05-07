// Copyright 2022 The etcd Authors
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

package embed

import (
	"context"

	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	tracesdk "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.4.0"
	"go.uber.org/zap"
)

func setupTracingExporter(ctx context.Context, cfg *Config) (exporter tracesdk.SpanExporter, options []otelgrpc.Option, err error) {
	exporter, err = otlptracegrpc.New(ctx,
		otlptracegrpc.WithInsecure(),
		otlptracegrpc.WithEndpoint(cfg.ExperimentalDistributedTracingAddress),
	)
	if err != nil {
		return nil, nil, err
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceNameKey.String(cfg.ExperimentalDistributedTracingServiceName),
		),
	)
	if err != nil {
		return nil, nil, err
	}

	if resWithIDKey := determineResourceWithIDKey(cfg.ExperimentalDistributedTracingServiceInstanceID); resWithIDKey != nil {
		// Merge resources into a new
		// resource in case of duplicates.
		res, err = resource.Merge(res, resWithIDKey)
		if err != nil {
			return nil, nil, err
		}
	}

	options = append(options,
		otelgrpc.WithPropagators(
			propagation.NewCompositeTextMapPropagator(
				propagation.TraceContext{},
				propagation.Baggage{},
			),
		),
		otelgrpc.WithTracerProvider(
			tracesdk.NewTracerProvider(
				tracesdk.WithBatcher(exporter),
				tracesdk.WithResource(res),
				tracesdk.WithSampler(tracesdk.ParentBased(tracesdk.NeverSample())),
			),
		),
	)

	cfg.logger.Debug(
		"distributed tracing enabled",
		zap.String("address", cfg.ExperimentalDistributedTracingAddress),
		zap.String("service-name", cfg.ExperimentalDistributedTracingServiceName),
		zap.String("service-instance-id", cfg.ExperimentalDistributedTracingServiceInstanceID),
	)

	return exporter, options, err
}

// As Tracing service Instance ID must be unique, it should
// never use the empty default string value, it's set if
// if it's a non empty string.
func determineResourceWithIDKey(serviceInstanceID string) *resource.Resource {
	if serviceInstanceID != "" {
		return resource.NewSchemaless(
			(semconv.ServiceInstanceIDKey.String(serviceInstanceID)),
		)
	}
	return nil
}
