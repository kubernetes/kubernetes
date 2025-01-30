// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otlptracegrpc // import "go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"

import (
	"context"

	"go.opentelemetry.io/otel/exporters/otlp/otlptrace"
)

// New constructs a new Exporter and starts it.
func New(ctx context.Context, opts ...Option) (*otlptrace.Exporter, error) {
	return otlptrace.New(ctx, NewClient(opts...))
}

// NewUnstarted constructs a new Exporter and does not start it.
func NewUnstarted(opts ...Option) *otlptrace.Exporter {
	return otlptrace.NewUnstarted(NewClient(opts...))
}
