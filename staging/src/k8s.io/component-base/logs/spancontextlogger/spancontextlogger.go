/*
Copyright 2024 The Kubernetes Authors.

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

// Package spancontextlogger includes utility functions for correlating logs
// with traces.
package spancontextlogger

import (
	"context"

	"go.opentelemetry.io/otel/trace"
	"k8s.io/klog/v2"
)

// LoggerWithSpanContext adds trace_id and span_id values to the logger if the
// context includes a span context.
func LoggerWithSpanContext(ctx context.Context, logger klog.Logger) klog.Logger {
	if spanCtx := trace.SpanContextFromContext(ctx); spanCtx.IsValid() {
		return klog.LoggerWithValues(logger, "trace_id", spanCtx.TraceID(), "span_id", spanCtx.SpanID())
	}
	return logger
}
