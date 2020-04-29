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

package trace

import (
	"context"

	"go.opentelemetry.io/otel/api/core"
)

type traceContextKeyType int

const (
	currentSpanKey traceContextKeyType = iota
	remoteContextKey
)

// ContextWithSpan creates a new context with a current span set to
// the passed span.
func ContextWithSpan(ctx context.Context, span Span) context.Context {
	return context.WithValue(ctx, currentSpanKey, span)
}

// SpanFromContext returns the current span stored in the context.
func SpanFromContext(ctx context.Context) Span {
	if span, has := ctx.Value(currentSpanKey).(Span); has {
		return span
	}
	return NoopSpan{}
}

// ContextWithRemoteSpanContext creates a new context with a remote
// span context set to the passed span context.
func ContextWithRemoteSpanContext(ctx context.Context, sc core.SpanContext) context.Context {
	return context.WithValue(ctx, remoteContextKey, sc)
}

// RemoteSpanContextFromContext returns the remote span context stored
// in the context.
func RemoteSpanContextFromContext(ctx context.Context) core.SpanContext {
	if sc, ok := ctx.Value(remoteContextKey).(core.SpanContext); ok {
		return sc
	}
	return core.EmptySpanContext()
}
