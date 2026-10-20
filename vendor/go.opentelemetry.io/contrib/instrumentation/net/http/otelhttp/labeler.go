// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelhttp

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel/attribute"
)

// Labeler is used to allow instrumented HTTP handlers to add custom attributes to
// the metrics recorded by the net/http instrumentation.
type Labeler struct {
	mu         sync.Mutex
	attributes []attribute.KeyValue
}

// Add attributes to a Labeler.
func (l *Labeler) Add(ls ...attribute.KeyValue) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.attributes = append(l.attributes, ls...)
}

// Get returns a copy of the attributes added to the Labeler.
func (l *Labeler) Get() []attribute.KeyValue {
	l.mu.Lock()
	defer l.mu.Unlock()
	ret := make([]attribute.KeyValue, len(l.attributes))
	copy(ret, l.attributes)
	return ret
}

type labelerContextKeyType int

const labelerContextKey labelerContextKeyType = 0

// ContextWithLabeler returns a new context with the provided Labeler instance.
// Attributes added to the specified labeler will be injected into metrics
// emitted by the instrumentation. Only one labeller can be injected into the
// context. Injecting it multiple times will override the previous calls.
func ContextWithLabeler(parent context.Context, l *Labeler) context.Context {
	return context.WithValue(parent, labelerContextKey, l)
}

// LabelerFromContext retrieves a Labeler instance from the provided context if
// one is available.  If no Labeler was found in the provided context a new, empty
// Labeler is returned and the second return value is false.  In this case it is
// safe to use the Labeler but any attributes added to it will not be used.
func LabelerFromContext(ctx context.Context) (*Labeler, bool) {
	l, ok := ctx.Value(labelerContextKey).(*Labeler)
	if !ok {
		l = &Labeler{}
	}
	return l, ok
}

// clientLabelerContextKey is a separate context key for the client-side Labeler.
// This prevents server-side Labeler attributes (e.g., http.route) from leaking
// into client-side metrics when a server handler propagates its request context
// into an outbound HTTP client request.
const clientLabelerContextKey labelerContextKeyType = 1

// ContextWithClientLabeler returns a new context with the provided Labeler instance
// for use with client-side HTTP instrumentation (otelhttp.Transport).
// Attributes added to this labeler will be attached to client-side metrics.
func ContextWithClientLabeler(parent context.Context, l *Labeler) context.Context {
	return context.WithValue(parent, clientLabelerContextKey, l)
}

// ClientLabelerFromContext retrieves a Labeler instance from the provided context
// for use with client-side HTTP instrumentation. If no Labeler was found, a new,
// empty Labeler is returned and the second return value is false.
func ClientLabelerFromContext(ctx context.Context) (*Labeler, bool) {
	l, ok := ctx.Value(clientLabelerContextKey).(*Labeler)
	if !ok {
		l = &Labeler{}
	}
	return l, ok
}
