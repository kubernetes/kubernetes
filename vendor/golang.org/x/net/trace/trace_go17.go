// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7

package trace

import "context"

// NewContext returns a copy of the parent context
// and associates it with a Trace.
func NewContext(ctx context.Context, tr Trace) context.Context {
	return context.WithValue(ctx, contextKey, tr)
}

// FromContext returns the Trace bound to the context, if any.
func FromContext(ctx context.Context) (tr Trace, ok bool) {
	tr, ok = ctx.Value(contextKey).(Trace)
	return
}
