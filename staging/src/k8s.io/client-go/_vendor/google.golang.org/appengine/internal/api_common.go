// Copyright 2015 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package internal

import (
	"github.com/golang/protobuf/proto"
	netcontext "golang.org/x/net/context"
)

type CallOverrideFunc func(ctx netcontext.Context, service, method string, in, out proto.Message) error

var callOverrideKey = "holds []CallOverrideFunc"

func WithCallOverride(ctx netcontext.Context, f CallOverrideFunc) netcontext.Context {
	// We avoid appending to any existing call override
	// so we don't risk overwriting a popped stack below.
	var cofs []CallOverrideFunc
	if uf, ok := ctx.Value(&callOverrideKey).([]CallOverrideFunc); ok {
		cofs = append(cofs, uf...)
	}
	cofs = append(cofs, f)
	return netcontext.WithValue(ctx, &callOverrideKey, cofs)
}

func callOverrideFromContext(ctx netcontext.Context) (CallOverrideFunc, netcontext.Context, bool) {
	cofs, _ := ctx.Value(&callOverrideKey).([]CallOverrideFunc)
	if len(cofs) == 0 {
		return nil, nil, false
	}
	// We found a list of overrides; grab the last, and reconstitute a
	// context that will hide it.
	f := cofs[len(cofs)-1]
	ctx = netcontext.WithValue(ctx, &callOverrideKey, cofs[:len(cofs)-1])
	return f, ctx, true
}

type logOverrideFunc func(level int64, format string, args ...interface{})

var logOverrideKey = "holds a logOverrideFunc"

func WithLogOverride(ctx netcontext.Context, f logOverrideFunc) netcontext.Context {
	return netcontext.WithValue(ctx, &logOverrideKey, f)
}

var appIDOverrideKey = "holds a string, being the full app ID"

func WithAppIDOverride(ctx netcontext.Context, appID string) netcontext.Context {
	return netcontext.WithValue(ctx, &appIDOverrideKey, appID)
}

var namespaceKey = "holds the namespace string"

func withNamespace(ctx netcontext.Context, ns string) netcontext.Context {
	return netcontext.WithValue(ctx, &namespaceKey, ns)
}

func NamespaceFromContext(ctx netcontext.Context) string {
	// If there's no namespace, return the empty string.
	ns, _ := ctx.Value(&namespaceKey).(string)
	return ns
}

// FullyQualifiedAppID returns the fully-qualified application ID.
// This may contain a partition prefix (e.g. "s~" for High Replication apps),
// or a domain prefix (e.g. "example.com:").
func FullyQualifiedAppID(ctx netcontext.Context) string {
	if id, ok := ctx.Value(&appIDOverrideKey).(string); ok {
		return id
	}
	return fullyQualifiedAppID(ctx)
}

func Logf(ctx netcontext.Context, level int64, format string, args ...interface{}) {
	if f, ok := ctx.Value(&logOverrideKey).(logOverrideFunc); ok {
		f(level, format, args...)
		return
	}
	logf(fromContext(ctx), level, format, args...)
}

// NamespacedContext wraps a Context to support namespaces.
func NamespacedContext(ctx netcontext.Context, namespace string) netcontext.Context {
	return withNamespace(ctx, namespace)
}
