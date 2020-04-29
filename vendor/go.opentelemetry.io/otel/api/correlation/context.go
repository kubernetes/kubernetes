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

package correlation

import (
	"context"

	"go.opentelemetry.io/otel/api/core"
)

type correlationsType struct{}

// SetHookFunc describes a type of a callback that is called when
// storing baggage in the context.
type SetHookFunc func(context.Context) context.Context

// GetHookFunc describes a type of a callback that is called when
// getting baggage from the context.
type GetHookFunc func(context.Context, Map) Map

// value under this key is either of type Map or correlationsData
var correlationsKey = &correlationsType{}

type correlationsData struct {
	m       Map
	setHook SetHookFunc
	getHook GetHookFunc
}

func (d correlationsData) isHookless() bool {
	return d.setHook == nil && d.getHook == nil
}

type hookKind int

const (
	hookKindSet hookKind = iota
	hookKindGet
)

func (d *correlationsData) overrideHook(kind hookKind, setHook SetHookFunc, getHook GetHookFunc) {
	switch kind {
	case hookKindSet:
		d.setHook = setHook
	case hookKindGet:
		d.getHook = getHook
	}
}

// ContextWithSetHook installs a hook function that will be invoked
// every time ContextWithMap is called. To avoid unnecessary callback
// invocations (recursive or not), the callback can temporarily clear
// the hooks from the context with the ContextWithNoHooks function.
//
// Note that NewContext also calls ContextWithMap, so the hook will be
// invoked.
//
// Passing nil SetHookFunc creates a context with no set hook to call.
//
// This function should not be used by applications or libraries. It
// is mostly for interoperation with other observability APIs.
func ContextWithSetHook(ctx context.Context, hook SetHookFunc) context.Context {
	return contextWithHook(ctx, hookKindSet, hook, nil)
}

// ContextWithGetHook installs a hook function that will be invoked
// every time MapFromContext is called. To avoid unnecessary callback
// invocations (recursive or not), the callback can temporarily clear
// the hooks from the context with the ContextWithNoHooks function.
//
// Note that NewContext also calls MapFromContext, so the hook will be
// invoked.
//
// Passing nil GetHookFunc creates a context with no get hook to call.
//
// This function should not be used by applications or libraries. It
// is mostly for interoperation with other observability APIs.
func ContextWithGetHook(ctx context.Context, hook GetHookFunc) context.Context {
	return contextWithHook(ctx, hookKindGet, nil, hook)
}

func contextWithHook(ctx context.Context, kind hookKind, setHook SetHookFunc, getHook GetHookFunc) context.Context {
	switch v := ctx.Value(correlationsKey).(type) {
	case correlationsData:
		v.overrideHook(kind, setHook, getHook)
		if v.isHookless() {
			return context.WithValue(ctx, correlationsKey, v.m)
		}
		return context.WithValue(ctx, correlationsKey, v)
	case Map:
		return contextWithOneHookAndMap(ctx, kind, setHook, getHook, v)
	default:
		m := NewEmptyMap()
		return contextWithOneHookAndMap(ctx, kind, setHook, getHook, m)
	}
}

func contextWithOneHookAndMap(ctx context.Context, kind hookKind, setHook SetHookFunc, getHook GetHookFunc, m Map) context.Context {
	d := correlationsData{m: m}
	d.overrideHook(kind, setHook, getHook)
	if d.isHookless() {
		return ctx
	}
	return context.WithValue(ctx, correlationsKey, d)
}

// ContextWithNoHooks creates a context with all the hooks
// disabled. Also returns old set and get hooks. This function can be
// used to temporarily clear the context from hooks and then reinstate
// them by calling ContextWithSetHook and ContextWithGetHook functions
// passing the hooks returned by this function.
//
// This function should not be used by applications or libraries. It
// is mostly for interoperation with other observability APIs.
func ContextWithNoHooks(ctx context.Context) (context.Context, SetHookFunc, GetHookFunc) {
	switch v := ctx.Value(correlationsKey).(type) {
	case correlationsData:
		return context.WithValue(ctx, correlationsKey, v.m), v.setHook, v.getHook
	default:
		return ctx, nil, nil
	}
}

// ContextWithMap returns a context with the Map entered into it.
func ContextWithMap(ctx context.Context, m Map) context.Context {
	switch v := ctx.Value(correlationsKey).(type) {
	case correlationsData:
		v.m = m
		ctx = context.WithValue(ctx, correlationsKey, v)
		if v.setHook != nil {
			ctx = v.setHook(ctx)
		}
		return ctx
	default:
		return context.WithValue(ctx, correlationsKey, m)
	}
}

// NewContext returns a context with the map from passed context
// updated with the passed key-value pairs.
func NewContext(ctx context.Context, keyvalues ...core.KeyValue) context.Context {
	return ContextWithMap(ctx, MapFromContext(ctx).Apply(MapUpdate{
		MultiKV: keyvalues,
	}))
}

// MapFromContext gets the current Map from a Context.
func MapFromContext(ctx context.Context) Map {
	switch v := ctx.Value(correlationsKey).(type) {
	case correlationsData:
		if v.getHook != nil {
			return v.getHook(ctx, v.m)
		}
		return v.m
	case Map:
		return v
	default:
		return NewEmptyMap()
	}
}
