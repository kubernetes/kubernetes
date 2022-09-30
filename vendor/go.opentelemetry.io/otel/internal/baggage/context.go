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

package baggage // import "go.opentelemetry.io/otel/internal/baggage"

import "context"

type baggageContextKeyType int

const baggageKey baggageContextKeyType = iota

// SetHookFunc is a callback called when storing baggage in the context.
type SetHookFunc func(context.Context, List) context.Context

// GetHookFunc is a callback called when getting baggage from the context.
type GetHookFunc func(context.Context, List) List

type baggageState struct {
	list List

	setHook SetHookFunc
	getHook GetHookFunc
}

// ContextWithSetHook returns a copy of parent with hook configured to be
// invoked every time ContextWithBaggage is called.
//
// Passing nil SetHookFunc creates a context with no set hook to call.
func ContextWithSetHook(parent context.Context, hook SetHookFunc) context.Context {
	var s baggageState
	if v, ok := parent.Value(baggageKey).(baggageState); ok {
		s = v
	}

	s.setHook = hook
	return context.WithValue(parent, baggageKey, s)
}

// ContextWithGetHook returns a copy of parent with hook configured to be
// invoked every time FromContext is called.
//
// Passing nil GetHookFunc creates a context with no get hook to call.
func ContextWithGetHook(parent context.Context, hook GetHookFunc) context.Context {
	var s baggageState
	if v, ok := parent.Value(baggageKey).(baggageState); ok {
		s = v
	}

	s.getHook = hook
	return context.WithValue(parent, baggageKey, s)
}

// ContextWithList returns a copy of parent with baggage. Passing nil list
// returns a context without any baggage.
func ContextWithList(parent context.Context, list List) context.Context {
	var s baggageState
	if v, ok := parent.Value(baggageKey).(baggageState); ok {
		s = v
	}

	s.list = list
	ctx := context.WithValue(parent, baggageKey, s)
	if s.setHook != nil {
		ctx = s.setHook(ctx, list)
	}

	return ctx
}

// ListFromContext returns the baggage contained in ctx.
func ListFromContext(ctx context.Context) List {
	switch v := ctx.Value(baggageKey).(type) {
	case baggageState:
		if v.getHook != nil {
			return v.getHook(ctx, v.list)
		}
		return v.list
	default:
		return nil
	}
}
