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

package baggage // import "go.opentelemetry.io/otel/baggage"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/internal/baggage"
)

// Set returns a copy of the set of baggage key-values in ctx.
func Set(ctx context.Context) attribute.Set {
	// TODO (MrAlias, #1222): The underlying storage, the Map, shares many of
	// the functional elements of the attribute.Set. These should be unified so
	// this conversion is unnecessary and there is no performance hit calling
	// this.
	m := baggage.MapFromContext(ctx)
	values := make([]attribute.KeyValue, 0, m.Len())
	m.Foreach(func(kv attribute.KeyValue) bool {
		values = append(values, kv)
		return true
	})
	return attribute.NewSet(values...)
}

// Value returns the value related to key in the baggage of ctx. If no
// value is set, the returned attribute.Value will be an uninitialized zero-value
// with type INVALID.
func Value(ctx context.Context, key attribute.Key) attribute.Value {
	v, _ := baggage.MapFromContext(ctx).Value(key)
	return v
}

// ContextWithValues returns a copy of parent with pairs updated in the baggage.
func ContextWithValues(parent context.Context, pairs ...attribute.KeyValue) context.Context {
	m := baggage.MapFromContext(parent).Apply(baggage.MapUpdate{
		MultiKV: pairs,
	})
	return baggage.ContextWithMap(parent, m)
}

// ContextWithoutValues returns a copy of parent in which the values related
// to keys have been removed from the baggage.
func ContextWithoutValues(parent context.Context, keys ...attribute.Key) context.Context {
	m := baggage.MapFromContext(parent).Apply(baggage.MapUpdate{
		DropMultiK: keys,
	})
	return baggage.ContextWithMap(parent, m)
}

// ContextWithEmpty returns a copy of parent without baggage.
func ContextWithEmpty(parent context.Context) context.Context {
	return baggage.ContextWithNoCorrelationData(parent)
}
