// Copyright 2017, OpenCensus Authors
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
//

package tag

import (
	"context"
)

// FromContext returns the tag map stored in the context.
func FromContext(ctx context.Context) *Map {
	// The returned tag map shouldn't be mutated.
	ts := ctx.Value(mapCtxKey)
	if ts == nil {
		return nil
	}
	return ts.(*Map)
}

// NewContext creates a new context with the given tag map.
// To propagate a tag map to downstream methods and downstream RPCs, add a tag map
// to the current context. NewContext will return a copy of the current context,
// and put the tag map into the returned one.
// If there is already a tag map in the current context, it will be replaced with m.
func NewContext(ctx context.Context, m *Map) context.Context {
	return context.WithValue(ctx, mapCtxKey, m)
}

type ctxKey struct{}

var mapCtxKey = ctxKey{}
