/*
Copyright 2022 The KCP Authors.

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

package request

import (
	"context"
)

type shardKey int

const (
	// shardKey is the context key for the request.
	shardContextKey shardKey = iota

	// ShardAnnotationKey is the name of the annotation key used to denote an object's shard name.
	ShardAnnotationKey = "kcp.io/shard"
)

// Shard describes a shard
type Shard string

// Empty returns true if the name of the shard is empty.
func (s Shard) Empty() bool {
	return s == ""
}

// Wildcard checks if the given shard name matches wildcard.
// If true the query applies to all shards.
func (s Shard) Wildcard() bool {
	return s == "*"
}

// String casts Shard to string type
func (s Shard) String() string {
	return string(s)
}

// WithShard returns a context that holds the given shard.
func WithShard(parent context.Context, shard Shard) context.Context {
	return context.WithValue(parent, shardContextKey, shard)
}

// ShardFrom returns the value of the shard key in the context, or an empty value if there is no shard key.
func ShardFrom(ctx context.Context) Shard {
	shard, ok := ctx.Value(shardContextKey).(Shard)
	if !ok {
		return ""
	}
	return shard
}
