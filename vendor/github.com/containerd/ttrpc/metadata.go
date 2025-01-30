/*
   Copyright The containerd Authors.

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

package ttrpc

import (
	"context"
	"strings"
)

// MD is the user type for ttrpc metadata
type MD map[string][]string

// Get returns the metadata for a given key when they exist.
// If there is no metadata, a nil slice and false are returned.
func (m MD) Get(key string) ([]string, bool) {
	key = strings.ToLower(key)
	list, ok := m[key]
	if !ok || len(list) == 0 {
		return nil, false
	}

	return list, true
}

// Set sets the provided values for a given key.
// The values will overwrite any existing values.
// If no values provided, a key will be deleted.
func (m MD) Set(key string, values ...string) {
	key = strings.ToLower(key)
	if len(values) == 0 {
		delete(m, key)
		return
	}
	m[key] = values
}

// Append appends additional values to the given key.
func (m MD) Append(key string, values ...string) {
	key = strings.ToLower(key)
	if len(values) == 0 {
		return
	}
	current, ok := m[key]
	if ok {
		m.Set(key, append(current, values...)...)
	} else {
		m.Set(key, values...)
	}
}

func (m MD) setRequest(r *Request) {
	for k, values := range m {
		for _, v := range values {
			r.Metadata = append(r.Metadata, &KeyValue{
				Key:   k,
				Value: v,
			})
		}
	}
}

func (m MD) fromRequest(r *Request) {
	for _, kv := range r.Metadata {
		m[kv.Key] = append(m[kv.Key], kv.Value)
	}
}

type metadataKey struct{}

// GetMetadata retrieves metadata from context.Context (previously attached with WithMetadata)
func GetMetadata(ctx context.Context) (MD, bool) {
	metadata, ok := ctx.Value(metadataKey{}).(MD)
	return metadata, ok
}

// GetMetadataValue gets a specific metadata value by name from context.Context
func GetMetadataValue(ctx context.Context, name string) (string, bool) {
	metadata, ok := GetMetadata(ctx)
	if !ok {
		return "", false
	}

	if list, ok := metadata.Get(name); ok {
		return list[0], true
	}

	return "", false
}

// WithMetadata attaches metadata map to a context.Context
func WithMetadata(ctx context.Context, md MD) context.Context {
	return context.WithValue(ctx, metadataKey{}, md)
}
