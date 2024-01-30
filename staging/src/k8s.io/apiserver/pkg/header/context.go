/*
Copyright 2024 The Kubernetes Authors.

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

package header

import (
	"context"
	"net/http"
	"sync"
)

// The key type is unexported to prevent collisions
type key int

const (
	// headerAccessorKey is the context key for the header accessor.
	headerAccessorKey key = iota
)

type Header interface {
	// Add adds or appends the given key/value
	Add(key, value string)
	// Set replaces the given key/value
	Set(key, value string)
	// Del removes any headers for key
	Del(key string)
	// Detach prevents future Add/Set/Del calls from propagating
	// to the shared header set (makes them local-only).
	Detach()
}

type header struct {
	mu       sync.Mutex
	detached bool
	header   http.Header
}

func (h *header) Add(key, value string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.header.Add(key, value)
}
func (h *header) Set(key, value string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.header.Set(key, value)
}
func (h *header) Del(key string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.header.Del(key)
}
func (h *header) Detach() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.detached {
		h.detached = true
		h.header = h.header.Clone()
	}
}

// WithSafeHeaderWriter adds an interface for setting response headers that is threadsafe.
func WithSafeHeaderWriter(ctx context.Context, h http.Header) context.Context {
	return context.WithValue(ctx, headerAccessorKey, &header{header: h})
}

// SafeResponseHeader returns an interface for setting response headers that is threadsafe.
// If no instance is registered into the context, (nil, false) is returned.
func SafeResponseHeader(ctx context.Context) (Header, bool) {
	accessor, ok := ctx.Value(headerAccessorKey).(*header)
	return accessor, ok
}
