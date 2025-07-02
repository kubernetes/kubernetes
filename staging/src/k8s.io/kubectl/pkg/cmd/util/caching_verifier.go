/*
Copyright 2025 The Kubernetes Authors.

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

package util

import (
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/resource"
)

// cachingVerifier wraps the given Verifier to cache return values forever.
type cachingVerifier struct {
	cache map[schema.GroupVersionKind]error
	mu    sync.RWMutex
	next  resource.Verifier
}

// newCachingVerifier creates a new cache using the given underlying verifier.
func newCachingVerifier(next resource.Verifier) *cachingVerifier {
	return &cachingVerifier{
		cache: make(map[schema.GroupVersionKind]error),
		next:  next,
	}
}

// HasSupport implements resource.Verifier. It cached return values from the underlying verifier forever.
func (cv *cachingVerifier) HasSupport(gvk schema.GroupVersionKind) error {
	// Try to get the cached value.
	cv.mu.RLock()
	err, ok := cv.cache[gvk]
	cv.mu.RUnlock()
	if ok {
		return err
	}

	// Cache miss. Get the actual result.
	err = cv.next.HasSupport(gvk)

	// Update the cache.
	cv.mu.Lock()
	cv.cache[gvk] = err
	cv.mu.Unlock()
	return err
}
