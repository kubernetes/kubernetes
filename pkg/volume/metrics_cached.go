/*
Copyright 2014 The Kubernetes Authors.

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

package volume

import (
	"sync"
	"sync/atomic"
)

var _ StatsProvider = &cachedMetrics{}

// cachedMetrics represents a StatsProvider that wraps another provider and
// caches the result.
type cachedMetrics struct {
	wrapped     StatsProvider
	resultError error
	resultStats *Stats
	once        cacheOnce
}

// NewCachedMetrics creates a new cachedMetrics wrapping another
// StatsProvider and caching the results.
func NewCachedMetrics(provider StatsProvider) StatsProvider {
	return &cachedMetrics{wrapped: provider}
}

// GetStats runs the wrapped stats provider's GetStats methd once and
// caches the result. Will not cache result if there is an error.
// See StatsProvider.GetStats
func (md *cachedMetrics) GetStats() (*Stats, error) {
	md.once.cache(func() error {
		md.resultStats, md.resultError = md.wrapped.GetStats()
		return md.resultError
	})
	return md.resultStats, md.resultError
}

// Copied from sync.Once but we don't want to cache the results if there is an
// error
type cacheOnce struct {
	m    sync.Mutex
	done uint32
}

// Copied from sync.Once but we don't want to cache the results if there is an
// error
func (o *cacheOnce) cache(f func() error) {
	if atomic.LoadUint32(&o.done) == 1 {
		return
	}
	// Slow-path.
	o.m.Lock()
	defer o.m.Unlock()
	if o.done == 0 {
		err := f()
		if err == nil {
			atomic.StoreUint32(&o.done, 1)
		}
	}
}
