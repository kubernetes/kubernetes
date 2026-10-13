/*
Copyright The Kubernetes Authors.

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

package generic

import (
	"context"
	"sync"
)

// HookSource provides hooks and sync status. This is the minimal interface
// needed for static sources that don't have their own Run lifecycle.
type HookSource[H Hook] interface {
	Hooks() []H
	HasSynced() bool
}

// compositePolicySource combines multiple policy sources into a single source.
// Static (manifest-based) policies are returned before API-based policies.
type compositePolicySource[H Hook] struct {
	staticSource HookSource[H]
	apiSource    Source[H]

	mu           sync.RWMutex
	lastStatic   []H
	lastAPI      []H
	lastCombined []H
}

var _ Source[Hook] = &compositePolicySource[Hook]{}

// NewCompositePolicySource creates a policy source that combines static and API-based sources.
// Static policies are evaluated first, followed by API-based policies.
// If staticSource is nil, only apiSource policies are returned.
func NewCompositePolicySource[H Hook](staticSource HookSource[H], apiSource Source[H]) Source[H] {
	if staticSource == nil {
		return apiSource
	}
	return &compositePolicySource[H]{
		staticSource: staticSource,
		apiSource:    apiSource,
	}
}

// Hooks returns all policy hooks from both sources.
// Static policies come first, followed by API-based policies.
// The combined slice is cached and reused when the underlying slices haven't changed.
func (c *compositePolicySource[H]) Hooks() []H {
	var staticHooks, apiHooks []H

	// Static policies first (platform policies take precedence)
	if c.staticSource != nil {
		staticHooks = c.staticSource.Hooks()
	}

	// Then API-based policies
	if c.apiSource != nil {
		apiHooks = c.apiSource.Hooks()
	}

	c.mu.RLock()
	if slicesAreEqual(staticHooks, c.lastStatic) && slicesAreEqual(apiHooks, c.lastAPI) {
		combined := c.lastCombined
		c.mu.RUnlock()
		return combined
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()
	// Re-check under write lock.
	if slicesAreEqual(staticHooks, c.lastStatic) && slicesAreEqual(apiHooks, c.lastAPI) {
		return c.lastCombined
	}

	combined := make([]H, 0, len(staticHooks)+len(apiHooks))
	combined = append(combined, staticHooks...)
	combined = append(combined, apiHooks...)

	c.lastStatic = staticHooks
	c.lastAPI = apiHooks
	c.lastCombined = combined
	return combined
}

// slicesAreEqual reports whether two slices share the same backing array and length.
func slicesAreEqual[T any](a, b []T) bool {
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0])
}

// Run starts the API-based source. The static source is started separately
// by the plugin via its own goroutine (file watcher), so this method only
// needs to handle the API source lifecycle.
func (c *compositePolicySource[H]) Run(ctx context.Context) error {
	if c.apiSource != nil {
		return c.apiSource.Run(ctx)
	}
	return nil
}

// HasSynced returns true only when both sources have synced.
func (c *compositePolicySource[H]) HasSynced() bool {
	staticSynced := c.staticSource == nil || c.staticSource.HasSynced()
	apiSynced := c.apiSource == nil || c.apiSource.HasSynced()
	return staticSynced && apiSynced
}
