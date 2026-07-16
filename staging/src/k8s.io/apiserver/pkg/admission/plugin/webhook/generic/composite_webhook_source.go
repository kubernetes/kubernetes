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
	"sync"

	"k8s.io/apiserver/pkg/admission/plugin/webhook"
)

// compositeWebhookSource combines multiple webhook sources into a single source.
// Static (manifest-based) webhooks are returned before API-based webhooks.
type compositeWebhookSource struct {
	staticSource Source
	apiSource    Source

	mu              sync.RWMutex
	lastStaticSlice []webhook.WebhookAccessor
	lastAPISlice    []webhook.WebhookAccessor
	lastCombined    []webhook.WebhookAccessor
}

var _ Source = &compositeWebhookSource{}

// NewCompositeWebhookSource creates a webhook source that combines static and API-based sources.
// Static webhooks are evaluated first, followed by API-based webhooks.
// If staticSource is nil, only apiSource webhooks are returned.
func NewCompositeWebhookSource(staticSource, apiSource Source) Source {
	if staticSource == nil {
		return apiSource
	}
	return &compositeWebhookSource{
		staticSource: staticSource,
		apiSource:    apiSource,
	}
}

// Webhooks returns all webhook accessors from both sources.
// Static webhooks come first, followed by API-based webhooks.
// The combined slice is cached and reused when the underlying slices haven't changed.
func (c *compositeWebhookSource) Webhooks() []webhook.WebhookAccessor {
	var staticSlice, apiSlice []webhook.WebhookAccessor
	if c.staticSource != nil {
		staticSlice = c.staticSource.Webhooks()
	}
	if c.apiSource != nil {
		apiSlice = c.apiSource.Webhooks()
	}

	c.mu.RLock()
	if slicesAreEqual(c.lastStaticSlice, staticSlice) && slicesAreEqual(c.lastAPISlice, apiSlice) {
		combined := c.lastCombined
		c.mu.RUnlock()
		return combined
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()
	// Re-check under write lock.
	if slicesAreEqual(c.lastStaticSlice, staticSlice) && slicesAreEqual(c.lastAPISlice, apiSlice) {
		return c.lastCombined
	}

	combined := make([]webhook.WebhookAccessor, 0, len(staticSlice)+len(apiSlice))
	combined = append(combined, staticSlice...)
	combined = append(combined, apiSlice...)

	c.lastStaticSlice = staticSlice
	c.lastAPISlice = apiSlice
	c.lastCombined = combined
	return combined
}

// slicesAreEqual reports whether two slices share the same backing array and length.
func slicesAreEqual[T any](a, b []T) bool {
	return len(a) == len(b) && (len(a) == 0 || &a[0] == &b[0])
}

// HasSynced returns true only when both sources have synced.
func (c *compositeWebhookSource) HasSynced() bool {
	staticSynced := c.staticSource == nil || c.staticSource.HasSynced()
	apiSynced := c.apiSource == nil || c.apiSource.HasSynced()
	return staticSynced && apiSynced
}
