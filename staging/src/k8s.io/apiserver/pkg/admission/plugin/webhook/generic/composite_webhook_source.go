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
	"slices"
	"strings"
	"sync"

	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
)

// compositeWebhookSource combines multiple webhook sources into a single source.
// The relative order of static (manifest-based) and API-based webhooks is determined
// by staticOrder.
type compositeWebhookSource struct {
	staticSource Source
	apiSource    Source
	staticOrder  webhookadmission.StaticOrder

	mu              sync.RWMutex
	lastStaticSlice []webhook.WebhookAccessor
	lastAPISlice    []webhook.WebhookAccessor
	lastCombined    []webhook.WebhookAccessor
}

var _ Source = &compositeWebhookSource{}

// newCompositeWebhookSource creates a webhook source that combines static and API-based
// sources using staticOrder, which callers are expected to have validated and normalized
// (see NewWebhook).
// If staticSource is nil, only apiSource webhooks are returned.
func newCompositeWebhookSource(staticSource, apiSource Source, staticOrder webhookadmission.StaticOrder) Source {
	if staticSource == nil {
		return apiSource
	}
	return &compositeWebhookSource{
		staticSource: staticSource,
		apiSource:    apiSource,
		staticOrder:  staticOrder,
	}
}

// Webhooks returns all webhook accessors from both sources, merged according to the
// configured order.
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

	combined := mergeAccessors(staticSlice, apiSlice, c.staticOrder)

	c.lastStaticSlice = staticSlice
	c.lastAPISlice = apiSlice
	c.lastCombined = combined
	return combined
}

// mergeAccessors merges static and API-based accessors according to staticOrder.
func mergeAccessors(staticSlice, apiSlice []webhook.WebhookAccessor, staticOrder webhookadmission.StaticOrder) []webhook.WebhookAccessor {
	combined := make([]webhook.WebhookAccessor, 0, len(staticSlice)+len(apiSlice))

	switch staticOrder {
	case webhookadmission.ByName:
		combined = append(combined, staticSlice...)
		combined = append(combined, apiSlice...)
		// Both sources already group the webhooks of a configuration contiguously and in
		// the user-provided order, so a stable sort interleaves the configurations
		// without reordering the webhooks within one.
		slices.SortStableFunc(combined, func(a, b webhook.WebhookAccessor) int {
			return strings.Compare(a.GetConfigurationName(), b.GetConfigurationName())
		})

	case webhookadmission.StaticLast:
		combined = append(combined, apiSlice...)
		combined = append(combined, staticSlice...)

	default: // StaticFirst; NewWebhook normalizes the empty value
		combined = append(combined, staticSlice...)
		combined = append(combined, apiSlice...)
	}

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
