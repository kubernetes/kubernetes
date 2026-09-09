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

package configmetrics

import (
	"slices"
	"sync/atomic"

	"k8s.io/component-base/metrics"
	"k8s.io/utils/ptr"
)

// HashProvider is an interface for getting the current config hash values
type HashProvider interface {
	GetCurrentHashes() []string
	SetHashes(hashes ...string)
}

// AtomicHashProvider implements HashProvider using a single atomic pointer to a slice
type AtomicHashProvider struct {
	hashes *atomic.Pointer[[]string]
}

// NewAtomicHashProvider creates a new atomic hash provider
func NewAtomicHashProvider() *AtomicHashProvider {
	p := &AtomicHashProvider{
		hashes: &atomic.Pointer[[]string]{},
	}
	// Initialize with empty slice
	p.hashes.Store(ptr.To([]string{}))
	return p
}

func (h *AtomicHashProvider) GetCurrentHashes() []string {
	hashesPtr := h.hashes.Load()
	if hashesPtr == nil {
		// should never happen, but just in case
		return []string{}
	}
	// Return a copy to prevent external modification
	hashes := *hashesPtr
	return slices.Clone(hashes)
}

func (h *AtomicHashProvider) SetHashes(hashes ...string) {
	hashCopy := slices.Clone(hashes)
	h.hashes.Store(&hashCopy)
}

// NewConfigInfoCustomCollector creates a custom collector for config hash info metrics.
// This eliminates the need for state management and locks by collecting metrics on demand.
func NewConfigInfoCustomCollector(desc *metrics.Desc, hashProvider HashProvider) metrics.StableCollector {
	return &configInfoCustomCollector{
		desc:         desc,
		hashProvider: hashProvider,
	}
}

type configInfoCustomCollector struct {
	metrics.BaseStableCollector
	desc         *metrics.Desc
	hashProvider HashProvider
}

var _ metrics.StableCollector = &configInfoCustomCollector{}

func (c *configInfoCustomCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- c.desc
}

func (c *configInfoCustomCollector) CollectWithStability(ch chan<- metrics.Metric) {
	hashes := c.hashProvider.GetCurrentHashes()
	if len(hashes) == 0 {
		return
	}

	ch <- metrics.NewLazyConstMetric(c.desc, metrics.GaugeValue, 1, hashes...)
}
