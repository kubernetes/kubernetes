/*
Copyright 2018 The Kubernetes Authors.

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

package logs

import (
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"
)

const expirationTimeout = 10 * time.Second

type metricsCacheItem struct {
	updated time.Time
	metrics volume.MetricsProvider
}

type metricsCache struct {
	mu    sync.RWMutex
	clock clock.Clock
	cache map[string]*metricsCacheItem
}

func newMetricsCache() *metricsCache {
	return &metricsCache{
		clock: clock.RealClock{},
		cache: make(map[string]*metricsCacheItem),
	}
}

func (m *metricsCache) start() {
	go wait.Forever(m.cleanup, 2*expirationTimeout)
}

func (m *metricsCache) cleanup() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for p, i := range m.cache {
		// Remove cache item if the cache is not updated/used for 2 * expirationTimeout.
		if m.clock.Since(i.updated) >= 2*expirationTimeout {
			delete(m.cache, p)
		}
	}
}

func (m *metricsCache) get(path string) volume.MetricsProvider {
	m.mu.Lock()
	defer m.mu.Unlock()
	item, ok := m.cache[path]
	if ok && m.clock.Since(item.updated) < expirationTimeout {
		return item.metrics
	}
	item = &metricsCacheItem{
		updated: m.clock.Now(),
		metrics: volume.NewCachedMetrics(volume.NewMetricsDu(path)),
	}
	m.cache[path] = item
	return item.metrics
}
