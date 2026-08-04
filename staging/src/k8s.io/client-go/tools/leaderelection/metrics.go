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

package leaderelection

import (
	"sync"
)

// This file provides abstractions for setting the provider (e.g., prometheus)
// of metrics.

type leaderMetricsAdapter interface {
	leaderOn(name string)
	leaderOff(name string)
	slowpathExercised(name string)
}

// LeaderMetric instruments metrics used in leader election.
type LeaderMetric interface {
	On(name string)
	Off(name string)
	SlowpathExercised(name string)
}

type defaultLeaderMetrics struct {
	leaders []LeaderMetric
}

func (m *defaultLeaderMetrics) leaderOn(name string) {
	if m == nil {
		return
	}
	for _, metric := range m.leaders {
		metric.On(name)
	}
}

func (m *defaultLeaderMetrics) leaderOff(name string) {
	if m == nil {
		return
	}
	for _, metric := range m.leaders {
		metric.Off(name)
	}
}

func (m *defaultLeaderMetrics) slowpathExercised(name string) {
	if m == nil {
		return
	}
	for _, metric := range m.leaders {
		metric.SlowpathExercised(name)
	}
}

type noMetrics struct{}

func (noMetrics) leaderOn(name string)          {}
func (noMetrics) leaderOff(name string)         {}
func (noMetrics) slowpathExercised(name string) {}

// MetricsProvider generates various metrics used by the leader election.
type MetricsProvider interface {
	NewLeaderMetric() LeaderMetric
}

var globalMetricsFactory = leaderMetricsFactory{}

type leaderMetricsFactory struct {
	lock             sync.RWMutex
	metricsProviders []MetricsProvider
}

func (f *leaderMetricsFactory) setProvider(mp MetricsProvider) {
	f.lock.Lock()
	defer f.lock.Unlock()
	f.metricsProviders = append(f.metricsProviders, mp)
}

func (f *leaderMetricsFactory) newLeaderMetrics() leaderMetricsAdapter {
	f.lock.RLock()
	providers := append([]MetricsProvider(nil), f.metricsProviders...)
	f.lock.RUnlock()

	if len(providers) == 0 {
		return noMetrics{}
	}

	leaders := make([]LeaderMetric, 0, len(providers))
	for _, provider := range providers {
		leaders = append(leaders, provider.NewLeaderMetric())
	}

	return &defaultLeaderMetrics{
		leaders: leaders,
	}
}

// SetProvider adds a metrics provider for all subsequently created leader
// elections.
func SetProvider(metricsProvider MetricsProvider) {
	globalMetricsFactory.setProvider(metricsProvider)
}
