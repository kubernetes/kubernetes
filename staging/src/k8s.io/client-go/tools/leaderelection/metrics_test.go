/*
Copyright 2026 The Kubernetes Authors.

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
	"sync/atomic"
	"testing"
)

type countingLeaderMetric struct {
	onCount       atomic.Int32
	offCount      atomic.Int32
	slowpathCount atomic.Int32
}

func (m *countingLeaderMetric) On(string) {
	m.onCount.Add(1)
}

func (m *countingLeaderMetric) Off(string) {
	m.offCount.Add(1)
}

func (m *countingLeaderMetric) SlowpathExercised(string) {
	m.slowpathCount.Add(1)
}

type testMetricsProvider struct {
	metric LeaderMetric
}

func (p testMetricsProvider) NewLeaderMetric() LeaderMetric {
	return p.metric
}

func TestLeaderMetricsFactoryMultipleProviders(t *testing.T) {
	factory := leaderMetricsFactory{}
	first := &countingLeaderMetric{}
	second := &countingLeaderMetric{}

	factory.setProvider(testMetricsProvider{metric: first})
	factory.setProvider(testMetricsProvider{metric: second})

	metric := factory.newLeaderMetrics()
	metric.leaderOn("test")
	metric.leaderOff("test")
	metric.slowpathExercised("test")

	for name, provider := range map[string]*countingLeaderMetric{
		"first":  first,
		"second": second,
	} {
		if got := provider.onCount.Load(); got != 1 {
			t.Errorf("%s provider On calls = %d, want 1", name, got)
		}
		if got := provider.offCount.Load(); got != 1 {
			t.Errorf("%s provider Off calls = %d, want 1", name, got)
		}
		if got := provider.slowpathCount.Load(); got != 1 {
			t.Errorf("%s provider SlowpathExercised calls = %d, want 1", name, got)
		}
	}
}

func TestLeaderMetricsFactoryProvidersApplyToSubsequentMetrics(t *testing.T) {
	factory := leaderMetricsFactory{}
	first := &countingLeaderMetric{}
	second := &countingLeaderMetric{}

	factory.setProvider(testMetricsProvider{metric: first})
	beforeSecondProvider := factory.newLeaderMetrics()
	factory.setProvider(testMetricsProvider{metric: second})
	afterSecondProvider := factory.newLeaderMetrics()

	beforeSecondProvider.leaderOn("test")
	if got := first.onCount.Load(); got != 1 {
		t.Fatalf("first provider On calls = %d, want 1", got)
	}
	if got := second.onCount.Load(); got != 0 {
		t.Fatalf("second provider received calls from previously created metrics: got %d, want 0", got)
	}

	afterSecondProvider.leaderOn("test")
	if got := first.onCount.Load(); got != 2 {
		t.Errorf("first provider On calls = %d, want 2", got)
	}
	if got := second.onCount.Load(); got != 1 {
		t.Errorf("second provider On calls = %d, want 1", got)
	}
}

func TestLeaderMetricsFactoryConcurrentRegistration(t *testing.T) {
	const providerCount = 32

	factory := leaderMetricsFactory{}
	providers := make([]*countingLeaderMetric, providerCount)
	start := make(chan struct{})
	var wg sync.WaitGroup

	for i := range providers {
		providers[i] = &countingLeaderMetric{}
		wg.Add(1)
		go func(metric *countingLeaderMetric) {
			defer wg.Done()
			<-start
			factory.setProvider(testMetricsProvider{metric: metric})
		}(providers[i])
	}

	close(start)
	for i := 0; i < providerCount; i++ {
		factory.newLeaderMetrics().leaderOn("test")
	}
	wg.Wait()

	factory.newLeaderMetrics().leaderOn("test")
	for i, provider := range providers {
		if got := provider.onCount.Load(); got < 1 {
			t.Errorf("provider %d On calls = %d, want at least 1", i, got)
		}
	}
}
