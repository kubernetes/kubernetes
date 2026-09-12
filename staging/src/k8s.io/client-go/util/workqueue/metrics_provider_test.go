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

package workqueue

import (
	"sync"
	"testing"
	"time"

	testingclock "k8s.io/utils/clock/testing"
)

func resetMetricsProviders(t *testing.T) {
	t.Helper()
	previous := globalMetricsFactory
	globalMetricsFactory = &metricsProviderFactory{}
	t.Cleanup(func() { globalMetricsFactory = previous })
}

func TestMultipleMetricsProviders(t *testing.T) {
	resetMetricsProviders(t)
	first, second := &testMetricsProvider{}, &testMetricsProvider{}
	SetProvider(first)
	SetProvider(second)
	c := testingclock.NewFakeClock(time.Unix(0, 0))
	q := NewTypedDelayingQueueWithConfig(TypedDelayingQueueConfig[string]{Name: "test", Clock: c})
	defer q.ShutDown()
	q.AddAfter("item", 0)
	c.Step(time.Second)
	item, _ := q.Get()
	c.Step(2 * time.Second)
	base := q.(*delayingType[string]).TypedInterface.(*Typed[string])
	base.cond.L.Lock()
	base.metrics.updateUnfinishedWork()
	for i, provider := range []*testMetricsProvider{first, second} {
		if got := provider.unfinished.gaugeValue(); got != 2 {
			t.Errorf("provider %d: unfinished work = %v, want 2", i, got)
		}
		if got := provider.longest.gaugeValue(); got != 2 {
			t.Errorf("provider %d: longest running processor = %v, want 2", i, got)
		}
	}
	base.cond.L.Unlock()
	q.Done(item)
	for i, provider := range []*testMetricsProvider{first, second} {
		if got := provider.depth.gaugeValue(); got != 0 {
			t.Errorf("provider %d: depth = %v, want 0", i, got)
		}
		if got := provider.adds.gaugeValue(); got != 1 {
			t.Errorf("provider %d: adds = %v, want 1", i, got)
		}
		if got := provider.latency.observationValue(); got != 1 || provider.latency.observationCount() != 1 {
			t.Errorf("provider %d: latency = %v with %d observations, want 1 with 1 observation", i, got, provider.latency.observationCount())
		}
		if got := provider.duration.observationValue(); got != 2 || provider.duration.observationCount() != 1 {
			t.Errorf("provider %d: work duration = %v with %d observations, want 2 with 1 observation", i, got, provider.duration.observationCount())
		}
		if got := provider.retries.gaugeValue(); got != 1 {
			t.Errorf("provider %d: retries = %v, want 1", i, got)
		}
	}
}

func TestMetricsProviderFactoryDefaults(t *testing.T) {
	factory := &metricsProviderFactory{}
	factory.setProvider(nil)
	if got := factory.getProvider(); got != (noopMetricsProvider{}) {
		t.Fatalf("empty factory provider = %T, want noopMetricsProvider", got)
	}
	provider := &testMetricsProvider{}
	factory.setProvider(provider)
	if got := factory.getProvider(); got != provider {
		t.Fatalf("single provider = %T, want the original provider", got)
	}
}

func TestMetricsProviderQueueConfiguration(t *testing.T) {
	constructors := map[string]func(string, MetricsProvider) TypedInterface[string]{
		"queue": func(name string, provider MetricsProvider) TypedInterface[string] {
			return NewTypedWithConfig(TypedQueueConfig[string]{Name: name, MetricsProvider: provider})
		},
		"delaying": func(name string, provider MetricsProvider) TypedInterface[string] {
			return NewTypedDelayingQueueWithConfig(TypedDelayingQueueConfig[string]{Name: name, MetricsProvider: provider})
		},
		"rate limiting": func(name string, provider MetricsProvider) TypedInterface[string] {
			return NewTypedRateLimitingQueueWithConfig(NewTypedItemExponentialFailureRateLimiter[string](0, 0), TypedRateLimitingQueueConfig[string]{Name: name, MetricsProvider: provider})
		},
	}
	for name, newQueue := range constructors {
		t.Run(name, func(t *testing.T) {
			resetMetricsProviders(t)
			first, second, override := &testMetricsProvider{}, &testMetricsProvider{}, &testMetricsProvider{}
			SetProvider(first)
			old := newQueue("old", nil)
			defer old.ShutDown()
			SetProvider(second)
			current := newQueue("current", nil)
			defer current.ShutDown()
			custom := newQueue("custom", override)
			defer custom.ShutDown()
			unnamed := newQueue("", nil)
			defer unnamed.ShutDown()
			unnamedCustom := newQueue("", override)
			defer unnamedCustom.ShutDown()
			for _, q := range []TypedInterface[string]{old, current, custom, unnamed, unnamedCustom} {
				switch q := q.(type) {
				case TypedRateLimitingInterface[string]:
					q.AddRateLimited("item")
					q.Forget("item")
				case TypedDelayingInterface[string]:
					q.AddAfter("item", 0)
				default:
					q.Add("item")
				}
				item, _ := q.Get()
				q.Done(item)
			}
			for _, check := range []struct {
				provider *testMetricsProvider
				want     float64
			}{{first, 2}, {second, 1}, {override, 1}} {
				if got := check.provider.adds.gaugeValue(); got != check.want {
					t.Errorf("adds = %v, want %v", got, check.want)
				}
				wantRetries := check.want
				if name == "queue" {
					wantRetries = 0
				}
				if got := check.provider.retries.gaugeValue(); got != wantRetries {
					t.Errorf("retries = %v, want %v", got, wantRetries)
				}
			}
		})
	}
}

type registeringMetricsProvider struct {
	MetricsProvider
	onDepth func()
}

func (p registeringMetricsProvider) NewDepthMetric(name string) GaugeMetric {
	p.onDepth()
	return p.MetricsProvider.NewDepthMetric(name)
}

func TestMetricsProviderRegistrationDuringConstruction(t *testing.T) {
	for _, name := range []string{"single", "multiple"} {
		t.Run(name, func(t *testing.T) {
			resetMetricsProviders(t)
			first, second := &testMetricsProvider{}, &testMetricsProvider{}
			var once sync.Once
			SetProvider(registeringMetricsProvider{MetricsProvider: first, onDepth: func() {
				once.Do(func() { SetProvider(second) })
			}})
			if name == "multiple" {
				SetProvider(noopMetricsProvider{})
			}
			created := make(chan TypedDelayingInterface[string], 1)
			go func() {
				created <- NewTypedDelayingQueueWithConfig(TypedDelayingQueueConfig[string]{Name: "test"})
			}()
			select {
			case q := <-created:
				defer q.ShutDown()
				q.AddAfter("item", 0)
				if got := second.adds.gaugeValue(); got != 0 {
					t.Errorf("late provider adds = %v, want 0", got)
				}
				if got := second.retries.gaugeValue(); got != 0 {
					t.Errorf("late provider retries = %v, want 0", got)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("queue construction blocked while registering a provider from a callback")
			}
			next := NewTypedDelayingQueueWithConfig(TypedDelayingQueueConfig[string]{Name: "next"})
			defer next.ShutDown()
			next.AddAfter("item", 0)
			if got := second.adds.gaugeValue(); got != 1 {
				t.Errorf("late provider adds for new queue = %v, want 1", got)
			}
			if got := second.retries.gaugeValue(); got != 1 {
				t.Errorf("late provider retries for new queue = %v, want 1", got)
			}
		})
	}
}

func TestMetricsProviderConcurrentRegistration(t *testing.T) {
	resetMetricsProviders(t)
	const count = 32
	providers := make([]*testMetricsProvider, count)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range providers {
		providers[i] = &testMetricsProvider{}
		wg.Go(func() {
			<-start
			SetProvider(providers[i])
		})
		wg.Go(func() {
			<-start
			q := NewTypedDelayingQueueWithConfig(TypedDelayingQueueConfig[string]{Name: "concurrent"})
			defer q.ShutDown()
			q.AddAfter("item", 0)
			item, _ := q.Get()
			q.Done(item)
		})
	}
	close(start)
	wg.Wait()
	adds, retries := make([]float64, count), make([]float64, count)
	for i, provider := range providers {
		adds[i], retries[i] = provider.adds.gaugeValue(), provider.retries.gaugeValue()
	}
	q := NewTypedDelayingQueueWithConfig(TypedDelayingQueueConfig[string]{Name: "all"})
	defer q.ShutDown()
	q.AddAfter("item", 0)
	for i, provider := range providers {
		if got := provider.adds.gaugeValue(); got != adds[i]+1 {
			t.Errorf("provider %d: adds = %v, want %v", i, got, adds[i]+1)
		}
		if got := provider.retries.gaugeValue(); got != retries[i]+1 {
			t.Errorf("provider %d: retries = %v, want %v", i, got, retries[i]+1)
		}
	}
}
