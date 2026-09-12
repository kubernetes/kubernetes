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
	"slices"
	"sync"
)

var globalMetricsFactory = &metricsProviderFactory{}

type metricsProviderFactory struct {
	mu        sync.RWMutex
	providers []MetricsProvider
}

func (f *metricsProviderFactory) setProvider(provider MetricsProvider) {
	if provider == nil {
		return
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.providers = append(f.providers, provider)
}

func (f *metricsProviderFactory) getProvider() MetricsProvider {
	f.mu.RLock()
	defer f.mu.RUnlock()
	switch len(f.providers) {
	case 0:
		return noopMetricsProvider{}
	case 1:
		return f.providers[0]
	default:
		// Provider callbacks run after the lock is released. Copy the slice so
		// subsequent registrations cannot change this queue's providers.
		return multiMetricsProvider(slices.Clone(f.providers))
	}
}

type multiMetricsProvider []MetricsProvider

func (p multiMetricsProvider) NewDepthMetric(name string) GaugeMetric {
	metrics := make(multiGaugeMetric, len(p))
	for i, provider := range p {
		metrics[i] = provider.NewDepthMetric(name)
	}
	return metrics
}

func (p multiMetricsProvider) NewAddsMetric(name string) CounterMetric {
	metrics := make(multiCounterMetric, len(p))
	for i, provider := range p {
		metrics[i] = provider.NewAddsMetric(name)
	}
	return metrics
}

func (p multiMetricsProvider) NewLatencyMetric(name string) HistogramMetric {
	metrics := make(multiHistogramMetric, len(p))
	for i, provider := range p {
		metrics[i] = provider.NewLatencyMetric(name)
	}
	return metrics
}

func (p multiMetricsProvider) NewWorkDurationMetric(name string) HistogramMetric {
	metrics := make(multiHistogramMetric, len(p))
	for i, provider := range p {
		metrics[i] = provider.NewWorkDurationMetric(name)
	}
	return metrics
}

func (p multiMetricsProvider) NewUnfinishedWorkSecondsMetric(name string) SettableGaugeMetric {
	metrics := make(multiSettableGaugeMetric, len(p))
	for i, provider := range p {
		metrics[i] = provider.NewUnfinishedWorkSecondsMetric(name)
	}
	return metrics
}

func (p multiMetricsProvider) NewLongestRunningProcessorSecondsMetric(name string) SettableGaugeMetric {
	metrics := make(multiSettableGaugeMetric, len(p))
	for i, provider := range p {
		metrics[i] = provider.NewLongestRunningProcessorSecondsMetric(name)
	}
	return metrics
}

func (p multiMetricsProvider) NewRetriesMetric(name string) CounterMetric {
	metrics := make(multiCounterMetric, len(p))
	for i, provider := range p {
		metrics[i] = provider.NewRetriesMetric(name)
	}
	return metrics
}

type multiGaugeMetric []GaugeMetric

func (m multiGaugeMetric) Inc() {
	for _, metric := range m {
		metric.Inc()
	}
}

func (m multiGaugeMetric) Dec() {
	for _, metric := range m {
		metric.Dec()
	}
}

type multiCounterMetric []CounterMetric

func (m multiCounterMetric) Inc() {
	for _, metric := range m {
		metric.Inc()
	}
}

type multiHistogramMetric []HistogramMetric

func (m multiHistogramMetric) Observe(value float64) {
	for _, metric := range m {
		metric.Observe(value)
	}
}

type multiSettableGaugeMetric []SettableGaugeMetric

func (m multiSettableGaugeMetric) Set(value float64) {
	for _, metric := range m {
		metric.Set(value)
	}
}
