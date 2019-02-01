/*
Copyright 2019 The Kubernetes Authors.

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

package manager

import (
	"k8s.io/client-go/tools/cache"
)

type noopMetric struct{}

func (noopMetric) Inc()            {}
func (noopMetric) Dec()            {}
func (noopMetric) Observe(float64) {}
func (noopMetric) Set(float64)     {}

// NoopMetricsProvider implements interface of reflector metrics
// by faking all the metrics.
type NoopMetricsProvider struct{}

func (NoopMetricsProvider) NewListsMetric(name string) cache.CounterMetric {
	return noopMetric{}
}
func (NoopMetricsProvider) NewListDurationMetric(name string) cache.SummaryMetric {
	return noopMetric{}
}
func (NoopMetricsProvider) NewItemsInListMetric(name string) cache.SummaryMetric   {
	return noopMetric{}
}
func (NoopMetricsProvider) NewWatchesMetric(name string) cache.CounterMetric       {
	return noopMetric{}
}
func (NoopMetricsProvider) NewShortWatchesMetric(name string) cache.CounterMetric  {
	return noopMetric{}
}
func (NoopMetricsProvider) NewWatchDurationMetric(name string) cache.SummaryMetric {
	return noopMetric{}
}
func (NoopMetricsProvider) NewItemsInWatchMetric(name string) cache.SummaryMetric  {
	return noopMetric{}
}
func (NoopMetricsProvider) NewLastResourceVersionMetric(name string) cache.GaugeMetric {
	return noopMetric{}
}
