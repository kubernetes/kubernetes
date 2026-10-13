/*
 * Copyright 2024 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package stats

import "maps"

// MetricSet is a set of metrics to record. Once created, MetricSet is immutable,
// however Add and Remove can make copies with specific metrics added or
// removed, respectively.
//
// Do not construct directly; use NewMetricSet instead.
type MetricSet struct {
	// metrics are the set of metrics to initialize.
	metrics map[string]bool
}

// NewMetricSet returns a MetricSet containing metricNames.
func NewMetricSet(metricNames ...string) *MetricSet {
	newMetrics := make(map[string]bool)
	for _, metric := range metricNames {
		newMetrics[metric] = true
	}
	return &MetricSet{metrics: newMetrics}
}

// Metrics returns the metrics set. The returned map is read-only and must not
// be modified.
func (m *MetricSet) Metrics() map[string]bool {
	return m.metrics
}

// Add adds the metricNames to the metrics set and returns a new copy with the
// additional metrics.
func (m *MetricSet) Add(metricNames ...string) *MetricSet {
	newMetrics := make(map[string]bool)
	for metric := range m.metrics {
		newMetrics[metric] = true
	}

	for _, metric := range metricNames {
		newMetrics[metric] = true
	}
	return &MetricSet{metrics: newMetrics}
}

// Join joins the metrics passed in with the metrics set, and returns a new copy
// with the merged metrics.
func (m *MetricSet) Join(metrics *MetricSet) *MetricSet {
	newMetrics := make(map[string]bool)
	maps.Copy(newMetrics, m.metrics)
	maps.Copy(newMetrics, metrics.metrics)
	return &MetricSet{metrics: newMetrics}
}

// Remove removes the metricNames from the metrics set and returns a new copy
// with the metrics removed.
func (m *MetricSet) Remove(metricNames ...string) *MetricSet {
	newMetrics := make(map[string]bool)
	for metric := range m.metrics {
		newMetrics[metric] = true
	}

	for _, metric := range metricNames {
		delete(newMetrics, metric)
	}
	return &MetricSet{metrics: newMetrics}
}
