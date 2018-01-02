// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metricsink

import (
	"sync"
	"time"

	"k8s.io/heapster/metrics/core"
)

// A simple in-memory storage for metrics. It divides metrics into 2 categories
// * metrics that need to be stored for couple minutes.
// * metrics that need to be stored for longer time (15 min, 1 hour).
// The user of this struct needs to decide what are the long-stored metrics uprfront.
type MetricSink struct {
	// Request can come from other threads.
	lock sync.Mutex

	// List of metrics that will be stored for up to X seconds.
	longStoreMetrics   []string
	longStoreDuration  time.Duration
	shortStoreDuration time.Duration

	// Stores full DataBatch with all metrics and labels.
	shortStore []*core.DataBatch
	// Memory-efficient long/mid term storage for metrics.
	longStore []*multimetricStore
}

// Stores values of a single metrics for different MetricSets.
// Assumes that the user knows what the metric is.
type int64Store map[string]int64

type multimetricStore struct {
	// Timestamp of the batch from which the metrics were taken.
	timestamp time.Time
	// Metric name to int64store with metric values.
	store map[string]int64Store
}

func buildMultimetricStore(metrics []string, batch *core.DataBatch) *multimetricStore {
	store := multimetricStore{
		timestamp: batch.Timestamp,
		store:     make(map[string]int64Store, len(metrics)),
	}
	for _, metric := range metrics {
		store.store[metric] = make(int64Store, len(batch.MetricSets))
	}
	for key, ms := range batch.MetricSets {
		for _, metric := range metrics {
			if metricValue, found := ms.MetricValues[metric]; found {
				metricstore := store.store[metric]
				metricstore[key] = metricValue.IntValue
			}
		}
	}
	return &store
}

func (this *MetricSink) Name() string {
	return "Metric Sink"
}

func (this *MetricSink) Stop() {
	// Do nothing.
}

func (this *MetricSink) ExportData(batch *core.DataBatch) {
	this.lock.Lock()
	defer this.lock.Unlock()

	now := time.Now()
	// TODO: add sorting
	this.longStore = append(popOldStore(this.longStore, now.Add(-this.longStoreDuration)),
		buildMultimetricStore(this.longStoreMetrics, batch))
	this.shortStore = append(popOld(this.shortStore, now.Add(-this.shortStoreDuration)), batch)
}

func (this *MetricSink) GetLatestDataBatch() *core.DataBatch {
	this.lock.Lock()
	defer this.lock.Unlock()

	if len(this.shortStore) == 0 {
		return nil
	}
	return this.shortStore[len(this.shortStore)-1]
}

func (this *MetricSink) GetShortStore() []*core.DataBatch {
	this.lock.Lock()
	defer this.lock.Unlock()

	result := make([]*core.DataBatch, 0, len(this.shortStore))
	for _, batch := range this.shortStore {
		result = append(result, batch)
	}
	return result
}

func (this *MetricSink) GetMetric(metricName string, keys []string, start, end time.Time) map[string][]core.TimestampedMetricValue {
	this.lock.Lock()
	defer this.lock.Unlock()

	useLongStore := false
	for _, longStoreMetric := range this.longStoreMetrics {
		if longStoreMetric == metricName {
			useLongStore = true
		}
	}

	result := make(map[string][]core.TimestampedMetricValue)
	if useLongStore {
		for _, store := range this.longStore {
			// Inclusive start and end.
			if !store.timestamp.Before(start) && !store.timestamp.After(end) {
				substore := store.store[metricName]
				for _, key := range keys {
					if val, found := substore[key]; found {
						result[key] = append(result[key], core.TimestampedMetricValue{
							Timestamp: store.timestamp,
							MetricValue: core.MetricValue{
								IntValue:   val,
								ValueType:  core.ValueInt64,
								MetricType: core.MetricGauge,
							},
						})
					}
				}
			}
		}
	} else {
		for _, batch := range this.shortStore {
			// Inclusive start and end.
			if !batch.Timestamp.Before(start) && !batch.Timestamp.After(end) {
				for _, key := range keys {
					metricSet, found := batch.MetricSets[key]
					if !found {
						continue
					}
					metricValue, found := metricSet.MetricValues[metricName]
					if !found {
						continue
					}
					keyResult, found := result[key]
					if !found {
						keyResult = make([]core.TimestampedMetricValue, 0)
					}
					keyResult = append(keyResult, core.TimestampedMetricValue{
						Timestamp:   batch.Timestamp,
						MetricValue: metricValue,
					})
					result[key] = keyResult
				}
			}
		}
	}
	return result
}

func (this *MetricSink) GetLabeledMetric(metricName string, labels map[string]string, keys []string, start, end time.Time) map[string][]core.TimestampedMetricValue {
	// NB: the long store doesn't store labeled metrics, so it's not relevant here
	result := make(map[string][]core.TimestampedMetricValue)
	for _, batch := range this.shortStore {
		// Inclusive start and end
		if !batch.Timestamp.Before(start) && !batch.Timestamp.After(end) {
			for _, key := range keys {
				metricSet, found := batch.MetricSets[key]
				if !found {
					continue
				}

				for _, labeledMetric := range metricSet.LabeledMetrics {
					if labeledMetric.Name != metricName {
						continue
					}

					if len(labeledMetric.Labels) != len(labels) {
						continue
					}

					labelsMatch := true
					for k, v := range labels {
						if lblMetricVal, ok := labeledMetric.Labels[k]; !ok || lblMetricVal != v {
							labelsMatch = false
							break
						}
					}

					if labelsMatch {
						result[key] = append(result[key], core.TimestampedMetricValue{
							Timestamp:   batch.Timestamp,
							MetricValue: labeledMetric.MetricValue,
						})
					}
				}
			}
		}
	}

	return result
}

func (this *MetricSink) GetMetricNames(key string) []string {
	this.lock.Lock()
	defer this.lock.Unlock()

	metricNames := make(map[string]bool)
	for _, batch := range this.shortStore {
		if set, found := batch.MetricSets[key]; found {
			for key := range set.MetricValues {
				metricNames[key] = true
			}
		}
	}
	result := make([]string, 0, len(metricNames))
	for key := range metricNames {
		result = append(result, key)
	}
	return result
}

func (this *MetricSink) getAllNames(predicate func(ms *core.MetricSet) bool,
	name func(key string, ms *core.MetricSet) string) []string {
	this.lock.Lock()
	defer this.lock.Unlock()

	if len(this.shortStore) == 0 {
		return []string{}
	}

	result := make([]string, 0, 0)
	for key, value := range this.shortStore[len(this.shortStore)-1].MetricSets {
		if predicate(value) {
			result = append(result, name(key, value))
		}
	}
	return result
}

/*
 * For debugging only.
 */
func (this *MetricSink) GetMetricSetKeys() []string {
	return this.getAllNames(
		func(ms *core.MetricSet) bool { return true },
		func(key string, ms *core.MetricSet) string { return key })
}

func (this *MetricSink) GetNodes() []string {
	return this.getAllNames(
		func(ms *core.MetricSet) bool { return ms.Labels[core.LabelMetricSetType.Key] == core.MetricSetTypeNode },
		func(key string, ms *core.MetricSet) string { return ms.Labels[core.LabelHostname.Key] })
}

func (this *MetricSink) GetPods() []string {
	return this.getAllNames(
		func(ms *core.MetricSet) bool { return ms.Labels[core.LabelMetricSetType.Key] == core.MetricSetTypePod },
		func(key string, ms *core.MetricSet) string {
			return ms.Labels[core.LabelNamespaceName.Key] + "/" + ms.Labels[core.LabelPodName.Key]
		})
}

func (this *MetricSink) GetNamespaces() []string {
	return this.getAllNames(
		func(ms *core.MetricSet) bool {
			return ms.Labels[core.LabelMetricSetType.Key] == core.MetricSetTypeNamespace
		},
		func(key string, ms *core.MetricSet) string { return ms.Labels[core.LabelNamespaceName.Key] })
}

func (this *MetricSink) GetPodsFromNamespace(namespace string) []string {
	return this.getAllNames(
		func(ms *core.MetricSet) bool {
			return ms.Labels[core.LabelMetricSetType.Key] == core.MetricSetTypePod &&
				ms.Labels[core.LabelNamespaceName.Key] == namespace
		},
		func(key string, ms *core.MetricSet) string {
			return ms.Labels[core.LabelPodName.Key]
		})
}

func (this *MetricSink) GetContainersForPodFromNamespace(namespace, pod string) []string {
	return this.getAllNames(
		func(ms *core.MetricSet) bool {
			return ms.Labels[core.LabelMetricSetType.Key] == core.MetricSetTypePodContainer &&
				ms.Labels[core.LabelNamespaceName.Key] == namespace &&
				ms.Labels[core.LabelPodName.Key] == pod
		},
		func(key string, ms *core.MetricSet) string {
			return ms.Labels[core.LabelContainerName.Key]
		})
}

func (this *MetricSink) GetSystemContainersFromNode(node string) []string {
	return this.getAllNames(
		func(ms *core.MetricSet) bool {
			return ms.Labels[core.LabelMetricSetType.Key] == core.MetricSetTypeSystemContainer &&
				ms.Labels[core.LabelHostname.Key] == node
		},
		func(key string, ms *core.MetricSet) string {
			return ms.Labels[core.LabelContainerName.Key]
		})
}

func popOld(storage []*core.DataBatch, cutoffTime time.Time) []*core.DataBatch {
	result := make([]*core.DataBatch, 0)
	for _, batch := range storage {
		if batch.Timestamp.After(cutoffTime) {
			result = append(result, batch)
		}
	}
	return result
}

func popOldStore(storages []*multimetricStore, cutoffTime time.Time) []*multimetricStore {
	result := make([]*multimetricStore, 0, len(storages))
	for _, store := range storages {
		if store.timestamp.After(cutoffTime) {
			result = append(result, store)
		}
	}
	return result
}

func NewMetricSink(shortStoreDuration, longStoreDuration time.Duration, longStoreMetrics []string) *MetricSink {
	return &MetricSink{
		longStoreMetrics:   longStoreMetrics,
		longStoreDuration:  longStoreDuration,
		shortStoreDuration: shortStoreDuration,
		longStore:          make([]*multimetricStore, 0),
		shortStore:         make([]*core.DataBatch, 0),
	}
}
