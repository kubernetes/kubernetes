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

package v1

import (
	"fmt"
	"time"

	"k8s.io/heapster/sinks/cache"
	"k8s.io/heapster/util"
)

type Decoder interface {
	// Timeseries returns the metrics found in input as a timeseries slice.
	TimeseriesFromPods([]*cache.PodElement) ([]Timeseries, error)
	TimeseriesFromContainers([]*cache.ContainerElement) ([]Timeseries, error)
}

type timeseriesKey struct {
	// Name of the metric.
	Name string

	// Mangled labels on the metric.
	Labels string
}

type decoder struct {
	supportedStatMetrics []SupportedStatMetric
	// TODO: Garbage collect data.
	// TODO: Deprecate this once we the core is fixed to never export duplicate stats.
	lastExported map[timeseriesKey]time.Time
}

func (self *decoder) TimeseriesFromPods(pods []*cache.PodElement) ([]Timeseries, error) {
	var result []Timeseries
	// Format metrics and push them.
	for index := range pods {
		result = append(result, self.getPodMetrics(pods[index])...)
	}
	return result, nil
}
func (self *decoder) TimeseriesFromContainers(containers []*cache.ContainerElement) ([]Timeseries, error) {
	labels := make(map[string]string)
	var result []Timeseries
	for index := range containers {
		labels[LabelHostname.Key] = containers[index].Hostname
		result = append(result, self.getContainerMetrics(containers[index], util.CopyLabels(labels))...)
	}
	return result, nil
}

// Generate the labels.
func (self *decoder) getPodLabels(pod *cache.PodElement) map[string]string {
	labels := make(map[string]string)
	labels[LabelPodId.Key] = pod.UID
	labels[LabelPodNamespace.Key] = pod.Namespace
	labels[LabelPodNamespaceUID.Key] = pod.NamespaceUID
	labels[LabelPodName.Key] = pod.Name
	labels[LabelLabels.Key] = util.LabelsToString(pod.Labels, ",")
	labels[LabelHostname.Key] = pod.Hostname
	labels[LabelHostID.Key] = pod.ExternalID

	return labels
}

func (self *decoder) getPodMetrics(pod *cache.PodElement) []Timeseries {
	// Break the individual metrics from the container statistics.
	result := []Timeseries{}
	if pod == nil || pod.Containers == nil {
		return result
	}
	for index := range pod.Containers {
		timeseries := self.getContainerMetrics(pod.Containers[index], self.getPodLabels(pod))
		result = append(result, timeseries...)
	}

	return result
}

func (self *decoder) getContainerMetrics(container *cache.ContainerElement, labels map[string]string) []Timeseries {
	if container == nil {
		return nil
	}
	labels[LabelContainerName.Key] = container.Name
	labels[LabelContainerBaseImage.Key] = container.Image
	// Add container specific labels along with existing labels.
	containerLabels := util.LabelsToString(container.Labels, ",")
	if labels[LabelLabels.Key] != "" {
		containerLabels = fmt.Sprintf("%s,%s", labels[LabelLabels.Key], containerLabels)
	}
	labels[LabelLabels.Key] = containerLabels

	if _, exists := labels[LabelHostID.Key]; !exists {
		labels[LabelHostID.Key] = container.ExternalID
	}
	// One metric value per data point.
	var result []Timeseries
	labelsAsString := util.LabelsToString(labels, ",")

	// Metrics are in reverse chronological order (most recent to oldest). See sinks/cache/cache.go.
	// Iterate over them in chronological order since the code below assumes such order.
	// TODO(piosz): Remove this hack.
	for i := len(container.Metrics) - 1; i >= 0; i-- {
		metric := container.Metrics[i]
		if metric == nil || metric.Spec == nil || metric.Stats == nil {
			continue
		}
		// Add all supported metrics that have values.
		for index, supported := range self.supportedStatMetrics {
			// Finest allowed granularity is seconds.
			metric.Stats.Timestamp = metric.Stats.Timestamp.Round(time.Second)
			key := timeseriesKey{
				Name:   supported.Name,
				Labels: labelsAsString,
			}

			// TODO: remove this once the heapster source is tested to not provide duplicate metric.Stats.
			if data, ok := self.lastExported[key]; ok && !data.Before(metric.Stats.Timestamp) {
				continue
			}

			if supported.HasValue(metric.Spec) {
				// Cumulative metric.Statss have container creation time as their start time.
				var startTime time.Time
				if supported.Type == MetricCumulative {
					startTime = metric.Spec.CreationTime
				} else {
					startTime = metric.Stats.Timestamp
				}
				points := supported.GetValue(metric.Spec, metric.Stats)
				for _, point := range points {
					labels := util.CopyLabels(labels)
					for name, value := range point.Labels {
						labels[name] = value
					}
					timeseries := Timeseries{
						MetricDescriptor: &self.supportedStatMetrics[index].MetricDescriptor,
						Point: &Point{
							Name:   supported.Name,
							Labels: labels,
							Start:  startTime.Round(time.Second),
							End:    metric.Stats.Timestamp,
							Value:  point.Value,
						},
					}
					result = append(result, timeseries)
				}
			}
			self.lastExported[key] = metric.Stats.Timestamp
		}

	}
	return result
}

func NewDecoder() Decoder {
	// Get supported metrics.
	return &decoder{
		supportedStatMetrics: statMetrics,
		lastExported:         make(map[timeseriesKey]time.Time),
	}
}
