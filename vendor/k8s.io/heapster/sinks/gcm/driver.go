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

package gcm

import (
	"fmt"
	"net/url"

	kube_api "k8s.io/kubernetes/pkg/api"

	"github.com/golang/glog"
	"k8s.io/heapster/extpoints"
	sink_api "k8s.io/heapster/sinks/api"
)

type gcmSink struct {
	core *GcmCore
}

// Adds the specified metrics or updates them if they already exist.
func (self gcmSink) Register(metrics []sink_api.MetricDescriptor) error {
	for _, metric := range metrics {
		if err := self.core.Register(metric.Name, metric.Description, metric.Type.String(), metric.ValueType.String(), metric.Labels); err != nil {
			return err
		}
		if rateMetric, exists := gcmRateMetrics[metric.Name]; exists {
			if err := self.core.Register(rateMetric.name, rateMetric.description, sink_api.MetricGauge.String(), sink_api.ValueDouble.String(), metric.Labels); err != nil {
				return err
			}
		}
	}
	return nil
}

func (self gcmSink) Unregister(metrics []sink_api.MetricDescriptor) error {
	for _, metric := range metrics {
		if err := self.core.Unregister(metric.Name); err != nil {
			return err
		}
		if rateMetric, exists := gcmRateMetrics[metric.Name]; exists {
			if err := self.core.Unregister(rateMetric.name); err != nil {
				return err
			}
		}
	}
	return nil
}

// Stores events into the backend.
func (self gcmSink) StoreEvents([]kube_api.Event) error {
	// No-op, Google Cloud Monitoring doesn't store events
	return nil
}

// Pushes the specified metric values in input. The metrics must already exist.
func (self gcmSink) StoreTimeseries(input []sink_api.Timeseries) error {
	// Build a map of metrics by name.
	metrics := make(map[string][]Timeseries)
	for _, entry := range input {
		metric := entry.Point

		metricTimeseries, err := self.core.GetMetric(metric)
		if err != nil {
			return err
		}
		metrics[metric.Name] = append(metrics[metric.Name], *metricTimeseries)
		// TODO(vmarmol): Stop doing this when GCM supports graphing cumulative metrics.
		// Translate cumulative to gauge by taking the delta over the time period.
		rateMetricTimeseries, err := self.core.GetEquivalentRateMetric(metric)
		if err != nil {
			return err
		}
		if rateMetricTimeseries == nil {
			continue
		}
		rateMetricName := rateMetricTimeseries.TimeseriesDescriptor.Metric
		metrics[rateMetricName] = append(metrics[rateMetricName], *rateMetricTimeseries)
	}

	return self.core.StoreTimeseries(metrics)
}

func (self gcmSink) DebugInfo() string {
	return "Sink Type: GCM"
}

func (self gcmSink) Name() string {
	return "Google Cloud Monitoring Sink"
}

func init() {
	extpoints.SinkFactories.Register(CreateGCMSink, "gcm")
}

func CreateGCMSink(uri *url.URL, _ extpoints.HeapsterConf) ([]sink_api.ExternalSink, error) {
	if *uri != (url.URL{}) {
		return nil, fmt.Errorf("gcm sinks don't take arguments")
	}
	core, err := NewCore()
	sink := gcmSink{core: core}
	glog.Infof("created GCM sink")
	return []sink_api.ExternalSink{sink}, err
}
