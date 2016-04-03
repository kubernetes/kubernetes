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

package gcmautoscaling

import (
	"fmt"
	"net/url"
	"time"

	"github.com/golang/glog"
	"k8s.io/heapster/extpoints"
	"k8s.io/heapster/sinks/cache"
	"k8s.io/heapster/sinks/gcm"

	sink_api "k8s.io/heapster/sinks/api"
	kube_api "k8s.io/kubernetes/pkg/api"
)

var (
	LabelHostname = sink_api.LabelDescriptor{
		Key:         "hostname",
		Description: "Hostname where the container ran",
	}
	LabelGCEResourceID = sink_api.LabelDescriptor{
		Key:         "compute.googleapis.com/resource_id",
		Description: "Resource id for nodes specific for GCE.",
	}
	LabelGCEResourceType = sink_api.LabelDescriptor{
		Key:         "compute.googleapis.com/resource_type",
		Description: "Resource types for nodes specific for GCE.",
	}
	cpuUsage   = "cpu/usage"
	cpuLimit   = "cpu/limit"
	cpuRequest = "cpu/request"
	memUsage   = "memory/usage"
	memLimit   = "memory/limit"
	memRequest = "memory/request"
)

var autoscalingLabels = []sink_api.LabelDescriptor{
	LabelHostname,
	LabelGCEResourceID,
	LabelGCEResourceType,
}

type utilizationMetric struct {
	name        string
	description string
}

var autoscalingMetrics = map[string]utilizationMetric{
	cpuUsage: {
		name:        "cpu/node_utilization",
		description: "Cpu utilization as a share of node capacity",
	},
	cpuLimit: {
		name:        "cpu/node_reservation",
		description: "Share of cpu that is reserved on the node",
	},
	memUsage: {
		name:        "memory/node_utilization",
		description: "Memory utilization as a share of memory capacity",
	},
	memLimit: {
		name:        "memory/node_reservation",
		description: "Share of memory that is reserved on the node",
	},
}

// Since the input may contain data from different time windows we want to support it.
type hostTime struct {
	host string
	time time.Time
}

type gcmAutocalingSink struct {
	core       *gcm.GcmCore
	resolution time.Duration
	// For given hostname and time remembers its cpu capacity in milicores.
	cpuCapacity map[hostTime]int64
	// For given hostname and time remembers amount of reserved cpu in milicores.
	cpuReservation map[hostTime]int64
	// For given hostname and time remembers its memory capacity in bytes.
	memCapacity map[hostTime]int64
	// For given hostname and time remembers amount of reserved memory in bytes.
	memReservation map[hostTime]int64
}

func (self gcmAutocalingSink) hostTime(host string, metric *sink_api.Point) hostTime {
	return hostTime{host, metric.End.Truncate(self.resolution)}
}

// Adds the specified metrics or updates them if they already exist.
func (self gcmAutocalingSink) Register(_ []sink_api.MetricDescriptor) error {
	for _, metric := range autoscalingMetrics {
		if err := self.core.Register(metric.name, metric.description, sink_api.MetricGauge.String(), sink_api.ValueDouble.String(), autoscalingLabels); err != nil {
			return err
		}
	}
	return nil
}

func (self gcmAutocalingSink) Unregister(_ []sink_api.MetricDescriptor) error {
	for _, metric := range autoscalingMetrics {
		if err := self.core.Unregister(metric.name); err != nil {
			return err
		}
	}
	return nil
}

// Stores events into the backend.
func (self gcmAutocalingSink) StoreEvents([]kube_api.Event) error {
	// No-op, Google Cloud Monitoring doesn't store events
	return nil
}

func isNode(metric *sink_api.Point) bool {
	return metric.Labels[sink_api.LabelContainerName.Key] == cache.NodeContainerName
}

func isPodContainer(metric *sink_api.Point) bool {
	return len(metric.Labels[sink_api.LabelPodName.Key]) > 0
}

func (self *gcmAutocalingSink) updateMachineCapacityAndReservation(input []sink_api.Timeseries) {
	self.cpuCapacity = make(map[hostTime]int64)
	self.cpuReservation = make(map[hostTime]int64)
	self.memCapacity = make(map[hostTime]int64)
	self.memReservation = make(map[hostTime]int64)
	for _, entry := range input {
		metric := entry.Point
		if metric.Name != cpuLimit && metric.Name != memLimit && metric.Name != cpuRequest && metric.Name != memRequest {
			continue
		}
		host := metric.Labels[sink_api.LabelHostname.Key]
		value, ok := metric.Value.(int64)
		if !ok || value < 1 {
			continue
		}

		if isNode(metric) {
			if metric.Name == cpuLimit {
				self.cpuCapacity[self.hostTime(host, metric)] = value
			} else if metric.Name == memLimit {
				self.memCapacity[self.hostTime(host, metric)] = value
			}
		} else if isPodContainer(metric) {
			if metric.Name == cpuRequest {
				self.cpuReservation[self.hostTime(host, metric)] += value
			} else if metric.Name == memRequest {
				self.memReservation[self.hostTime(host, metric)] += value
			}
		}
	}
}

// For the given metric compute minimal set of labels required by autoscaling.
// See more: https://cloud.google.com/compute/docs/autoscaler/scaling-cloud-monitoring-metrics#custom_metrics_beta
func getLabels(metric *sink_api.Point) map[string]string {
	return map[string]string{
		gcm.FullLabelName(LabelHostname.Key):        metric.Labels[sink_api.LabelHostname.Key],
		gcm.FullLabelName(LabelGCEResourceID.Key):   metric.Labels[sink_api.LabelHostID.Key],
		gcm.FullLabelName(LabelGCEResourceType.Key): "instance",
	}
}

// For the given metric compute value of corresponding metric based on
// the original value and precomputed node stats.
func (self *gcmAutocalingSink) getNewValue(metric *sink_api.Point, ts *gcm.Timeseries) *float64 {
	host := metric.Labels[sink_api.LabelHostname.Key]

	var val float64
	switch metric.Name {
	case cpuUsage:
		capacity, ok := self.cpuCapacity[self.hostTime(host, metric)]
		if !ok || capacity < 1 || ts.Point.DoubleValue == nil {
			return nil
		}
		val = *ts.Point.DoubleValue / float64(capacity)
	case cpuLimit:
		reserved, ok := self.cpuReservation[self.hostTime(host, metric)]
		capacity, ok2 := self.cpuCapacity[self.hostTime(host, metric)]
		if !ok || !ok2 || capacity < 1 {
			return nil
		}
		val = float64(reserved) / float64(capacity)
	case memUsage:
		capacity, ok := self.memCapacity[self.hostTime(host, metric)]
		if !ok || capacity < 1 || ts.Point.Int64Value == nil {
			return nil
		}
		val = float64(*ts.Point.Int64Value) / float64(capacity)
	case memLimit:
		reserved, ok := self.memReservation[self.hostTime(host, metric)]
		capacity, ok2 := self.memCapacity[self.hostTime(host, metric)]
		if !ok || !ok2 || capacity < 1 {
			return nil
		}
		val = float64(reserved) / float64(capacity)
	default:
		return nil
	}
	return &val
}

// Pushes the specified metric values in input. The metrics must already exist.
func (self gcmAutocalingSink) StoreTimeseries(input []sink_api.Timeseries) error {
	self.updateMachineCapacityAndReservation(input)

	// Build a map of metrics by name.
	metrics := make(map[string][]gcm.Timeseries)
	for _, entry := range input {
		metric := entry.Point
		// We want to export only node metrics.
		if !isNode(metric) {
			continue
		}

		var ts *gcm.Timeseries
		var err error
		if metric.Name == cpuUsage {
			ts, err = self.core.GetEquivalentRateMetric(metric)
		} else if metric.Name == cpuLimit || metric.Name == memUsage || metric.Name == memLimit {
			ts, err = self.core.GetMetric(metric)
		} else {
			continue
		}
		if err != nil || ts == nil {
			glog.Errorf("Failed to create Timeseries for metric %v, host %v. Error %v.", autoscalingMetrics[metric.Name].name, metric.Labels[sink_api.LabelHostname.Key], err)
			continue
		}

		val := self.getNewValue(metric, ts)
		if val == nil {
			glog.Errorf("Failed to compute new value for metric %v, host %v.", autoscalingMetrics[metric.Name].name, metric.Labels[sink_api.LabelHostname.Key])
			continue
		}
		ts.Point.Int64Value = nil
		ts.Point.DoubleValue = val
		name := gcm.FullMetricName(autoscalingMetrics[metric.Name].name)
		ts.TimeseriesDescriptor.Metric = name
		ts.TimeseriesDescriptor.Labels = getLabels(metric)

		metrics[name] = append(metrics[name], *ts)
	}

	return self.core.StoreTimeseries(metrics)
}

func (self gcmAutocalingSink) DebugInfo() string {
	return "Sink Type: GCM Autoscaling"
}

func (self gcmAutocalingSink) Name() string {
	return "Google Cloud Monitoring Sink for Autoscaling"
}

func init() {
	extpoints.SinkFactories.Register(CreateGCMAutoscalingSink, "gcmautoscaling")
}

func CreateGCMAutoscalingSink(uri *url.URL, conf extpoints.HeapsterConf) ([]sink_api.ExternalSink, error) {
	if *uri != (url.URL{}) {
		return nil, fmt.Errorf("gcmautoscaling sinks don't take arguments")
	}
	core, err := gcm.NewCore()
	sink := gcmAutocalingSink{
		core:       core,
		resolution: conf.StatsResolution,
	}
	glog.Infof("created GCM Autocaling sink")
	return []sink_api.ExternalSink{sink}, err
}
