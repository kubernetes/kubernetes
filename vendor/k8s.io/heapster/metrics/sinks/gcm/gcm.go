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
	"strings"
	"sync"
	"time"

	gce_util "k8s.io/heapster/common/gce"
	"k8s.io/heapster/metrics/core"

	"github.com/golang/glog"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	gcm "google.golang.org/api/cloudmonitoring/v2beta2"
	gce "google.golang.org/cloud/compute/metadata"
)

const (
	metricDomain    = "kubernetes.io"
	customApiPrefix = "custom.cloudmonitoring.googleapis.com"
	maxNumLabels    = 10
	// The largest number of timeseries we can write to per request.
	maxTimeseriesPerRequest = 200
)

type MetricFilter int8

const (
	metricsAll MetricFilter = iota
	metricsOnlyAutoscaling
)

type gcmSink struct {
	sync.RWMutex
	registered   bool
	project      string
	metricFilter MetricFilter
	gcmService   *gcm.Service
}

func (sink *gcmSink) Name() string {
	return "GCM Sink"
}

func getReq() *gcm.WriteTimeseriesRequest {
	return &gcm.WriteTimeseriesRequest{Timeseries: make([]*gcm.TimeseriesPoint, 0)}
}

func fullLabelName(name string) string {
	// handle correctly GCE specific labels
	if !strings.Contains(name, "compute.googleapis.com") {
		return fmt.Sprintf("%s/%s/label/%s", customApiPrefix, metricDomain, name)
	}
	return name
}

func fullMetricName(name string) string {
	return fmt.Sprintf("%s/%s/%s", customApiPrefix, metricDomain, name)
}

func (sink *gcmSink) getTimeseriesPoint(timestamp time.Time, labels map[string]string, metric string, val core.MetricValue, createTime time.Time) *gcm.TimeseriesPoint {
	point := &gcm.Point{
		Start: timestamp.Format(time.RFC3339),
		End:   timestamp.Format(time.RFC3339),
	}
	switch val.ValueType {
	case core.ValueInt64:
		point.Int64Value = &val.IntValue
	case core.ValueFloat:
		v := float64(val.FloatValue)
		point.DoubleValue = &v
	default:
		glog.Errorf("Type not supported %v in %v", val.ValueType, metric)
		return nil
	}
	// For cumulative metric use the provided start time.
	if val.MetricType == core.MetricCumulative {
		point.Start = createTime.Format(time.RFC3339)
	}

	finalLabels := make(map[string]string)
	if core.IsNodeAutoscalingMetric(metric) {
		// All and autoscaling. Do not populate for other filters.
		if sink.metricFilter != metricsAll &&
			sink.metricFilter != metricsOnlyAutoscaling {
			return nil
		}

		finalLabels[fullLabelName(core.LabelHostname.Key)] = labels[core.LabelHostname.Key]
		finalLabels[fullLabelName(core.LabelGCEResourceID.Key)] = labels[core.LabelHostID.Key]
		finalLabels[fullLabelName(core.LabelGCEResourceType.Key)] = "instance"
	} else {
		// Only all.
		if sink.metricFilter != metricsAll {
			return nil
		}
		supportedLables := core.GcmLabels()
		for key, value := range labels {
			if _, ok := supportedLables[key]; ok {
				finalLabels[fullLabelName(key)] = value
			}
		}
	}
	desc := &gcm.TimeseriesDescriptor{
		Project: sink.project,
		Labels:  finalLabels,
		Metric:  fullMetricName(metric),
	}

	return &gcm.TimeseriesPoint{Point: point, TimeseriesDesc: desc}
}

func (sink *gcmSink) getTimeseriesPointForLabeledMetrics(timestamp time.Time, labels map[string]string, metric core.LabeledMetric, createTime time.Time) *gcm.TimeseriesPoint {
	// Only all. There are no atuoscaling labeled metrics.
	if sink.metricFilter != metricsAll {
		return nil
	}

	point := &gcm.Point{
		Start: timestamp.Format(time.RFC3339),
		End:   timestamp.Format(time.RFC3339),
	}
	switch metric.ValueType {
	case core.ValueInt64:
		point.Int64Value = &metric.IntValue
	case core.ValueFloat:
		v := float64(metric.FloatValue)
		point.DoubleValue = &v
	default:
		glog.Errorf("Type not supported %v in %v", metric.ValueType, metric)
		return nil
	}
	// For cumulative metric use the provided start time.
	if metric.MetricType == core.MetricCumulative {
		point.Start = createTime.Format(time.RFC3339)
	}

	finalLabels := make(map[string]string)
	supportedLables := core.GcmLabels()
	for key, value := range labels {
		if _, ok := supportedLables[key]; ok {
			finalLabels[fullLabelName(key)] = value
		}
	}
	for key, value := range metric.Labels {
		if _, ok := supportedLables[key]; ok {
			finalLabels[fullLabelName(key)] = value
		}
	}

	desc := &gcm.TimeseriesDescriptor{
		Project: sink.project,
		Labels:  finalLabels,
		Metric:  fullMetricName(metric.Name),
	}

	return &gcm.TimeseriesPoint{Point: point, TimeseriesDesc: desc}
}

func (sink *gcmSink) sendRequest(req *gcm.WriteTimeseriesRequest) {
	_, err := sink.gcmService.Timeseries.Write(sink.project, req).Do()
	if err != nil {
		glog.Errorf("Error while sending request to GCM %v", err)
	} else {
		glog.V(4).Infof("Successfully sent %v timeserieses to GCM", len(req.Timeseries))
	}
}

func (sink *gcmSink) ExportData(dataBatch *core.DataBatch) {
	if err := sink.registerAllMetrics(); err != nil {
		glog.Warningf("Error during metrics registration: %v", err)
		return
	}

	req := getReq()
	for _, metricSet := range dataBatch.MetricSets {
		for metric, val := range metricSet.MetricValues {
			point := sink.getTimeseriesPoint(dataBatch.Timestamp, metricSet.Labels, metric, val, metricSet.CreateTime)
			if point != nil {
				req.Timeseries = append(req.Timeseries, point)
			}
			if len(req.Timeseries) >= maxTimeseriesPerRequest {
				sink.sendRequest(req)
				req = getReq()
			}
		}
		for _, metric := range metricSet.LabeledMetrics {
			point := sink.getTimeseriesPointForLabeledMetrics(dataBatch.Timestamp, metricSet.Labels, metric, metricSet.CreateTime)
			if point != nil {
				req.Timeseries = append(req.Timeseries, point)
			}
			if len(req.Timeseries) >= maxTimeseriesPerRequest {
				sink.sendRequest(req)
				req = getReq()
			}
		}
	}
	if len(req.Timeseries) > 0 {
		sink.sendRequest(req)
	}
}

func (sink *gcmSink) Stop() {
	// nothing needs to be done.
}

func (sink *gcmSink) registerAllMetrics() error {
	return sink.register(core.AllMetrics)
}

// Adds the specified metrics or updates them if they already exist.
func (sink *gcmSink) register(metrics []core.Metric) error {
	sink.Lock()
	defer sink.Unlock()
	if sink.registered {
		return nil
	}

	for _, metric := range metrics {
		metricName := fullMetricName(metric.MetricDescriptor.Name)
		if _, err := sink.gcmService.MetricDescriptors.Delete(sink.project, metricName).Do(); err != nil {
			glog.Infof("[GCM] Deleting metric %v failed: %v", metricName, err)
		}
		labels := make([]*gcm.MetricDescriptorLabelDescriptor, 0)

		// Node autoscaling metrics have special labels.
		if core.IsNodeAutoscalingMetric(metric.MetricDescriptor.Name) {
			// All and autoscaling. Do not populate for other filters.
			if sink.metricFilter != metricsAll &&
				sink.metricFilter != metricsOnlyAutoscaling {
				continue
			}

			for _, l := range core.GcmNodeAutoscalingLabels() {
				labels = append(labels, &gcm.MetricDescriptorLabelDescriptor{
					Key:         fullLabelName(l.Key),
					Description: l.Description,
				})
			}
		} else {
			// Only all.
			if sink.metricFilter != metricsAll {
				continue
			}

			for _, l := range core.GcmLabels() {
				labels = append(labels, &gcm.MetricDescriptorLabelDescriptor{
					Key:         fullLabelName(l.Key),
					Description: l.Description,
				})
			}
		}

		t := &gcm.MetricDescriptorTypeDescriptor{
			MetricType: metric.MetricDescriptor.Type.String(),
			ValueType:  metric.MetricDescriptor.ValueType.String(),
		}
		desc := &gcm.MetricDescriptor{
			Name:           metricName,
			Project:        sink.project,
			Description:    metric.MetricDescriptor.Description,
			Labels:         labels,
			TypeDescriptor: t,
		}
		if _, err := sink.gcmService.MetricDescriptors.Create(sink.project, desc).Do(); err != nil {
			return err
		}
	}
	sink.registered = true
	return nil
}

func CreateGCMSink(uri *url.URL) (core.DataSink, error) {
	if len(uri.Scheme) > 0 {
		return nil, fmt.Errorf("scheme should not be set for GCM sink")
	}
	if len(uri.Host) > 0 {
		return nil, fmt.Errorf("host should not be set for GCM sink")
	}

	opts, err := url.ParseQuery(uri.RawQuery)

	metrics := "all"
	if len(opts["metrics"]) > 0 {
		metrics = opts["metrics"][0]
	}
	var metricFilter MetricFilter = metricsAll
	switch metrics {
	case "all":
		metricFilter = metricsAll
	case "autoscaling":
		metricFilter = metricsOnlyAutoscaling
	default:
		return nil, fmt.Errorf("invalid metrics parameter: %s", metrics)
	}

	if err := gce_util.EnsureOnGCE(); err != nil {
		return nil, err
	}

	// Detect project ID
	projectId, err := gce.ProjectID()
	if err != nil {
		return nil, err
	}

	// Create Google Cloud Monitoring service.
	client := oauth2.NewClient(oauth2.NoContext, google.ComputeTokenSource(""))
	gcmService, err := gcm.New(client)
	if err != nil {
		return nil, err
	}

	sink := &gcmSink{
		registered:   false,
		project:      projectId,
		gcmService:   gcmService,
		metricFilter: metricFilter,
	}
	glog.Infof("created GCM sink")
	if err := sink.registerAllMetrics(); err != nil {
		glog.Warningf("Error during metrics registration: %v", err)
	}
	return sink, nil
}
