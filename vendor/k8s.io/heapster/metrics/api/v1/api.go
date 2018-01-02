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
	"time"

	restful "github.com/emicklei/go-restful"

	"k8s.io/heapster/metrics/api/v1/types"
	"k8s.io/heapster/metrics/core"
	metricsink "k8s.io/heapster/metrics/sinks/metric"
)

type Api struct {
	runningInKubernetes bool
	metricSink          *metricsink.MetricSink
	historicalSource    core.HistoricalSource
	gkeMetrics          map[string]core.MetricDescriptor
	gkeLabels           map[string]core.LabelDescriptor
}

// Create a new Api to serve from the specified cache.
func NewApi(runningInKubernetes bool, metricSink *metricsink.MetricSink, historicalSource core.HistoricalSource) *Api {
	gkeMetrics := make(map[string]core.MetricDescriptor)
	gkeLabels := make(map[string]core.LabelDescriptor)
	for _, val := range core.StandardMetrics {
		gkeMetrics[val.Name] = val.MetricDescriptor
	}
	for _, val := range core.LabeledMetrics {
		gkeMetrics[val.Name] = val.MetricDescriptor
	}
	gkeMetrics[core.MetricCpuLimit.Name] = core.MetricCpuLimit.MetricDescriptor
	gkeMetrics[core.MetricMemoryLimit.Name] = core.MetricMemoryLimit.MetricDescriptor

	for _, val := range core.CommonLabels() {
		gkeLabels[val.Key] = val
	}
	for _, val := range core.ContainerLabels() {
		gkeLabels[val.Key] = val
	}
	for _, val := range core.PodLabels() {
		gkeLabels[val.Key] = val
	}

	return &Api{
		runningInKubernetes: runningInKubernetes,
		metricSink:          metricSink,
		historicalSource:    historicalSource,
		gkeMetrics:          gkeMetrics,
		gkeLabels:           gkeLabels,
	}
}

// Register the mainApi on the specified endpoint.
func (a *Api) Register(container *restful.Container) {
	ws := new(restful.WebService)
	ws.Path("/api/v1/metric-export").
		Doc("Exports the latest point for all Heapster metrics").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(a.exportMetrics).
		Doc("export the latest data point for all metrics").
		Operation("exportMetrics").
		Writes([]*types.Timeseries{}))
	container.Add(ws)
	ws = new(restful.WebService)
	ws.Path("/api/v1/metric-export-schema").
		Doc("Schema for metrics exported by heapster").
		Produces(restful.MIME_JSON)
	ws.Route(ws.GET("").
		To(a.exportMetricsSchema).
		Doc("export the schema for all metrics").
		Operation("exportmetricsSchema").
		Writes(types.TimeseriesSchema{}))
	container.Add(ws)

	if a.metricSink != nil {
		a.RegisterModel(container)
	}

	if a.historicalSource != nil {
		a.RegisterHistorical(container)
	}
}

func convertLabelDescriptor(ld core.LabelDescriptor) types.LabelDescriptor {
	return types.LabelDescriptor{
		Key:         ld.Key,
		Description: ld.Description,
	}
}

func convertMetricDescriptor(md core.MetricDescriptor) types.MetricDescriptor {
	result := types.MetricDescriptor{
		Name:        md.Name,
		Description: md.Description,
		Labels:      make([]types.LabelDescriptor, 0, len(md.Labels)),
	}
	for _, label := range md.Labels {
		result.Labels = append(result.Labels, convertLabelDescriptor(label))
	}

	switch md.Type {
	case core.MetricCumulative:
		result.Type = "cumulative"
	case core.MetricGauge:
		result.Type = "gauge"
	case core.MetricDelta:
		result.Type = "delta"
	}

	switch md.ValueType {
	case core.ValueInt64:
		result.ValueType = "int64"
	case core.ValueFloat:
		result.ValueType = "double"
	}

	switch md.Units {
	case core.UnitsBytes:
		result.Units = "bytes"
	case core.UnitsMilliseconds:
		result.Units = "ms"
	case core.UnitsNanoseconds:
		result.Units = "ns"
	case core.UnitsMillicores:
		result.Units = "millicores"
	}
	return result
}

func (a *Api) exportMetricsSchema(_ *restful.Request, response *restful.Response) {
	result := types.TimeseriesSchema{
		Metrics:      make([]types.MetricDescriptor, 0),
		CommonLabels: make([]types.LabelDescriptor, 0),
		PodLabels:    make([]types.LabelDescriptor, 0),
	}
	for _, metric := range core.StandardMetrics {
		if _, found := a.gkeMetrics[metric.Name]; found {
			result.Metrics = append(result.Metrics, convertMetricDescriptor(metric.MetricDescriptor))
		}
	}
	for _, metric := range core.AdditionalMetrics {
		if _, found := a.gkeMetrics[metric.Name]; found {
			result.Metrics = append(result.Metrics, convertMetricDescriptor(metric.MetricDescriptor))
		}
	}
	for _, metric := range core.LabeledMetrics {
		if _, found := a.gkeMetrics[metric.Name]; found {
			result.Metrics = append(result.Metrics, convertMetricDescriptor(metric.MetricDescriptor))
		}
	}

	for _, label := range core.CommonLabels() {
		if _, found := a.gkeLabels[label.Key]; found {
			result.CommonLabels = append(result.CommonLabels, convertLabelDescriptor(label))
		}
	}
	for _, label := range core.ContainerLabels() {
		if _, found := a.gkeLabels[label.Key]; found {
			result.CommonLabels = append(result.CommonLabels, convertLabelDescriptor(label))
		}
	}
	for _, label := range core.PodLabels() {
		if _, found := a.gkeLabels[label.Key]; found {
			result.PodLabels = append(result.PodLabels, convertLabelDescriptor(label))
		}
	}
	response.WriteEntity(result)
}

func (a *Api) exportMetrics(_ *restful.Request, response *restful.Response) {
	response.PrettyPrint(false)
	response.WriteEntity(a.processMetricsRequest(a.metricSink.GetShortStore()))
}

func (a *Api) processMetricsRequest(shortStorage []*core.DataBatch) []*types.Timeseries {
	tsmap := make(map[string]*types.Timeseries)

	var newestBatch *core.DataBatch
	for _, batch := range shortStorage {
		if newestBatch == nil || newestBatch.Timestamp.Before(batch.Timestamp) {
			newestBatch = batch
		}
	}

	var timeseries []*types.Timeseries
	if newestBatch == nil {
		return timeseries
	}
	for key, ms := range newestBatch.MetricSets {
		ts := tsmap[key]

		msType := ms.Labels[core.LabelMetricSetType.Key]

		switch msType {
		case core.MetricSetTypeNode, core.MetricSetTypePod, core.MetricSetTypePodContainer, core.MetricSetTypeSystemContainer:
		default:
			continue
		}

		if ts == nil {
			ts = &types.Timeseries{
				Metrics: make(map[string][]types.Point),
				Labels:  make(map[string]string),
			}
			for labelName, labelValue := range ms.Labels {
				if _, ok := a.gkeLabels[labelName]; ok {
					ts.Labels[labelName] = labelValue
				}
			}
			if msType == core.MetricSetTypeNode {
				ts.Labels[core.LabelContainerName.Key] = "machine"
			}
			if msType == core.MetricSetTypePod {
				ts.Labels[core.LabelContainerName.Key] = "/pod"
			}
			tsmap[key] = ts
		}
		for metricName, metricVal := range ms.MetricValues {
			if _, ok := a.gkeMetrics[metricName]; ok {
				processPoint(ts, newestBatch, metricName, &metricVal, nil, ms.CreateTime)
			}
		}
		for _, metric := range ms.LabeledMetrics {
			if _, ok := a.gkeMetrics[metric.Name]; ok {
				processPoint(ts, newestBatch, metric.Name, &metric.MetricValue, metric.Labels, ms.CreateTime)
			}
		}
	}
	timeseries = make([]*types.Timeseries, 0, len(tsmap))
	for _, ts := range tsmap {
		timeseries = append(timeseries, ts)
	}
	return timeseries
}

func processPoint(ts *types.Timeseries, db *core.DataBatch, metricName string, metricVal *core.MetricValue, labels map[string]string, creationTime time.Time) {
	points := ts.Metrics[metricName]
	if points == nil {
		points = make([]types.Point, 0, 1)
	}
	point := types.Point{
		Start: db.Timestamp,
		End:   db.Timestamp,
	}
	// For cumulative metric use the provided start time.
	if metricVal.MetricType == core.MetricCumulative {
		point.Start = creationTime
	}
	var value interface{}
	if metricVal.ValueType == core.ValueInt64 {
		value = metricVal.IntValue
	} else if metricVal.ValueType == core.ValueFloat {
		value = metricVal.FloatValue
	} else {
		return
	}
	point.Value = value
	if labels != nil {
		point.Labels = make(map[string]string)
		for key, value := range labels {
			point.Labels[key] = value
		}
	}
	points = append(points, point)
	ts.Metrics[metricName] = points
}
