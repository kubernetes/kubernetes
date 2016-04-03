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
	"net/http"

	restful "github.com/emicklei/go-restful"
	"k8s.io/heapster/api/v1/types"
	"k8s.io/heapster/manager"
	sinksApi "k8s.io/heapster/sinks/api"
	"k8s.io/heapster/util"
)

type Api struct {
	manager             manager.Manager
	runningInKubernetes bool
}

// Create a new Api to serve from the specified cache.
func NewApi(m manager.Manager, runningInKuberentes bool) *Api {
	return &Api{
		manager:             m,
		runningInKubernetes: runningInKuberentes,
	}
}

// Register the mainApi on the specified endpoint.
func (a *Api) Register(container *restful.Container) {
	ws := new(restful.WebService)
	ws.
		Path("/api/v1/metric-export").
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
	ws = new(restful.WebService)
	ws.Path("/api/v1/sinks").
		Doc("Configuration for Heapster sinks for exporting data").
		Produces(restful.MIME_JSON)
	ws.Route(ws.POST("").
		To(a.setSinks).
		Doc("set the current sinks").
		Operation("setSinks").
		Reads([]string{}))
	ws.Route(ws.GET("").
		To(a.getSinks).
		Doc("get the current sinks").
		Operation("getSinks").
		Writes([]string{}))
	container.Add(ws)

	a.RegisterModel(container)
}

// Labels used by the target schema. A target schema uniquely identifies a container.
var targetLabelNames = map[string]struct{}{
	sinksApi.LabelPodId.Key:              {},
	sinksApi.LabelPodName.Key:            {},
	sinksApi.LabelPodNamespace.Key:       {},
	sinksApi.LabelContainerName.Key:      {},
	sinksApi.LabelLabels.Key:             {},
	sinksApi.LabelHostname.Key:           {},
	sinksApi.LabelHostID.Key:             {},
	sinksApi.LabelPodNamespaceUID.Key:    {},
	sinksApi.LabelContainerBaseImage.Key: {},
}

// Separates target schema labels from other labels.
func separateLabels(labels map[string]string) (map[string]string, map[string]string) {
	targetLabels := make(map[string]string, len(targetLabelNames))
	otherLabels := make(map[string]string, len(labels)-len(targetLabels))
	for label := range labels {
		// Ignore blank labels.
		if label == "" {
			continue
		}

		if _, ok := targetLabelNames[label]; ok {
			targetLabels[label] = labels[label]
		} else {
			otherLabels[label] = labels[label]
		}
	}

	return targetLabels, otherLabels
}

func (a *Api) exportMetricsSchema(request *restful.Request, response *restful.Response) {
	result := types.TimeseriesSchema{}
	for _, label := range sinksApi.CommonLabels() {
		result.CommonLabels = append(result.CommonLabels, types.LabelDescriptor{
			Key:         label.Key,
			Description: label.Description,
		})
	}
	for _, label := range sinksApi.PodLabels() {
		result.PodLabels = append(result.PodLabels, types.LabelDescriptor{
			Key:         label.Key,
			Description: label.Description,
		})
	}

	for _, metric := range sinksApi.SupportedStatMetrics() {
		md := types.MetricDescriptor{
			Name:        metric.Name,
			Description: metric.Description,
			Type:        metric.Type.String(),
			ValueType:   metric.ValueType.String(),
			Units:       metric.Units.String(),
		}
		for _, label := range metric.Labels {
			md.Labels = append(md.Labels, types.LabelDescriptor{
				Key:         label.Key,
				Description: label.Description,
			})
		}
		result.Metrics = append(result.Metrics, md)
	}
	response.WriteEntity(result)
}

func (a *Api) exportMetrics(request *restful.Request, response *restful.Response) {
	points, err := a.manager.ExportMetrics()
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}

	// Group points by target labels.
	timeseriesForTargetLabels := map[string]*types.Timeseries{}
	for _, point := range points {
		targetLabels, otherLabels := separateLabels(point.Labels)
		labelsStr := util.LabelsToString(targetLabels, ",")

		// Add timeseries if it does not exist.
		timeseries, ok := timeseriesForTargetLabels[labelsStr]
		if !ok {
			timeseries = &types.Timeseries{
				Metrics: map[string][]types.Point{},
				Labels:  targetLabels,
			}
			timeseriesForTargetLabels[labelsStr] = timeseries
		}

		// Add point to this timeseries
		timeseries.Metrics[point.Name] = append(timeseries.Metrics[point.Name], types.Point{
			Start:  point.Start,
			End:    point.End,
			Labels: otherLabels,
			Value:  point.Value,
		})
	}

	// Turn into a slice.
	timeseries := make([]*types.Timeseries, 0, len(timeseriesForTargetLabels))
	for _, val := range timeseriesForTargetLabels {
		timeseries = append(timeseries, val)
	}

	response.WriteEntity(timeseries)
}

func (a *Api) setSinks(req *restful.Request, resp *restful.Response) {
	sinkUris := new([]string)
	if err := req.ReadEntity(sinkUris); err != nil {
		resp.WriteError(http.StatusBadRequest, err)
		return
	}
	var uris manager.Uris
	for _, s := range *sinkUris {
		if err := uris.Set(s); err != nil {
			resp.WriteError(http.StatusBadRequest, err)
			return
		}
	}
	if err := a.manager.SetSinkUris(uris); err != nil {
		resp.WriteError(http.StatusInternalServerError, err)
		return
	}
}

func (a *Api) getSinks(req *restful.Request, resp *restful.Response) {
	sinkUris := a.manager.SinkUris()
	var strs []string
	if sinkUris == nil {
		strs = make([]string, 0)
	} else {
		strs = make([]string, 0, len(sinkUris))
		for _, u := range sinkUris {
			strs = append(strs, u.String())
		}
	}
	resp.WriteEntity(strs)
}
