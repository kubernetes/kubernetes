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
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	restful "github.com/emicklei/go-restful"

	"k8s.io/heapster/metrics/api/v1/types"
	"k8s.io/heapster/metrics/core"
	"k8s.io/heapster/metrics/util/metrics"
)

// for testing
var nowFunc = time.Now

// errModelNotActivated is the error that is returned by the API handlers
// when manager.model has not been initialized.
var errModelNotActivated = errors.New("the model is not activated")

// Deprecated - clients should switch to full metric names ASAP.
var deprecatedMetricNamesConversion = map[string]string{
	"cpu-usage":      "cpu/usage_rate",
	"cpu-limit":      "cpu/limit",
	"memory-limit":   "memory/limit",
	"memory-usage":   "memory/usage",
	"memory-working": "memory/working_set",
}

type clusterMetricsFetcher interface {
	availableClusterMetrics(request *restful.Request, response *restful.Response)
	clusterMetrics(request *restful.Request, response *restful.Response)

	nodeList(request *restful.Request, response *restful.Response)
	availableNodeMetrics(request *restful.Request, response *restful.Response)
	nodeMetrics(request *restful.Request, response *restful.Response)

	namespaceList(request *restful.Request, response *restful.Response)
	availableNamespaceMetrics(request *restful.Request, response *restful.Response)
	namespaceMetrics(request *restful.Request, response *restful.Response)

	namespacePodList(request *restful.Request, response *restful.Response)
	availablePodMetrics(request *restful.Request, response *restful.Response)
	podMetrics(request *restful.Request, response *restful.Response)

	availablePodContainerMetrics(request *restful.Request, response *restful.Response)
	podContainerMetrics(request *restful.Request, response *restful.Response)

	nodeSystemContainerList(request *restful.Request, response *restful.Response)
	availableFreeContainerMetrics(request *restful.Request, response *restful.Response)
	freeContainerMetrics(request *restful.Request, response *restful.Response)

	podListMetrics(request *restful.Request, response *restful.Response)

	isRunningInKubernetes() bool
}

// addClusterMetricsRoutes adds all the standard model routes to a WebService.
// It should already have a base path registered.
func addClusterMetricsRoutes(a clusterMetricsFetcher, ws *restful.WebService) {
	// The /metrics/ endpoint returns a list of all available metrics for the Cluster entity of the model.
	ws.Route(ws.GET("/metrics/").
		To(metrics.InstrumentRouteFunc("availableClusterMetrics", a.availableClusterMetrics)).
		Doc("Get a list of all available metrics for the Cluster entity").
		Operation("availableClusterMetrics"))

	// The /metrics/{metric-name} endpoint exposes an aggregated metric for the Cluster entity of the model.
	ws.Route(ws.GET("/metrics/{metric-name:*}").
		To(metrics.InstrumentRouteFunc("clusterMetrics", a.clusterMetrics)).
		Doc("Export an aggregated cluster-level metric").
		Operation("clusterMetrics").
		Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
		Param(ws.QueryParameter("start", "Start time for requested metric").DataType("string")).
		Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
		Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
		Writes(types.MetricResult{}))

	// The /nodes/{node-name}/metrics endpoint returns a list of all nodes with some metrics.
	ws.Route(ws.GET("/nodes/").
		To(metrics.InstrumentRouteFunc("nodeList", a.nodeList)).
		Doc("Get a list of all nodes that have some current metrics").
		Operation("nodeList"))

	// The /nodes/{node-name}/metrics endpoint returns a list of all available metrics for a Node entity.
	ws.Route(ws.GET("/nodes/{node-name}/metrics/").
		To(metrics.InstrumentRouteFunc("availableNodeMetrics", a.availableNodeMetrics)).
		Doc("Get a list of all available metrics for a Node entity").
		Operation("availableNodeMetrics").
		Param(ws.PathParameter("node-name", "The name of the node to lookup").DataType("string")))

	// The /nodes/{node-name}/metrics/{metric-name} endpoint exposes a metric for a Node entity of the model.
	// The {node-name} parameter is the hostname of a specific node.
	ws.Route(ws.GET("/nodes/{node-name}/metrics/{metric-name:*}").
		To(metrics.InstrumentRouteFunc("nodeMetrics", a.nodeMetrics)).
		Doc("Export a node-level metric").
		Operation("nodeMetrics").
		Param(ws.PathParameter("node-name", "The name of the node to lookup").DataType("string")).
		Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
		Param(ws.QueryParameter("start", "Start time for requested metric").DataType("string")).
		Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
		Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
		Writes(types.MetricResult{}))

	if a.isRunningInKubernetes() {

		ws.Route(ws.GET("/namespaces/").
			To(metrics.InstrumentRouteFunc("namespaceList", a.namespaceList)).
			Doc("Get a list of all namespaces that have some current metrics").
			Operation("namespaceList"))

		// The /namespaces/{namespace-name}/metrics endpoint returns a list of all available metrics for a Namespace entity.
		ws.Route(ws.GET("/namespaces/{namespace-name}/metrics").
			To(metrics.InstrumentRouteFunc("availableNamespaceMetrics", a.availableNamespaceMetrics)).
			Doc("Get a list of all available metrics for a Namespace entity").
			Operation("availableNamespaceMetrics").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")))

		// The /namespaces/{namespace-name}/metrics/{metric-name} endpoint exposes an aggregated metrics
		// for a Namespace entity of the model.
		ws.Route(ws.GET("/namespaces/{namespace-name}/metrics/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("namespaceMetrics", a.namespaceMetrics)).
			Doc("Export an aggregated namespace-level metric").
			Operation("namespaceMetrics").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricResult{}))

		ws.Route(ws.GET("/namespaces/{namespace-name}/pods/").
			To(metrics.InstrumentRouteFunc("namespacePodList", a.namespacePodList)).
			Doc("Get a list of pods from the given namespace that have some metrics").
			Operation("namespacePodList").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")))

		// The /namespaces/{namespace-name}/pods/{pod-name}/metrics endpoint returns a list of all available metrics for a Pod entity.
		ws.Route(ws.GET("/namespaces/{namespace-name}/pods/{pod-name}/metrics").
			To(metrics.InstrumentRouteFunc("availablePodMetrics", a.availablePodMetrics)).
			Doc("Get a list of all available metrics for a Pod entity").
			Operation("availablePodMetrics").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")).
			Param(ws.PathParameter("pod-name", "The name of the pod to lookup").DataType("string")))

		// The /namespaces/{namespace-name}/pods/{pod-name}/metrics/{metric-name} endpoint exposes
		// an aggregated metric for a Pod entity of the model.
		ws.Route(ws.GET("/namespaces/{namespace-name}/pods/{pod-name}/metrics/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podMetrics", a.podMetrics)).
			Doc("Export an aggregated pod-level metric").
			Operation("podMetrics").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")).
			Param(ws.PathParameter("pod-name", "The name of the pod to lookup").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricResult{}))

		// The /namespaces/{namespace-name}/pods/{pod-name}/containers/metrics/{container-name}/metrics endpoint
		// returns a list of all available metrics for a Pod Container entity.
		ws.Route(ws.GET("/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics").
			To(metrics.InstrumentRouteFunc("availableContainerMetrics", a.availablePodContainerMetrics)).
			Doc("Get a list of all available metrics for a Pod entity").
			Operation("availableContainerMetrics").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")).
			Param(ws.PathParameter("pod-name", "The name of the pod to lookup").DataType("string")).
			Param(ws.PathParameter("container-name", "The name of the namespace to use").DataType("string")))

		// The /namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics/{metric-name} endpoint exposes
		// a metric for a Container entity of the model.
		ws.Route(ws.GET("/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podContainerMetrics", a.podContainerMetrics)).
			Doc("Export an aggregated metric for a Pod Container").
			Operation("podContainerMetrics").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to use").DataType("string")).
			Param(ws.PathParameter("pod-name", "The name of the pod to use").DataType("string")).
			Param(ws.PathParameter("container-name", "The name of the namespace to use").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricResult{}))
	}

	ws.Route(ws.GET("/nodes/{node-name}/freecontainers/").
		To(metrics.InstrumentRouteFunc("systemContainerList", a.nodeSystemContainerList)).
		Doc("Get a list of all non-pod containers with some metrics").
		Operation("systemContainerList").
		Param(ws.PathParameter("node-name", "The name of the namespace to lookup").DataType("string")))

	// The /nodes/{node-name}/freecontainers/{container-name}/metrics endpoint
	// returns a list of all available metrics for a Free Container entity.
	ws.Route(ws.GET("/nodes/{node-name}/freecontainers/{container-name}/metrics").
		To(metrics.InstrumentRouteFunc("availableMetrics", a.availableFreeContainerMetrics)).
		Doc("Get a list of all available metrics for a free Container entity").
		Operation("availableMetrics").
		Param(ws.PathParameter("node-name", "The name of the namespace to lookup").DataType("string")).
		Param(ws.PathParameter("container-name", "The name of the namespace to use").DataType("string")))

	// The /nodes/{node-name}/freecontainers/{container-name}/metrics/{metric-name} endpoint exposes
	// a metric for a free Container entity of the model.
	ws.Route(ws.GET("/nodes/{node-name}/freecontainers/{container-name}/metrics/{metric-name:*}").
		To(metrics.InstrumentRouteFunc("freeContainerMetrics", a.freeContainerMetrics)).
		Doc("Export a container-level metric for a free container").
		Operation("freeContainerMetrics").
		Param(ws.PathParameter("node-name", "The name of the node to use").DataType("string")).
		Param(ws.PathParameter("container-name", "The name of the container to use").DataType("string")).
		Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
		Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
		Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
		Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
		Writes(types.MetricResult{}))

	if a.isRunningInKubernetes() {
		// The /namespaces/{namespace-name}/pod-list/{pod-list}/metrics/{metric-name} endpoint exposes
		// metrics for a list od pods of the model.
		ws.Route(ws.GET("/namespaces/{namespace-name}/pod-list/{pod-list}/metrics/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podListMetric", a.podListMetrics)).
			Doc("Export a metric for all pods from the given list").
			Operation("podListMetric").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")).
			Param(ws.PathParameter("pod-list", "Comma separated list of pod names to lookup").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricResult{}))
	}
}

func (a *Api) isRunningInKubernetes() bool {
	return a.runningInKubernetes
}

// RegisterModel registers the Model API endpoints.
// All endpoints that end with a {metric-name} also receive a start time query parameter.
// The start and end times should be specified as a string, formatted according to RFC 3339.
func (a *Api) RegisterModel(container *restful.Container) {
	ws := new(restful.WebService)
	ws.Path("/api/v1/model").
		Doc("Root endpoint of the stats model").
		Consumes("*/*").
		Produces(restful.MIME_JSON)

	addClusterMetricsRoutes(a, ws)

	ws.Route(ws.GET("/debug/allkeys").
		To(metrics.InstrumentRouteFunc("debugAllKeys", a.allKeys)).
		Doc("Get keys of all metric sets available").
		Operation("debugAllKeys"))
	container.Add(ws)
}

// availableMetrics returns a list of available cluster metric names.
func (a *Api) availableClusterMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricNamesRequest(core.ClusterKey(), response)
}

// availableMetrics returns a list of available node metric names.
func (a *Api) availableNodeMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricNamesRequest(core.NodeKey(request.PathParameter("node-name")), response)
}

// availableMetrics returns a list of available namespace metric names.
func (a *Api) availableNamespaceMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricNamesRequest(core.NamespaceKey(request.PathParameter("namespace-name")), response)
}

// availableMetrics returns a list of available pod metric names.
func (a *Api) availablePodMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricNamesRequest(
		core.PodKey(request.PathParameter("namespace-name"),
			request.PathParameter("pod-name")), response)
}

// availableMetrics returns a list of available pod metric names.
func (a *Api) availablePodContainerMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricNamesRequest(
		core.PodContainerKey(request.PathParameter("namespace-name"),
			request.PathParameter("pod-name"),
			request.PathParameter("container-name"),
		), response)
}

// availableMetrics returns a list of available pod metric names.
func (a *Api) availableFreeContainerMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricNamesRequest(
		core.NodeContainerKey(request.PathParameter("node-name"),
			request.PathParameter("container-name"),
		), response)
}

func (a *Api) nodeList(request *restful.Request, response *restful.Response) {
	response.WriteEntity(a.metricSink.GetNodes())
}

func (a *Api) namespaceList(request *restful.Request, response *restful.Response) {
	response.WriteEntity(a.metricSink.GetNamespaces())
}

func (a *Api) namespacePodList(request *restful.Request, response *restful.Response) {
	response.WriteEntity(a.metricSink.GetPodsFromNamespace(request.PathParameter("namespace-name")))
}

func (a *Api) nodeSystemContainerList(request *restful.Request, response *restful.Response) {
	response.WriteEntity(a.metricSink.GetSystemContainersFromNode(request.PathParameter("node-name")))
}

func (a *Api) allKeys(request *restful.Request, response *restful.Response) {
	response.WriteEntity(a.metricSink.GetMetricSetKeys())
}

// clusterMetrics returns a metric timeseries for a metric of the Cluster entity.
func (a *Api) clusterMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricRequest(core.ClusterKey(), request, response)
}

// nodeMetrics returns a metric timeseries for a metric of the Node entity.
func (a *Api) nodeMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricRequest(core.NodeKey(request.PathParameter("node-name")),
		request, response)
}

// namespaceMetrics returns a metric timeseries for a metric of the Namespace entity.
func (a *Api) namespaceMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricRequest(core.NamespaceKey(request.PathParameter("namespace-name")),
		request, response)
}

// podMetrics returns a metric timeseries for a metric of the Pod entity.
func (a *Api) podMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricRequest(
		core.PodKey(request.PathParameter("namespace-name"),
			request.PathParameter("pod-name")),
		request, response)
}

func (a *Api) podListMetrics(request *restful.Request, response *restful.Response) {
	start, end, err := getStartEndTime(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	ns := request.PathParameter("namespace-name")
	keys := []string{}
	metricName := request.PathParameter("metric-name")
	convertedMetricName := convertMetricName(metricName)
	for _, podName := range strings.Split(request.PathParameter("pod-list"), ",") {
		keys = append(keys, core.PodKey(ns, podName))
	}

	labels, err := getLabels(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}

	var metrics map[string][]core.TimestampedMetricValue
	if labels != nil {
		metrics = a.metricSink.GetLabeledMetric(convertedMetricName, labels, keys, start, end)
	} else {
		metrics = a.metricSink.GetMetric(convertedMetricName, keys, start, end)
	}

	result := types.MetricResultList{
		Items: make([]types.MetricResult, 0, len(keys)),
	}
	for _, key := range keys {
		result.Items = append(result.Items, exportTimestampedMetricValue(metrics[key]))
	}
	response.PrettyPrint(false)
	response.WriteEntity(result)
}

// podContainerMetrics returns a metric timeseries for a metric of a Pod Container entity.
// podContainerMetrics uses the namespace-name/pod-name/container-name path.
func (a *Api) podContainerMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricRequest(
		core.PodContainerKey(request.PathParameter("namespace-name"),
			request.PathParameter("pod-name"),
			request.PathParameter("container-name"),
		),
		request, response)
}

// freeContainerMetrics returns a metric timeseries for a metric of the Container entity.
// freeContainerMetrics addresses only free containers, by using the node-name/container-name path.
func (a *Api) freeContainerMetrics(request *restful.Request, response *restful.Response) {
	a.processMetricRequest(
		core.NodeContainerKey(request.PathParameter("node-name"),
			request.PathParameter("container-name"),
		),
		request, response)
}

// parseRequestParam parses a time.Time from a named QueryParam, using the RFC3339 format.
func parseTimeParam(queryParam string, defaultValue time.Time) (time.Time, error) {
	if queryParam != "" {
		reqStamp, err := time.Parse(time.RFC3339, queryParam)
		if err != nil {
			return time.Time{}, fmt.Errorf("timestamp argument cannot be parsed: %s", err)
		}
		return reqStamp, nil
	}
	return defaultValue, nil
}

func (a *Api) processMetricRequest(key string, request *restful.Request, response *restful.Response) {
	start, end, err := getStartEndTime(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	metricName := request.PathParameter("metric-name")
	convertedMetricName := convertMetricName(metricName)
	labels, err := getLabels(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}

	var metrics map[string][]core.TimestampedMetricValue
	if labels != nil {
		metrics = a.metricSink.GetLabeledMetric(convertedMetricName, labels, []string{key}, start, end)
	} else {
		metrics = a.metricSink.GetMetric(convertedMetricName, []string{key}, start, end)
	}
	converted := exportTimestampedMetricValue(metrics[key])
	response.WriteEntity(converted)
}

func (a *Api) processMetricNamesRequest(key string, response *restful.Response) {
	metricNames := a.metricSink.GetMetricNames(key)
	response.WriteEntity(metricNames)
}

func convertMetricName(metricName string) string {
	if convertedMetricName, ok := deprecatedMetricNamesConversion[metricName]; ok {
		return convertedMetricName
	}
	return metricName
}

func getStartEndTime(request *restful.Request) (time.Time, time.Time, error) {
	start, err := parseTimeParam(request.QueryParameter("start"), time.Time{})
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	end, err := parseTimeParam(request.QueryParameter("end"), nowFunc())
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	return start, end, nil
}

func exportTimestampedMetricValue(values []core.TimestampedMetricValue) types.MetricResult {
	result := types.MetricResult{
		Metrics: make([]types.MetricPoint, 0, len(values)),
	}
	for _, value := range values {
		if result.LatestTimestamp.Before(value.Timestamp) {
			result.LatestTimestamp = value.Timestamp
		}
		// TODO: clean up types in model api
		var intValue int64
		if value.ValueType == core.ValueInt64 {
			intValue = value.IntValue
		} else {
			intValue = int64(value.FloatValue)
		}

		result.Metrics = append(result.Metrics, types.MetricPoint{
			Timestamp: value.Timestamp,
			Value:     uint64(intValue),
		})
	}
	return result
}

func getLabels(request *restful.Request) (map[string]string, error) {
	labelsRaw := request.QueryParameter("labels")
	if labelsRaw == "" {
		return nil, nil
	}

	kvPairs := strings.Split(labelsRaw, ",")
	labels := make(map[string]string, len(kvPairs))
	for _, kvPair := range kvPairs {
		kvSplit := strings.SplitN(kvPair, ":", 2)
		if len(kvSplit) != 2 || kvSplit[0] == "" || kvSplit[1] == "" {
			return nil, fmt.Errorf("invalid label pair %q", kvPair)
		}
		labels[kvSplit[0]] = kvSplit[1]
	}

	return labels, nil
}
