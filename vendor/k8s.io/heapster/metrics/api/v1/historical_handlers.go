// Copyright 2016 Google Inc. All Rights Reserved.
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
	"net/http"
	"strconv"
	"strings"
	"time"

	restful "github.com/emicklei/go-restful"

	"k8s.io/heapster/metrics/api/v1/types"
	"k8s.io/heapster/metrics/core"
	"k8s.io/heapster/metrics/util/metrics"
)

// HistoricalApi wraps the standard API to overload the fetchers to use the
// one of the persistent storage sinks instead of the in-memory metrics sink
type HistoricalApi struct {
	*Api
}

// metricsAggregationFetcher represents the capability to fetch aggregations for metrics
type metricsAggregationFetcher interface {
	clusterAggregations(request *restful.Request, response *restful.Response)
	nodeAggregations(request *restful.Request, response *restful.Response)
	namespaceAggregations(request *restful.Request, response *restful.Response)
	podAggregations(request *restful.Request, response *restful.Response)
	podContainerAggregations(request *restful.Request, response *restful.Response)
	freeContainerAggregations(request *restful.Request, response *restful.Response)
	podListAggregations(request *restful.Request, response *restful.Response)

	isRunningInKubernetes() bool
}

// addAggregationRoutes adds routes to a webservice which point to a metricsAggregationFetcher's methods
func addAggregationRoutes(a metricsAggregationFetcher, ws *restful.WebService) {
	// The /metrics-aggregated/{aggregations}/{metric-name} endpoint exposes some aggregations for the Cluster entity of the historical API.
	ws.Route(ws.GET("/metrics-aggregated/{aggregations}/{metric-name:*}").
		To(metrics.InstrumentRouteFunc("clusterMetrics", a.clusterAggregations)).
		Doc("Export some cluster-level metric aggregations").
		Operation("clusterAggregations").
		Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
		Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
		Param(ws.QueryParameter("start", "Start time for requested metric").DataType("string")).
		Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
		Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
		Writes(types.MetricAggregationResult{}))

	// The /nodes/{node-name}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes some aggregations for a Node entity of the historical API.
	// The {node-name} parameter is the hostname of a specific node.
	ws.Route(ws.GET("/nodes/{node-name}/metrics-aggregated/{aggregations}/{metric-name:*}").
		To(metrics.InstrumentRouteFunc("nodeMetrics", a.nodeAggregations)).
		Doc("Export a node-level metric").
		Operation("nodeAggregations").
		Param(ws.PathParameter("node-name", "The name of the node to lookup").DataType("string")).
		Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
		Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
		Param(ws.QueryParameter("start", "Start time for requested metric").DataType("string")).
		Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
		Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
		Writes(types.MetricAggregationResult{}))

	if a.isRunningInKubernetes() {
		// The /namespaces/{namespace-name}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes some aggregations
		// for a Namespace entity of the historical API.
		ws.Route(ws.GET("/namespaces/{namespace-name}/metrics-aggregated/{aggregations}/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("namespaceMetrics", a.namespaceAggregations)).
			Doc("Export some namespace-level metric aggregations").
			Operation("namespaceAggregations").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")).
			Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricAggregationResult{}))

		// The /namespaces/{namespace-name}/pods/{pod-name}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// some aggregations for a Pod entity of the historical API.
		ws.Route(ws.GET("/namespaces/{namespace-name}/pods/{pod-name}/metrics-aggregated/{aggregations}/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podMetrics", a.podAggregations)).
			Doc("Export some pod-level metric aggregations").
			Operation("podAggregations").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")).
			Param(ws.PathParameter("pod-name", "The name of the pod to lookup").DataType("string")).
			Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricAggregationResult{}))

		// The /namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// some aggregations for a Container entity of the historical API.
		ws.Route(ws.GET("/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics-aggregated/{aggregations}/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podContainerMetrics", a.podContainerAggregations)).
			Doc("Export some aggregations for a Pod Container").
			Operation("podContainerAggregations").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to use").DataType("string")).
			Param(ws.PathParameter("pod-name", "The name of the pod to use").DataType("string")).
			Param(ws.PathParameter("container-name", "The name of the namespace to use").DataType("string")).
			Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricAggregationResult{}))

		// The /pod-id/{pod-id}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// some aggregations for a Pod entity of the historical API.
		ws.Route(ws.GET("/pod-id/{pod-id}/metrics-aggregated/{aggregations}/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podMetrics", a.podAggregations)).
			Doc("Export some pod-level metric aggregations").
			Operation("podAggregations").
			Param(ws.PathParameter("pod-id", "The UID of the pod to lookup").DataType("string")).
			Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricAggregationResult{}))

		// The /pod-id/{pod-id}/containers/{container-name}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// some aggregations for a Container entity of the historical API.
		ws.Route(ws.GET("/pod-id/{pod-id}/containers/{container-name}/metrics-aggregated/{aggregations}/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podContainerMetrics", a.podContainerAggregations)).
			Doc("Export some aggregations for a Pod Container").
			Operation("podContainerAggregations").
			Param(ws.PathParameter("pod-id", "The name of the pod to use").DataType("string")).
			Param(ws.PathParameter("container-name", "The name of the namespace to use").DataType("string")).
			Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricAggregationResult{}))
	}

	// The /nodes/{node-name}/freecontainers/{container-name}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
	// some aggregations for a free Container entity of the historical API.
	ws.Route(ws.GET("/nodes/{node-name}/freecontainers/{container-name}/metrics-aggregated/{aggregations}/{metric-name:*}").
		To(metrics.InstrumentRouteFunc("freeContainerMetrics", a.freeContainerAggregations)).
		Doc("Export a contsome iner-level metric aggregations for a free container").
		Operation("freeContainerAggregations").
		Param(ws.PathParameter("node-name", "The name of the node to use").DataType("string")).
		Param(ws.PathParameter("container-name", "The name of the container to use").DataType("string")).
		Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
		Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
		Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
		Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
		Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
		Writes(types.MetricAggregationResult{}))

	if a.isRunningInKubernetes() {
		// The /namespaces/{namespace-name}/pod-list/{pod-list}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// metrics for a list of pods of the historical API.
		ws.Route(ws.GET("/namespaces/{namespace-name}/pod-list/{pod-list}/metrics-aggregated/{aggregations}/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podListAggregations", a.podListAggregations)).
			Doc("Export some aggregations for all pods from the given list").
			Operation("podListAggregations").
			Param(ws.PathParameter("namespace-name", "The name of the namespace to lookup").DataType("string")).
			Param(ws.PathParameter("pod-list", "Comma separated list of pod names to lookup").DataType("string")).
			Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricAggregationResultList{}))

		// The /pod-id-list/{pod-id-list}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// metrics for a list of pod ids of the historical API.
		ws.Route(ws.GET("/pod-id-list/{pod-id-list}/metrics-aggregated/{aggregations}/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podListAggregations", a.podListAggregations)).
			Doc("Export an aggregation for all pods from the given list").
			Operation("podListAggregations").
			Param(ws.PathParameter("pod-id-list", "Comma separated list of pod UIDs to lookup").DataType("string")).
			Param(ws.PathParameter("aggregations", "A comma-separated list of requested aggregations").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Param(ws.QueryParameter("labels", "A comma-separated list of key:values pairs to use to search for a labeled metric").DataType("string")).
			Writes(types.MetricAggregationResultList{}))
	}
}

// RegisterHistorical registers the Historical API endpoints.  It will register the same endpoints
// as those in the model API, plus endpoints for aggregation retrieval, and endpoints to retrieve pod
// metrics by using the pod id.
func (normalApi *Api) RegisterHistorical(container *restful.Container) {
	ws := new(restful.WebService)
	ws.Path("/api/v1/historical").
		Doc("Root endpoint of the historical access API").
		Consumes("*/*").
		Produces(restful.MIME_JSON)

	a := &HistoricalApi{normalApi}
	addClusterMetricsRoutes(a, ws)
	addAggregationRoutes(a, ws)

	// register the endpoint for fetching raw metrics based on pod id
	if a.isRunningInKubernetes() {
		// The /pod-id/{pod-id}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// some aggregations for a Pod entity of the historical API.
		ws.Route(ws.GET("/pod-id/{pod-id}/metrics/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podMetrics", a.podMetrics)).
			Doc("Export some pod-level metric aggregations").
			Operation("podAggregations").
			Param(ws.PathParameter("pod-id", "The UID of the pod to lookup").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Writes(types.MetricResult{}))

		// The /pod-id/{pod-id}/containers/{container-name}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// some aggregations for a Container entity of the historical API.
		ws.Route(ws.GET("/pod-id/{pod-id}/containers/{container-name}/metrics/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podContainerMetrics", a.podContainerMetrics)).
			Doc("Export some aggregations for a Pod Container").
			Operation("podContainerAggregations").
			Param(ws.PathParameter("pod-id", "The uid of the pod to use").DataType("string")).
			Param(ws.PathParameter("container-name", "The name of the namespace to use").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Writes(types.MetricResult{}))

		// The /pod-id-list/{pod-id-list}/metrics-aggregated/{aggregations}/{metric-name} endpoint exposes
		// metrics for a list of pod ids of the historical API.
		ws.Route(ws.GET("/pod-id-list/{pod-id-list}/metrics/{metric-name:*}").
			To(metrics.InstrumentRouteFunc("podListAggregations", a.podListMetrics)).
			Doc("Export an aggregation for all pods from the given list").
			Operation("podListAggregations").
			Param(ws.PathParameter("pod-id-list", "Comma separated list of pod UIDs to lookup").DataType("string")).
			Param(ws.PathParameter("metric-name", "The name of the requested metric").DataType("string")).
			Param(ws.QueryParameter("start", "Start time for requested metrics").DataType("string")).
			Param(ws.QueryParameter("end", "End time for requested metric").DataType("string")).
			Writes(types.MetricResultList{}))
	}

	container.Add(ws)
}

// availableClusterMetrics returns a list of available cluster metric names.
func (a *HistoricalApi) availableClusterMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{ObjectType: core.MetricSetTypeCluster}
	a.processMetricNamesRequest(key, response)
}

// availableNodeMetrics returns a list of available node metric names.
func (a *HistoricalApi) availableNodeMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType: core.MetricSetTypeNode,
		NodeName:   request.PathParameter("node-name"),
	}
	a.processMetricNamesRequest(key, response)
}

// availableNamespaceMetrics returns a list of available namespace metric names.
func (a *HistoricalApi) availableNamespaceMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType:    core.MetricSetTypeNamespace,
		NamespaceName: request.PathParameter("namespace-name"),
	}
	a.processMetricNamesRequest(key, response)
}

// availablePodMetrics returns a list of available pod metric names.
func (a *HistoricalApi) availablePodMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType:    core.MetricSetTypePod,
		NamespaceName: request.PathParameter("namespace-name"),
		PodName:       request.PathParameter("pod-name"),
	}
	a.processMetricNamesRequest(key, response)
}

// availablePodContainerMetrics returns a list of available pod container metric names.
func (a *HistoricalApi) availablePodContainerMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType:    core.MetricSetTypePodContainer,
		NamespaceName: request.PathParameter("namespace-name"),
		PodName:       request.PathParameter("pod-name"),
		ContainerName: request.PathParameter("container-name"),
	}
	a.processMetricNamesRequest(key, response)
}

// availableFreeContainerMetrics returns a list of available pod metric names.
func (a *HistoricalApi) availableFreeContainerMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType:    core.MetricSetTypeSystemContainer,
		NodeName:      request.PathParameter("node-name"),
		ContainerName: request.PathParameter("container-name"),
	}
	a.processMetricNamesRequest(key, response)
}

// nodeList lists all nodes for which we have metrics
func (a *HistoricalApi) nodeList(request *restful.Request, response *restful.Response) {
	if resp, err := a.historicalSource.GetNodes(); err != nil {
		response.WriteError(http.StatusInternalServerError, err)
	} else {
		response.WriteEntity(resp)
	}
}

// namespaceList lists all namespaces for which we have metrics
func (a *HistoricalApi) namespaceList(request *restful.Request, response *restful.Response) {
	if resp, err := a.historicalSource.GetNamespaces(); err != nil {
		response.WriteError(http.StatusInternalServerError, err)
	} else {
		response.WriteEntity(resp)
	}
}

// namespacePodList lists all pods for which we have metrics in a particular namespace
func (a *HistoricalApi) namespacePodList(request *restful.Request, response *restful.Response) {
	if resp, err := a.historicalSource.GetPodsFromNamespace(request.PathParameter("namespace-name")); err != nil {
		response.WriteError(http.StatusInternalServerError, err)
	} else {
		response.WriteEntity(resp)
	}
}

// nodeSystemContainerList lists all system containers on a node for which we have metrics
func (a *HistoricalApi) nodeSystemContainerList(request *restful.Request, response *restful.Response) {
	if resp, err := a.historicalSource.GetSystemContainersFromNode(request.PathParameter("node-name")); err != nil {
		response.WriteError(http.StatusInternalServerError, err)
	} else {
		response.WriteEntity(resp)
	}
}

// clusterMetrics returns a metric timeseries for a metric of the Cluster entity.
func (a *HistoricalApi) clusterMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{ObjectType: core.MetricSetTypeCluster}
	a.processMetricRequest(key, request, response)
}

// nodeMetrics returns a metric timeseries for a metric of the Node entity.
func (a *HistoricalApi) nodeMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType: core.MetricSetTypeNode,
		NodeName:   request.PathParameter("node-name"),
	}
	a.processMetricRequest(key, request, response)
}

// namespaceMetrics returns a metric timeseries for a metric of the Namespace entity.
func (a *HistoricalApi) namespaceMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType:    core.MetricSetTypeNamespace,
		NamespaceName: request.PathParameter("namespace-name"),
	}
	a.processMetricRequest(key, request, response)
}

// podMetrics returns a metric timeseries for a metric of the Pod entity.
func (a *HistoricalApi) podMetrics(request *restful.Request, response *restful.Response) {
	var key core.HistoricalKey
	if request.PathParameter("pod-id") != "" {
		key = core.HistoricalKey{
			ObjectType: core.MetricSetTypePod,
			PodId:      request.PathParameter("pod-id"),
		}
	} else {
		key = core.HistoricalKey{
			ObjectType:    core.MetricSetTypePod,
			NamespaceName: request.PathParameter("namespace-name"),
			PodName:       request.PathParameter("pod-name"),
		}
	}
	a.processMetricRequest(key, request, response)
}

// freeContainerMetrics returns a metric timeseries for a metric of the Container entity.
// freeContainerMetrics addresses only free containers.
func (a *HistoricalApi) freeContainerMetrics(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType:    core.MetricSetTypeSystemContainer,
		NodeName:      request.PathParameter("node-name"),
		ContainerName: request.PathParameter("container-name"),
	}
	a.processMetricRequest(key, request, response)
}

// podListMetrics returns a list of metric timeseries for each for the listed nodes
func (a *HistoricalApi) podListMetrics(request *restful.Request, response *restful.Response) {
	start, end, err := getStartEndTimeHistorical(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}

	keys := []core.HistoricalKey{}
	if request.PathParameter("pod-id-list") != "" {
		for _, podId := range strings.Split(request.PathParameter("pod-id-list"), ",") {
			key := core.HistoricalKey{
				ObjectType: core.MetricSetTypePod,
				PodId:      podId,
			}
			keys = append(keys, key)
		}
	} else {
		for _, podName := range strings.Split(request.PathParameter("pod-list"), ",") {
			key := core.HistoricalKey{
				ObjectType:    core.MetricSetTypePod,
				NamespaceName: request.PathParameter("namespace-name"),
				PodName:       podName,
			}
			keys = append(keys, key)
		}
	}

	labels, err := getLabels(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}

	metricName := request.PathParameter("metric-name")
	convertedMetricName := convertMetricName(metricName)

	var metrics map[core.HistoricalKey][]core.TimestampedMetricValue
	if labels != nil {
		metrics, err = a.historicalSource.GetLabeledMetric(convertedMetricName, labels, keys, start, end)
	} else {
		metrics, err = a.historicalSource.GetMetric(convertedMetricName, keys, start, end)
	}

	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
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
func (a *HistoricalApi) podContainerMetrics(request *restful.Request, response *restful.Response) {
	var key core.HistoricalKey
	if request.PathParameter("pod-id") != "" {
		key = core.HistoricalKey{
			ObjectType:    core.MetricSetTypePodContainer,
			PodId:         request.PathParameter("pod-id"),
			ContainerName: request.PathParameter("container-name"),
		}
	} else {
		key = core.HistoricalKey{
			ObjectType:    core.MetricSetTypePodContainer,
			NamespaceName: request.PathParameter("namespace-name"),
			PodName:       request.PathParameter("pod-name"),
			ContainerName: request.PathParameter("container-name"),
		}
	}
	a.processMetricRequest(key, request, response)
}

// clusterAggregations returns a metric timeseries for a metric of the Cluster entity.
func (a *HistoricalApi) clusterAggregations(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{ObjectType: core.MetricSetTypeCluster}
	a.processAggregationRequest(key, request, response)
}

// nodeAggregations returns a metric timeseries for a metric of the Node entity.
func (a *HistoricalApi) nodeAggregations(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType: core.MetricSetTypeNode,
		NodeName:   request.PathParameter("node-name"),
	}
	a.processAggregationRequest(key, request, response)
}

// namespaceAggregations returns a metric timeseries for a metric of the Namespace entity.
func (a *HistoricalApi) namespaceAggregations(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType:    core.MetricSetTypeNamespace,
		NamespaceName: request.PathParameter("namespace-name"),
	}
	a.processAggregationRequest(key, request, response)
}

// podAggregations returns a metric timeseries for a metric of the Pod entity.
func (a *HistoricalApi) podAggregations(request *restful.Request, response *restful.Response) {
	var key core.HistoricalKey
	if request.PathParameter("pod-id") != "" {
		key = core.HistoricalKey{
			ObjectType: core.MetricSetTypePod,
			PodId:      request.PathParameter("pod-id"),
		}
	} else {
		key = core.HistoricalKey{
			ObjectType:    core.MetricSetTypePod,
			NamespaceName: request.PathParameter("namespace-name"),
			PodName:       request.PathParameter("pod-name"),
		}
	}
	a.processAggregationRequest(key, request, response)
}

// podContainerAggregations returns a metric timeseries for a metric of a Pod Container entity.
func (a *HistoricalApi) podContainerAggregations(request *restful.Request, response *restful.Response) {
	var key core.HistoricalKey
	if request.PathParameter("pod-id") != "" {
		key = core.HistoricalKey{
			ObjectType:    core.MetricSetTypePodContainer,
			PodId:         request.PathParameter("pod-id"),
			ContainerName: request.PathParameter("container-name"),
		}
	} else {
		key = core.HistoricalKey{
			ObjectType:    core.MetricSetTypePodContainer,
			NamespaceName: request.PathParameter("namespace-name"),
			PodName:       request.PathParameter("pod-name"),
			ContainerName: request.PathParameter("container-name"),
		}
	}
	a.processAggregationRequest(key, request, response)
}

// freeContainerAggregations returns a metric timeseries for a metric of the Container entity.
// freeContainerAggregations addresses only free containers.
func (a *HistoricalApi) freeContainerAggregations(request *restful.Request, response *restful.Response) {
	key := core.HistoricalKey{
		ObjectType:    core.MetricSetTypeSystemContainer,
		NodeName:      request.PathParameter("node-name"),
		ContainerName: request.PathParameter("container-name"),
	}
	a.processAggregationRequest(key, request, response)
}

// podListAggregations returns a list of metric timeseries for the specified pods.
func (a *HistoricalApi) podListAggregations(request *restful.Request, response *restful.Response) {
	start, end, err := getStartEndTimeHistorical(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	bucketSize, err := getBucketSize(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	aggregations, err := getAggregations(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	labels, err := getLabels(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	keys := []core.HistoricalKey{}
	if request.PathParameter("pod-id-list") != "" {
		for _, podId := range strings.Split(request.PathParameter("pod-id-list"), ",") {
			key := core.HistoricalKey{
				ObjectType: core.MetricSetTypePod,
				PodId:      podId,
			}
			keys = append(keys, key)
		}
	} else {
		for _, podName := range strings.Split(request.PathParameter("pod-list"), ",") {
			key := core.HistoricalKey{
				ObjectType:    core.MetricSetTypePod,
				NamespaceName: request.PathParameter("namespace-name"),
				PodName:       podName,
			}
			keys = append(keys, key)
		}
	}
	metricName := request.PathParameter("metric-name")
	convertedMetricName := convertMetricName(metricName)
	var metrics map[core.HistoricalKey][]core.TimestampedAggregationValue
	if labels != nil {
		metrics, err = a.historicalSource.GetLabeledAggregation(convertedMetricName, labels, aggregations, keys, start, end, bucketSize)
	} else {
		metrics, err = a.historicalSource.GetAggregation(convertedMetricName, aggregations, keys, start, end, bucketSize)
	}
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}

	result := types.MetricAggregationResultList{
		Items: make([]types.MetricAggregationResult, 0, len(keys)),
	}
	for _, key := range keys {
		result.Items = append(result.Items, exportTimestampedAggregationValue(metrics[key]))
	}
	response.PrettyPrint(false)
	response.WriteEntity(result)
}

// processMetricRequest retrieves a metric for the object at the requested key.
func (a *HistoricalApi) processMetricRequest(key core.HistoricalKey, request *restful.Request, response *restful.Response) {
	start, end, err := getStartEndTimeHistorical(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	labels, err := getLabels(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	metricName := request.PathParameter("metric-name")
	convertedMetricName := convertMetricName(metricName)

	var metrics map[core.HistoricalKey][]core.TimestampedMetricValue
	if labels != nil {
		metrics, err = a.historicalSource.GetLabeledMetric(convertedMetricName, labels, []core.HistoricalKey{key}, start, end)
	} else {
		metrics, err = a.historicalSource.GetMetric(convertedMetricName, []core.HistoricalKey{key}, start, end)
	}
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}

	converted := exportTimestampedMetricValue(metrics[key])
	response.WriteEntity(converted)
}

// processMetricNamesRequest retrieves the available metrics for the object at the specified key.
func (a *HistoricalApi) processMetricNamesRequest(key core.HistoricalKey, response *restful.Response) {
	if resp, err := a.historicalSource.GetMetricNames(key); err != nil {
		response.WriteError(http.StatusInternalServerError, err)
	} else {
		response.WriteEntity(resp)
	}
}

// processAggregationRequest retrieves one or more aggregations (across time) of a metric for the object specified at the given key.
func (a *HistoricalApi) processAggregationRequest(key core.HistoricalKey, request *restful.Request, response *restful.Response) {
	start, end, err := getStartEndTimeHistorical(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	bucketSize, err := getBucketSize(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	aggregations, err := getAggregations(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}
	labels, err := getLabels(request)
	if err != nil {
		response.WriteError(http.StatusBadRequest, err)
		return
	}

	metricName := request.PathParameter("metric-name")
	convertedMetricName := convertMetricName(metricName)
	var metrics map[core.HistoricalKey][]core.TimestampedAggregationValue
	if labels != nil {
		metrics, err = a.historicalSource.GetLabeledAggregation(convertedMetricName, labels, aggregations, []core.HistoricalKey{key}, start, end, bucketSize)
	} else {
		metrics, err = a.historicalSource.GetAggregation(convertedMetricName, aggregations, []core.HistoricalKey{key}, start, end, bucketSize)
	}
	if err != nil {
		response.WriteError(http.StatusInternalServerError, err)
		return
	}

	converted := exportTimestampedAggregationValue(metrics[key])
	response.WriteEntity(converted)
}

// getBucketSize parses the bucket size specifier into a
func getBucketSize(request *restful.Request) (time.Duration, error) {
	rawSize := request.QueryParameter("bucket")
	if rawSize == "" {
		return 0, nil
	}

	if len(rawSize) < 2 {
		return 0, fmt.Errorf("unable to parse bucket size: %q is too short to be a duration", rawSize)
	}
	var multiplier time.Duration
	var num string

	switch rawSize[len(rawSize)-1] {
	case 's':
		// could be s or ms
		if len(rawSize) < 3 || rawSize[len(rawSize)-2] != 'm' {
			multiplier = time.Second
			num = rawSize[:len(rawSize)-1]
		} else {
			multiplier = time.Millisecond
			num = rawSize[:len(rawSize)-2]
		}
	case 'h':
		multiplier = time.Hour
		num = rawSize[:len(rawSize)-1]
	case 'd':
		multiplier = 24 * time.Hour
		num = rawSize[:len(rawSize)-1]
	case 'm':
		multiplier = time.Minute
		num = rawSize[:len(rawSize)-1]
	default:
		return 0, fmt.Errorf("unable to parse bucket size: %q has no known duration suffix", rawSize)
	}

	parsedNum, err := strconv.ParseUint(num, 10, 64)
	if err != nil {
		return 0, err
	}

	return time.Duration(parsedNum) * multiplier, nil
}

// getAggregations extracts and validates the list of requested aggregations
func getAggregations(request *restful.Request) ([]core.AggregationType, error) {
	aggregationsRaw := strings.Split(request.PathParameter("aggregations"), ",")
	if len(aggregationsRaw) == 0 {
		return nil, fmt.Errorf("No aggregations specified")
	}

	aggregations := make([]core.AggregationType, len(aggregationsRaw))

	for ind, aggNameRaw := range aggregationsRaw {
		aggName := core.AggregationType(aggNameRaw)
		if _, ok := core.AllAggregations[aggName]; !ok {
			return nil, fmt.Errorf("Unknown aggregation %q", aggName)
		}
		aggregations[ind] = aggName
	}

	return aggregations, nil
}

// exportMetricValue converts a core.MetricValue into an API MetricValue
func exportMetricValue(value *core.MetricValue) *types.MetricValue {
	if value == nil {
		return nil
	}

	if value.ValueType == core.ValueInt64 {
		return &types.MetricValue{
			IntValue: &value.IntValue,
		}
	} else {
		floatVal := float64(value.FloatValue)
		return &types.MetricValue{
			FloatValue: &floatVal,
		}
	}
}

// extractMetricValue checks to see if the given metric was present in the results, and if so,
// returns it in API form
func extractMetricValue(aggregations *core.AggregationValue, aggName core.AggregationType) *types.MetricValue {
	if inputVal, ok := aggregations.Aggregations[aggName]; ok {
		return exportMetricValue(&inputVal)
	} else {
		return nil
	}
}

// exportTimestampedAggregationValue converts a core.TimestampedAggregationValue into an API MetricAggregationResult
func exportTimestampedAggregationValue(values []core.TimestampedAggregationValue) types.MetricAggregationResult {
	result := types.MetricAggregationResult{
		Buckets:    make([]types.MetricAggregationBucket, 0, len(values)),
		BucketSize: 0,
	}
	for _, value := range values {
		// just use the largest bucket size, since all bucket sizes should be uniform
		// (except for the last one, which may be smaller)
		if result.BucketSize < value.BucketSize {
			result.BucketSize = value.BucketSize
		}

		bucket := types.MetricAggregationBucket{
			Timestamp: value.Timestamp,

			Count: value.Count,

			Average: extractMetricValue(&value.AggregationValue, core.AggregationTypeAverage),
			Maximum: extractMetricValue(&value.AggregationValue, core.AggregationTypeMaximum),
			Minimum: extractMetricValue(&value.AggregationValue, core.AggregationTypeMinimum),
			Median:  extractMetricValue(&value.AggregationValue, core.AggregationTypeMedian),

			Percentiles: make(map[string]types.MetricValue, 3),
		}

		if val, ok := value.Aggregations[core.AggregationTypePercentile50]; ok {
			bucket.Percentiles["50"] = *exportMetricValue(&val)
		}
		if val, ok := value.Aggregations[core.AggregationTypePercentile95]; ok {
			bucket.Percentiles["95"] = *exportMetricValue(&val)
		}
		if val, ok := value.Aggregations[core.AggregationTypePercentile99]; ok {
			bucket.Percentiles["99"] = *exportMetricValue(&val)
		}

		result.Buckets = append(result.Buckets, bucket)
	}
	return result
}

// getStartEndTimeHistorical fetches the start and end times of the request.  Unlike
// getStartEndTime, this function returns an error if the start time is not passed
// (or is zero, since many things in go use time.Time{} as an empty value) --
// different sinks have different defaults, and certain sinks have issues if an actual
// time of zero (i.e. the epoch) is used (because that would be too many data points
// to consider in certain cases).  Require applications to pass an explicit end time
// that they can deal with.
func getStartEndTimeHistorical(request *restful.Request) (time.Time, time.Time, error) {
	start, end, err := getStartEndTime(request)
	if err != nil {
		return start, end, err
	}

	if start.IsZero() {
		return start, end, fmt.Errorf("no start time (or a start time of zero) provided")
	}

	return start, end, err
}
