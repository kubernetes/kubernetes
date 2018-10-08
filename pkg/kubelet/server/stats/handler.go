/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package stats

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path"
	"time"

	restful "github.com/emicklei/go-restful"
	"github.com/golang/glog"
	cadvisorapi "github.com/google/cadvisor/info/v1"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/volume"
)

// Host methods required by stats handlers.
type StatsProvider interface {
	// The following stats are provided by either CRI or cAdvisor.
	//
	// ListPodStats returns the stats of all the containers managed by pods.
	ListPodStats() ([]statsapi.PodStats, error)
	// ImageFsStats returns the stats of the image filesystem.
	ImageFsStats() (*statsapi.FsStats, error)

	// The following stats are provided by cAdvisor.
	//
	// GetCgroupStats returns the stats and the networking usage of the cgroup
	// with the specified cgroupName.
	GetCgroupStats(cgroupName string, updateStats bool) (*statsapi.ContainerStats, *statsapi.NetworkStats, error)
	// RootFsStats returns the stats of the node root filesystem.
	RootFsStats() (*statsapi.FsStats, error)

	// The following stats are provided by cAdvisor for legacy usage.
	//
	// GetContainerInfo returns the information of the container with the
	// containerName managed by the pod with the uid.
	GetContainerInfo(podFullName string, uid types.UID, containerName string, req *cadvisorapi.ContainerInfoRequest) (*cadvisorapi.ContainerInfo, error)
	// GetRawContainerInfo returns the information of the container with the
	// containerName. If subcontainers is true, this function will return the
	// information of all the sub-containers as well.
	GetRawContainerInfo(containerName string, req *cadvisorapi.ContainerInfoRequest, subcontainers bool) (map[string]*cadvisorapi.ContainerInfo, error)

	// The following information is provided by Kubelet.
	//
	// GetPodByName returns the spec of the pod with the name in the specified
	// namespace.
	GetPodByName(namespace, name string) (*v1.Pod, bool)
	// GetNode returns the spec of the local node.
	GetNode() (*v1.Node, error)
	// GetNodeConfig returns the configuration of the local node.
	GetNodeConfig() cm.NodeConfig
	// ListVolumesForPod returns the stats of the volume used by the pod with
	// the podUID.
	ListVolumesForPod(podUID types.UID) (map[string]volume.Volume, bool)
	// GetPods returns the specs of all the pods running on this node.
	GetPods() []*v1.Pod

	// RlimitStats returns the rlimit stats of system.
	RlimitStats() (*statsapi.RlimitStats, error)

	// GetPodCgroupRoot returns the literal cgroupfs value for the cgroup containing all pods
	GetPodCgroupRoot() string

	// GetPodByCgroupfs provides the pod that maps to the specified cgroup literal, as well
	// as whether the pod was found.
	GetPodByCgroupfs(cgroupfs string) (*v1.Pod, bool)
}

type handler struct {
	provider        StatsProvider
	summaryProvider SummaryProvider
}

func CreateHandlers(rootPath string, provider StatsProvider, summaryProvider SummaryProvider) *restful.WebService {
	h := &handler{provider, summaryProvider}

	ws := &restful.WebService{}
	ws.Path(rootPath).
		Produces(restful.MIME_JSON)

	endpoints := []struct {
		path    string
		handler restful.RouteFunction
	}{
		{"", h.handleStats},
		{"/summary", h.handleSummary},
		{"/container", h.handleSystemContainer},
		{"/{podName}/{containerName}", h.handlePodContainer},
		{"/{namespace}/{podName}/{uid}/{containerName}", h.handlePodContainer},
	}

	for _, e := range endpoints {
		for _, method := range []string{"GET", "POST"} {
			ws.Route(ws.
				Method(method).
				Path(e.path).
				To(e.handler))
		}
	}

	return ws
}

type StatsRequest struct {
	// The name of the container for which to request stats.
	// Default: /
	// +optional
	ContainerName string `json:"containerName,omitempty"`

	// Max number of stats to return.
	// If start and end time are specified this limit is ignored.
	// Default: 60
	// +optional
	NumStats int `json:"num_stats,omitempty"`

	// Start time for which to query information.
	// If omitted, the beginning of time is assumed.
	// +optional
	Start time.Time `json:"start,omitempty"`

	// End time for which to query information.
	// If omitted, current time is assumed.
	// +optional
	End time.Time `json:"end,omitempty"`

	// Whether to also include information from subcontainers.
	// Default: false.
	// +optional
	Subcontainers bool `json:"subcontainers,omitempty"`
}

func (r *StatsRequest) cadvisorRequest() *cadvisorapi.ContainerInfoRequest {
	return &cadvisorapi.ContainerInfoRequest{
		NumStats: r.NumStats,
		Start:    r.Start,
		End:      r.End,
	}
}

func parseStatsRequest(request *restful.Request) (StatsRequest, error) {
	// Default request.
	query := StatsRequest{
		NumStats: 60,
	}

	err := json.NewDecoder(request.Request.Body).Decode(&query)
	if err != nil && err != io.EOF {
		return query, err
	}
	return query, nil
}

// Handles root container stats requests to /stats
func (h *handler) handleStats(request *restful.Request, response *restful.Response) {
	query, err := parseStatsRequest(request)
	if err != nil {
		handleError(response, "/stats", err)
		return
	}

	// Root container stats.
	statsMap, err := h.provider.GetRawContainerInfo("/", query.cadvisorRequest(), false)
	if err != nil {
		handleError(response, fmt.Sprintf("/stats %v", query), err)
		return
	}
	writeResponse(response, statsMap["/"])
}

// Handles stats summary requests to /stats/summary
// If "only_cpu_and_memory" GET param is true then only cpu and memory is returned in response.
func (h *handler) handleSummary(request *restful.Request, response *restful.Response) {
	onlyCPUAndMemory := false
	request.Request.ParseForm()
	if onlyCluAndMemoryParam, found := request.Request.Form["only_cpu_and_memory"]; found &&
		len(onlyCluAndMemoryParam) == 1 && onlyCluAndMemoryParam[0] == "true" {
		onlyCPUAndMemory = true
	}
	var summary *statsapi.Summary
	var err error
	if onlyCPUAndMemory {
		summary, err = h.summaryProvider.GetCPUAndMemoryStats()
	} else {
		// external calls to the summary API use cached stats
		forceStatsUpdate := false
		summary, err = h.summaryProvider.Get(forceStatsUpdate)
	}
	if err != nil {
		handleError(response, "/stats/summary", err)
	} else {
		writeResponse(response, summary)
	}
}

// Handles non-kubernetes container stats requests to /stats/container/
func (h *handler) handleSystemContainer(request *restful.Request, response *restful.Response) {
	query, err := parseStatsRequest(request)
	if err != nil {
		handleError(response, "/stats/container", err)
		return
	}

	// Non-Kubernetes container stats.
	containerName := path.Join("/", query.ContainerName)
	stats, err := h.provider.GetRawContainerInfo(
		containerName, query.cadvisorRequest(), query.Subcontainers)
	if err != nil {
		if _, ok := stats[containerName]; ok {
			// If the failure is partial, log it and return a best-effort response.
			glog.Errorf("Partial failure issuing GetRawContainerInfo(%v): %v", query, err)
		} else {
			handleError(response, fmt.Sprintf("/stats/container %v", query), err)
			return
		}
	}
	writeResponse(response, stats)
}

// Handles kubernetes pod/container stats requests to:
// /stats/<pod name>/<container name>
// /stats/<namespace>/<pod name>/<uid>/<container name>
func (h *handler) handlePodContainer(request *restful.Request, response *restful.Response) {
	query, err := parseStatsRequest(request)
	if err != nil {
		handleError(response, request.Request.URL.String(), err)
		return
	}

	// Default parameters.
	params := map[string]string{
		"namespace": metav1.NamespaceDefault,
		"uid":       "",
	}
	for k, v := range request.PathParameters() {
		params[k] = v
	}

	if params["podName"] == "" || params["containerName"] == "" {
		response.WriteErrorString(http.StatusBadRequest,
			fmt.Sprintf("Invalid pod container request: %v", params))
		return
	}

	pod, ok := h.provider.GetPodByName(params["namespace"], params["podName"])
	if !ok {
		glog.V(4).Infof("Container not found: %v", params)
		response.WriteError(http.StatusNotFound, kubecontainer.ErrContainerNotFound)
		return
	}
	stats, err := h.provider.GetContainerInfo(
		kubecontainer.GetPodFullName(pod),
		types.UID(params["uid"]),
		params["containerName"],
		query.cadvisorRequest())

	if err != nil {
		handleError(response, fmt.Sprintf("%s %v", request.Request.URL.String(), query), err)
		return
	}
	writeResponse(response, stats)
}

func writeResponse(response *restful.Response, stats interface{}) {
	if err := response.WriteAsJson(stats); err != nil {
		glog.Errorf("Error writing response: %v", err)
	}
}

// handleError serializes an error object into an HTTP response.
// request is provided for logging.
func handleError(response *restful.Response, request string, err error) {
	switch err {
	case kubecontainer.ErrContainerNotFound:
		response.WriteError(http.StatusNotFound, err)
	default:
		msg := fmt.Sprintf("Internal Error: %v", err)
		glog.Errorf("HTTP InternalServerError serving %s: %s", request, msg)
		response.WriteErrorString(http.StatusInternalServerError, msg)
	}
}
