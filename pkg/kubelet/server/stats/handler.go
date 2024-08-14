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

//go:generate mockery
package stats

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	restful "github.com/emicklei/go-restful/v3"
	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/volume"
)

// Provider hosts methods required by stats handlers.
type Provider interface {
	// The following stats are provided by either CRI or cAdvisor.
	//
	// ListPodStats returns the stats of all the containers managed by pods.
	ListPodStats(ctx context.Context) ([]statsapi.PodStats, error)
	// ListPodStatsAndUpdateCPUNanoCoreUsage updates the cpu nano core usage for
	// the containers and returns the stats for all the pod-managed containers.
	ListPodCPUAndMemoryStats(ctx context.Context) ([]statsapi.PodStats, error)
	// ListPodStatsAndUpdateCPUNanoCoreUsage returns the stats of all the
	// containers managed by pods and force update the cpu usageNanoCores.
	// This is a workaround for CRI runtimes that do not integrate with
	// cadvisor. See https://github.com/kubernetes/kubernetes/issues/72788
	// for more details.
	ListPodStatsAndUpdateCPUNanoCoreUsage(ctx context.Context) ([]statsapi.PodStats, error)
	// ImageFsStats returns the stats of the image filesystem.
	// Kubelet allows three options for container filesystems
	// Everything is on node fs (so no image filesystem)
	// Container storage is on a dedicated disk (imageFs is separate from root)
	// Container Filesystem is on root and Images are stored on ImageFs
	// First return parameter is the image filesystem and
	// second parameter is the container filesystem
	ImageFsStats(ctx context.Context) (imageFs *statsapi.FsStats, containerFs *statsapi.FsStats, callErr error)
	// The following stats are provided by cAdvisor.
	//
	// GetCgroupStats returns the stats and the networking usage of the cgroup
	// with the specified cgroupName.
	GetCgroupStats(cgroupName string, updateStats bool) (*statsapi.ContainerStats, *statsapi.NetworkStats, error)
	// GetCgroupCPUAndMemoryStats returns the CPU and memory stats of the cgroup with the specified cgroupName.
	GetCgroupCPUAndMemoryStats(cgroupName string, updateStats bool) (*statsapi.ContainerStats, error)

	// RootFsStats returns the stats of the node root filesystem.
	RootFsStats() (*statsapi.FsStats, error)

	// GetRequestedContainersInfo returns the information of the container with
	// the containerName, and with the specified cAdvisor options.
	GetRequestedContainersInfo(containerName string, options cadvisorv2.RequestOptions) (map[string]*cadvisorapi.ContainerInfo, error)

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
	// ListBlockVolumesForPod returns the stats of the volume used by the
	// pod with the podUID.
	ListBlockVolumesForPod(podUID types.UID) (map[string]volume.BlockVolume, bool)
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
	provider        Provider
	summaryProvider SummaryProvider
}

// CreateHandlers creates the REST handlers for the stats.
func CreateHandlers(rootPath string, provider Provider, summaryProvider SummaryProvider) *restful.WebService {
	h := &handler{provider, summaryProvider}

	ws := &restful.WebService{}
	ws.Path(rootPath).
		Produces(restful.MIME_JSON)

	endpoints := []struct {
		path    string
		handler restful.RouteFunction
	}{
		{"/summary", h.handleSummary},
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

// Handles stats summary requests to /stats/summary
// If "only_cpu_and_memory" GET param is true then only cpu and memory is returned in response.
func (h *handler) handleSummary(request *restful.Request, response *restful.Response) {
	ctx := request.Request.Context()
	onlyCPUAndMemory := false
	err := request.Request.ParseForm()
	if err != nil {
		handleError(response, "/stats/summary", fmt.Errorf("parse form failed: %w", err))
		return
	}
	if onlyCluAndMemoryParam, found := request.Request.Form["only_cpu_and_memory"]; found &&
		len(onlyCluAndMemoryParam) == 1 && onlyCluAndMemoryParam[0] == "true" {
		onlyCPUAndMemory = true
	}
	var summary *statsapi.Summary
	if onlyCPUAndMemory {
		summary, err = h.summaryProvider.GetCPUAndMemoryStats(ctx)
	} else {
		// external calls to the summary API use cached stats
		forceStatsUpdate := false
		summary, err = h.summaryProvider.Get(ctx, forceStatsUpdate)
	}
	if err != nil {
		handleError(response, "/stats/summary", err)
	} else {
		writeResponse(response, summary)
	}
}

func writeResponse(response *restful.Response, stats interface{}) {
	if err := response.WriteAsJson(stats); err != nil {
		klog.ErrorS(err, "Error writing response")
	}
}

// handleError serializes an error object into an HTTP response.
// request is provided for logging.
func handleError(response *restful.Response, request string, err error) {
	switch {
	case errors.Is(err, kubecontainer.ErrContainerNotFound):
		response.WriteError(http.StatusNotFound, err)
	default:
		msg := fmt.Sprintf("Internal Error: %v", err)
		klog.ErrorS(err, "HTTP InternalServerError serving", "request", request)
		response.WriteErrorString(http.StatusInternalServerError, msg)
	}
}
