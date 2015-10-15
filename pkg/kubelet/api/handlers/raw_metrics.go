/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	restful "github.com/emicklei/go-restful"
	"github.com/golang/glog"
	cadvisorv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/api"
	kubeapi "k8s.io/kubernetes/pkg/kubelet/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

type StatsProvider interface {
	GetPods() []*api.Pod
	GetContainerInfos(options cadvisorv2.RequestOptions) (map[string]cadvisorv2.ContainerInfo, error)
}

func RawMetricsHandler(stats StatsProvider, statusManager status.Manager) restful.RouteFunction {
	handler := rawMetricsHandler{stats, statusManager}
	return handler.Handle
}

type rawMetricsHandler struct {
	stats         StatsProvider
	statusManager status.Manager
}

func (h *rawMetricsHandler) Handle(request *restful.Request, response *restful.Response) {
	options, err := h.parseRequest(request)
	if err != nil {
		glog.V(1).Infof("Bad request: %v", err)
		response.WriteError(http.StatusBadRequest, err)
		return
	}

	infos, err := h.stats.GetContainerInfos(options)
	if err != nil {
		glog.Errorf("Unable to get container stats: %v", err)
		response.WriteError(http.StatusInternalServerError, err)
		return
	}

	metrics := h.buildRawMetrics(infos)
	response.WriteAsJson(metrics)
}

func (h *rawMetricsHandler) parseRequest(request *restful.Request) (cadvisorv2.RequestOptions, error) {
	// TODO: Add support for POST requests.
	if request.Request.Method != "GET" {
		return cadvisorv2.RequestOptions{}, fmt.Errorf("Cannot parse %q request method.", request.Request.Method)
	}

	options := cadvisorv2.RequestOptions{
		Count: 60, // Default number of stats if unspecified.

	}

	if start := request.QueryParameter("start"); start != "" {
		var startTime time.Time
		if err := json.Unmarshal([]byte(start), &startTime); err != nil {
			return options, err
		}
		options.Start = startTime
	}

	if end := request.QueryParameter("end"); end != "" {
		var endTime time.Time
		if err := json.Unmarshal([]byte(end), &endTime); err != nil {
			return options, err
		}
		options.End = endTime
	}

	if count := request.QueryParameter("count"); count != "" {
		var countVal int
		if err := json.Unmarshal([]byte(count), &countVal); err != nil {
			return options, err
		}
		options.Count = countVal
	}

	return options, nil
}

func (h *rawMetricsHandler) buildRawMetrics(infos map[string]cadvisorv2.ContainerInfo) kubeapi.RawMetrics {
	const prefix = "/"

	// FIXME - add TypeMeta fields to everything
	var metrics kubeapi.RawMetrics // FIXME - node name

	// Machine metrics.
	if machineInfo, found := infos[prefix]; found {
		metrics.Machine = h.infoToMetrics(machineInfo, "/")
	}

	// System metrics.
	for _, name := range [...]string{"system", "kubelet", "kube-proxy", "docker-daemon"} {
		if info, found := infos[prefix+name]; found {
			metrics.SystemContainers = append(metrics.SystemContainers, h.infoToMetrics(info, name))
		}
	}

	for _, pod := range h.stats.GetPods() {
		status, found := h.statusManager.GetPodStatus(pod.UID)
		if !found {
			continue // Not running.
		}

		var podMetrics kubeapi.PodMetrics
		podMetrics.Name = pod.Name

		for _, c := range status.ContainerStatuses {
			id := kubecontainer.ParseContainerID(c.ContainerID)
			if info, found := infos[prefix+id.ID]; found {
				podMetrics.Containers = append(podMetrics.Containers, h.infoToMetrics(info, c.Name))
			}
		}

		metrics.Pods = append(metrics.Pods, podMetrics)
	}

	return metrics
}

func (h *rawMetricsHandler) infoToMetrics(info cadvisorv2.ContainerInfo, name string) kubeapi.ContainerMetrics {
	return kubeapi.ContainerMetrics{
		ObjectMeta: api.ObjectMeta{Name: name},
		Spec:       &info.Spec,
		Stats:      info.Stats,
	}
}
