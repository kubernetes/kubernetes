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

package sources

import (
	"fmt"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/heapster/sources/api"
	"k8s.io/heapster/sources/datasource"
	"k8s.io/heapster/sources/nodes"
)

type kubeNodeMetrics struct {
	kubeletApi  datasource.Kubelet
	kubeletPort int
	nodesApi    nodes.NodesApi
}

func NewKubeNodeMetrics(kubeletPort int, kubeletApi datasource.Kubelet, nodesApi nodes.NodesApi) api.Source {
	return &kubeNodeMetrics{
		kubeletApi:  kubeletApi,
		kubeletPort: kubeletPort,
		nodesApi:    nodesApi,
	}
}

const (
	rootContainer             = "/"
	KubeNodeMetricsSourceName = "Kube Node Metrics Source"
)

var knownContainers = map[string]string{
	"/docker-daemon": "docker-daemon",
	"/kubelet":       "kubelet",
	"/kube-proxy":    "kube-proxy",
	"/system":        "system",
}

// Returns the host container, non-Kubernetes containers, and an error (if any).
func (self *kubeNodeMetrics) updateStats(host nodes.Host, info nodes.Info, start, end time.Time) (*api.Container, []api.Container, error) {
	// Get information for all containers.
	containers, err := self.kubeletApi.GetAllRawContainers(datasource.Host{IP: info.InternalIP, Port: self.kubeletPort}, start, end)
	if err != nil {
		glog.Errorf("Failed to get container stats from Kubelet on node %q", host)
		return nil, []api.Container{}, fmt.Errorf("failed to get container stats from Kubelet on node %q: %v", host, err)
	}
	if len(containers) == 0 {
		// no stats found.
		glog.V(3).Infof("No container stats from Kubelet on node %q", host)
		return nil, []api.Container{}, fmt.Errorf("no container stats from Kubelet on node %q", host)
	}

	// Find host container.
	hostIndex := -1
	hostString := string(host)
	externalID := string(info.ExternalID)
	for i := range containers {
		if containers[i].Name == rootContainer {
			hostIndex = i
		}
		if newName, exists := knownContainers[containers[i].Name]; exists {
			containers[i].Name = newName
		}
		containers[i].Hostname = hostString
		containers[i].ExternalID = externalID
	}
	var hostContainer *api.Container
	if hostIndex >= 0 {
		hostCopy := containers[hostIndex]
		hostContainer = &hostCopy
		containers = append(containers[:hostIndex], containers[hostIndex+1:]...)
		// This is temporary workaround for #399. To make unit consistent with cadvisor normalize to a conversion factor of 1024.
		hostContainer.Spec.Cpu.Limit = info.CpuCapacity * 1024 / 1000
		hostContainer.Spec.Memory.Limit = info.MemCapacity
		return hostContainer, containers, nil
	} else {
		return nil, []api.Container{}, fmt.Errorf("Host container not found")
	}
}

// Returns the host containers, non-Kubernetes containers, and an error (if any).
func (self *kubeNodeMetrics) getNodesInfo(nodeList *nodes.NodeList, start, end time.Time) ([]api.Container, []api.Container, error) {
	var (
		lock sync.Mutex
		wg   sync.WaitGroup
	)
	hostContainers := make([]api.Container, 0, len(nodeList.Items))
	rawContainers := make([]api.Container, 0, len(nodeList.Items))
	for host, info := range nodeList.Items {
		wg.Add(1)
		go func(host nodes.Host, info nodes.Info) {
			defer wg.Done()
			if hostContainer, containers, err := self.updateStats(host, info, start, end); err == nil {
				lock.Lock()
				defer lock.Unlock()
				if hostContainers != nil {
					hostContainers = append(hostContainers, *hostContainer)
				}
				rawContainers = append(rawContainers, containers...)
			}
		}(host, info)
	}
	wg.Wait()

	return hostContainers, rawContainers, nil
}

func (self *kubeNodeMetrics) GetInfo(start, end time.Time) (api.AggregateData, error) {
	kubeNodes, err := self.nodesApi.List()
	if err != nil || len(kubeNodes.Items) == 0 {
		return api.AggregateData{}, err
	}
	glog.V(3).Info("Fetched list of nodes from the master")
	hostContainers, rawContainers, err := self.getNodesInfo(kubeNodes, start, end)
	if err != nil {
		return api.AggregateData{}, err
	}

	return api.AggregateData{
		Machine:    hostContainers,
		Containers: rawContainers,
	}, nil
}

func (self *kubeNodeMetrics) DebugInfo() string {
	desc := "Source type: Kube Node Metrics\n"
	desc += self.nodesApi.DebugInfo() + "\n"

	return desc
}

func (kns *kubeNodeMetrics) Name() string {
	return KubeNodeMetricsSourceName
}
