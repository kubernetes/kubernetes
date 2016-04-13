// Copyright 2014 Google Inc. All Rights Reserved.
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
	"path/filepath"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/heapster/sources/api"
	"k8s.io/heapster/sources/datasource"
	"k8s.io/heapster/sources/nodes"
)

type kubePodsSource struct {
	kubeletPort int
	nodesApi    nodes.NodesApi
	podsApi     podsApi
	kubeletApi  datasource.Kubelet
	stateLock   sync.RWMutex
	podErrors   map[podInstance]int // guarded by stateLock
}

const (
	KubePodsSourceName = "Kube Pods Source"
)

func NewKubePodMetrics(kubeletPort int, kubeletApi datasource.Kubelet, nodesApi nodes.NodesApi, podsApi podsApi) api.Source {
	return &kubePodsSource{
		kubeletPort: kubeletPort,
		kubeletApi:  kubeletApi,
		podsApi:     podsApi,
		podErrors:   make(map[podInstance]int),
		nodesApi:    nodesApi,
	}
}

type podInstance struct {
	name string
	id   string
	ip   string
}

func (self *kubePodsSource) recordPodError(pod api.Pod) {
	// Heapster knows about pods before they are up and running on a node.
	// Ignore errors for Pods that are not Running.
	if pod.Status != "Running" {
		return
	}

	self.stateLock.Lock()
	defer self.stateLock.Unlock()

	podInstance := podInstance{name: pod.Name, id: pod.ID, ip: pod.HostPublicIP}
	self.podErrors[podInstance]++
}

func (self *kubePodsSource) getState() string {
	self.stateLock.RLock()
	defer self.stateLock.RUnlock()

	state := ""
	if len(self.podErrors) != 0 {
		state += fmt.Sprintf("\tPod Errors: %+v\n", self.podErrors)
	} else {
		state += "\tNo pod errors\n"
	}
	return state
}

func (self *kubePodsSource) getStatsFromKubelet(pod *api.Pod, containerName string, start, end time.Time) (*api.Container, error) {
	resource := filepath.Join("stats", pod.Namespace, pod.Name, pod.ID, containerName)
	if containerName == "/" {
		resource += "/"
	}

	return self.kubeletApi.GetContainer(datasource.Host{IP: pod.HostInternalIP, Port: self.kubeletPort, Resource: resource}, start, end)
}

func (self *kubePodsSource) getPodInfo(nodeList *nodes.NodeList, start, end time.Time) ([]api.Pod, error) {
	pods, err := self.podsApi.List(nodeList)
	if err != nil {
		return []api.Pod{}, err
	}
	var (
		wg sync.WaitGroup
	)
	for index := range pods {
		wg.Add(1)
		go func(pod *api.Pod) {
			defer wg.Done()
			for index, container := range pod.Containers {
				rawContainer, err := self.getStatsFromKubelet(pod, container.Name, start, end)
				if err != nil {
					// Containers could be in the process of being setup or restarting while the pod is alive.
					glog.Errorf("failed to get stats for container %q in pod %q/%q", container.Name, pod.Namespace, pod.Name)
					self.recordPodError(*pod)
					continue
				}
				if rawContainer == nil {
					continue
				}
				glog.V(5).Infof("Fetched stats from kubelet for container %s in pod %s", container.Name, pod.Name)
				pod.Containers[index].Hostname = pod.Hostname
				pod.Containers[index].ExternalID = pod.ExternalID
				pod.Containers[index].Spec.ContainerSpec = rawContainer.Spec.ContainerSpec
				pod.Containers[index].Stats = rawContainer.Stats
			}
		}(&pods[index])
	}
	wg.Wait()

	return pods, nil
}

func (self *kubePodsSource) GetInfo(start, end time.Time) (api.AggregateData, error) {
	kubeNodes, err := self.nodesApi.List()
	if err != nil || len(kubeNodes.Items) == 0 {
		return api.AggregateData{}, err
	}
	podsInfo, err := self.getPodInfo(kubeNodes, start, end)
	if err != nil {
		return api.AggregateData{}, err
	}

	return api.AggregateData{Pods: podsInfo}, nil
}

func (self *kubePodsSource) DebugInfo() string {
	desc := "Source type: kube-pod-metrics\n"
	desc += self.getState()
	desc += "\n"

	return desc
}

func (kps *kubePodsSource) Name() string {
	return KubePodsSourceName
}
