/*
Copyright 2014 Google Inc. All rights reserved.

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

package scheduler

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/resources"
	"github.com/golang/glog"
)

type NodeInfo interface {
	GetNodeInfo(nodeID string) (*api.Minion, error)
}

type StaticNodeInfo struct {
	*api.MinionList
}

func (nodes StaticNodeInfo) GetNodeInfo(nodeID string) (*api.Minion, error) {
	for ix := range nodes.Items {
		if nodes.Items[ix].ID == nodeID {
			return &nodes.Items[ix], nil
		}
	}
	return nil, fmt.Errorf("failed to find node: %s, %#v", nodeID, nodes)
}

type ClientNodeInfo struct {
	*client.Client
}

func (nodes ClientNodeInfo) GetNodeInfo(nodeID string) (*api.Minion, error) {
	return nodes.GetMinion(nodeID)
}

type ResourceFit struct {
	info NodeInfo
}

type resourceRequest struct {
	milliCPU int
	memory   int
}

func getResourceRequest(pod *api.Pod) resourceRequest {
	result := resourceRequest{}
	for ix := range pod.DesiredState.Manifest.Containers {
		result.memory += pod.DesiredState.Manifest.Containers[ix].Memory
		result.milliCPU += pod.DesiredState.Manifest.Containers[ix].CPU
	}
	return result
}

// PodFitsResources calculates fit based on requested, rather than used resources
func (r *ResourceFit) PodFitsResources(pod api.Pod, existingPods []api.Pod, node string) (bool, error) {
	podRequest := getResourceRequest(&pod)
	if podRequest.milliCPU == 0 && podRequest.memory == 0 {
		// no resources requested always fits.
		return true, nil
	}
	info, err := r.info.GetNodeInfo(node)
	if err != nil {
		return false, err
	}
	milliCPURequested := 0
	memoryRequested := 0
	for ix := range existingPods {
		existingRequest := getResourceRequest(&existingPods[ix])
		milliCPURequested += existingRequest.milliCPU
		memoryRequested += existingRequest.memory
	}

	// TODO: convert to general purpose resource matching, when pods ask for resources
	totalMilliCPU := int(resources.GetFloatResource(info.NodeResources.Capacity, resources.CPU, 0) * 1000)
	totalMemory := resources.GetIntegerResource(info.NodeResources.Capacity, resources.Memory, 0)

	fitsCPU := totalMilliCPU == 0 || (totalMilliCPU-milliCPURequested) >= podRequest.milliCPU
	fitsMemory := totalMemory == 0 || (totalMemory-memoryRequested) >= podRequest.memory
	glog.V(3).Infof("Calculated fit: cpu: %s, memory %s", fitsCPU, fitsMemory)

	return fitsCPU && fitsMemory, nil
}

func NewResourceFitPredicate(info NodeInfo) FitPredicate {
	fit := &ResourceFit{
		info: info,
	}
	return fit.PodFitsResources
}

func PodFitsPorts(pod api.Pod, existingPods []api.Pod, node string) (bool, error) {
	for _, scheduledPod := range existingPods {
		for _, container := range pod.DesiredState.Manifest.Containers {
			for _, port := range container.Ports {
				if port.HostPort == 0 {
					continue
				}
				if containsPort(scheduledPod, port) {
					return false, nil
				}
			}
		}
	}
	return true, nil
}

func containsPort(pod api.Pod, port api.Port) bool {
	for _, container := range pod.DesiredState.Manifest.Containers {
		for _, podPort := range container.Ports {
			if podPort.HostPort == port.HostPort {
				return true
			}
		}
	}
	return false
}

// MapPodsToMachines obtains a list of pods and pivots that list into a map where the keys are host names
// and the values are the list of pods running on that host.
func MapPodsToMachines(lister PodLister) (map[string][]api.Pod, error) {
	machineToPods := map[string][]api.Pod{}
	// TODO: perform more targeted query...
	pods, err := lister.ListPods(labels.Everything())
	if err != nil {
		return map[string][]api.Pod{}, err
	}
	for _, scheduledPod := range pods {
		host := scheduledPod.DesiredState.Host
		machineToPods[host] = append(machineToPods[host], scheduledPod)
	}
	return machineToPods, nil
}
