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
		if nodes.Items[ix].Name == nodeID {
			return &nodes.Items[ix], nil
		}
	}
	return nil, fmt.Errorf("failed to find node: %s, %#v", nodeID, nodes)
}

type ClientNodeInfo struct {
	*client.Client
}

func (nodes ClientNodeInfo) GetNodeInfo(nodeID string) (*api.Minion, error) {
	return nodes.Minions().Get(nodeID)
}

func isVolumeConflict(volume api.Volume, pod *api.Pod) bool {
	if volume.Source.GCEPersistentDisk == nil {
		return false
	}
	pdName := volume.Source.GCEPersistentDisk.PDName

	manifest := &(pod.DesiredState.Manifest)
	for ix := range manifest.Volumes {
		if manifest.Volumes[ix].Source.GCEPersistentDisk != nil &&
			manifest.Volumes[ix].Source.GCEPersistentDisk.PDName == pdName {
			return true
		}
	}
	return false
}

// NoDiskConflict evaluates if a pod can fit due to the volumes it requests, and those that
// are already mounted. Some times of volumes are mounted onto node machines.  For now, these mounts
// are exclusive so if there is already a volume mounted on that node, another pod can't schedule
// there. This is GCE specific for now.
// TODO: migrate this into some per-volume specific code?
func NoDiskConflict(pod api.Pod, existingPods []api.Pod, node string) (bool, error) {
	manifest := &(pod.DesiredState.Manifest)
	for ix := range manifest.Volumes {
		for podIx := range existingPods {
			if isVolumeConflict(manifest.Volumes[ix], &existingPods[podIx]) {
				return false, nil
			}
		}
	}
	return true, nil
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

func NewSelectorMatchPredicate(info NodeInfo) FitPredicate {
	selector := &NodeSelector{
		info: info,
	}
	return selector.PodSelectorMatches
}

type NodeSelector struct {
	info NodeInfo
}

func (n *NodeSelector) PodSelectorMatches(pod api.Pod, existingPods []api.Pod, node string) (bool, error) {
	if len(pod.NodeSelector) == 0 {
		return true, nil
	}
	selector := labels.SelectorFromSet(pod.NodeSelector)
	minion, err := n.info.GetNodeInfo(node)
	if err != nil {
		return false, err
	}
	return selector.Matches(labels.Set(minion.Labels)), nil
}

func PodFitsPorts(pod api.Pod, existingPods []api.Pod, node string) (bool, error) {
	existingPorts := getUsedPorts(existingPods...)
	wantPorts := getUsedPorts(pod)
	for wport := range wantPorts {
		if wport == 0 {
			continue
		}
		if existingPorts[wport] {
			return false, nil
		}
	}
	return true, nil
}

func getUsedPorts(pods ...api.Pod) map[int]bool {
	ports := make(map[int]bool)
	for _, pod := range pods {
		for _, container := range pod.DesiredState.Manifest.Containers {
			for _, podPort := range container.Ports {
				ports[podPort.HostPort] = true
			}
		}
	}
	return ports
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
