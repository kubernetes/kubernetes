/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package predicates

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"

	"github.com/golang/glog"
)

type NodeInfo interface {
	GetNodeInfo(nodeID string) (*api.Node, error)
}

type StaticNodeInfo struct {
	*api.NodeList
}

func (nodes StaticNodeInfo) GetNodeInfo(nodeID string) (*api.Node, error) {
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

func (nodes ClientNodeInfo) GetNodeInfo(nodeID string) (*api.Node, error) {
	return nodes.Nodes().Get(nodeID)
}

func isVolumeConflict(volume api.Volume, pod *api.Pod) bool {
	if volume.GCEPersistentDisk != nil {
		disk := volume.GCEPersistentDisk

		manifest := &(pod.Spec)
		for ix := range manifest.Volumes {
			if manifest.Volumes[ix].GCEPersistentDisk != nil &&
				manifest.Volumes[ix].GCEPersistentDisk.PDName == disk.PDName &&
				!(manifest.Volumes[ix].GCEPersistentDisk.ReadOnly && disk.ReadOnly) {
				return true
			}
		}
	}
	if volume.AWSElasticBlockStore != nil {
		volumeID := volume.AWSElasticBlockStore.VolumeID

		manifest := &(pod.Spec)
		for ix := range manifest.Volumes {
			if manifest.Volumes[ix].AWSElasticBlockStore != nil &&
				manifest.Volumes[ix].AWSElasticBlockStore.VolumeID == volumeID {
				return true
			}
		}
	}
	return false
}

// NoDiskConflict evaluates if a pod can fit due to the volumes it requests, and those that
// are already mounted. Some times of volumes are mounted onto node machines.  For now, these mounts
// are exclusive so if there is already a volume mounted on that node, another pod can't schedule
// there. This is GCE and Amazon EBS specific for now.
// TODO: migrate this into some per-volume specific code?
func NoDiskConflict(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error) {
	manifest := &(pod.Spec)
	for ix := range manifest.Volumes {
		for podIx := range existingPods {
			if isVolumeConflict(manifest.Volumes[ix], existingPods[podIx]) {
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
	milliCPU int64
	memory   int64
}

var FailedResourceType string

func getResourceRequest(pod *api.Pod) resourceRequest {
	result := resourceRequest{}
	for _, container := range pod.Spec.Containers {
		requests := container.Resources.Requests
		result.memory += requests.Memory().Value()
		result.milliCPU += requests.Cpu().MilliValue()
	}
	return result
}

func CheckPodsExceedingFreeResources(pods []*api.Pod, capacity api.ResourceList) (fitting []*api.Pod, notFittingCPU, notFittingMemory []*api.Pod) {
	totalMilliCPU := capacity.Cpu().MilliValue()
	totalMemory := capacity.Memory().Value()
	milliCPURequested := int64(0)
	memoryRequested := int64(0)
	for _, pod := range pods {
		podRequest := getResourceRequest(pod)
		fitsCPU := totalMilliCPU == 0 || (totalMilliCPU-milliCPURequested) >= podRequest.milliCPU
		fitsMemory := totalMemory == 0 || (totalMemory-memoryRequested) >= podRequest.memory
		if !fitsCPU {
			// the pod doesn't fit due to CPU request
			notFittingCPU = append(notFittingCPU, pod)
			continue
		}
		if !fitsMemory {
			// the pod doesn't fit due to Memory request
			notFittingMemory = append(notFittingMemory, pod)
			continue
		}
		// the pod fits
		milliCPURequested += podRequest.milliCPU
		memoryRequested += podRequest.memory
		fitting = append(fitting, pod)
	}
	return
}

func podName(pod *api.Pod) string {
	return pod.Namespace + "/" + pod.Name
}

// PodFitsResources calculates fit based on requested, rather than used resources
func (r *ResourceFit) PodFitsResources(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error) {
	podRequest := getResourceRequest(pod)
	info, err := r.info.GetNodeInfo(node)
	if err != nil {
		return false, err
	}
	if podRequest.milliCPU == 0 && podRequest.memory == 0 {
		return int64(len(existingPods)) < info.Status.Capacity.Pods().Value(), nil
	}
	pods := []*api.Pod{}
	copy(pods, existingPods)
	pods = append(existingPods, pod)
	_, exceedingCPU, exceedingMemory := CheckPodsExceedingFreeResources(pods, info.Status.Capacity)
	if int64(len(pods)) > info.Status.Capacity.Pods().Value() {
		glog.V(10).Infof("Cannot schedule Pod %v, because Node %v is full, running %v out of %v Pods.", podName(pod), node, len(pods)-1, info.Status.Capacity.Pods().Value())
		FailedResourceType = "PodExceedsMaxPodNumber"
		return false, nil
	}
	if len(exceedingCPU) > 0 {
		glog.V(10).Infof("Cannot schedule Pod %v, because Node %v does not have sufficient CPU", podName(pod), node)
		FailedResourceType = "PodExceedsFreeCPU"
		return false, nil
	}
	if len(exceedingMemory) > 0 {
		glog.V(10).Infof("Cannot schedule Pod %v, because Node %v does not have sufficient Memory", podName(pod), node)
		FailedResourceType = "PodExceedsFreeMemory"
		return false, nil
	}
	glog.V(10).Infof("Schedule Pod %v on Node %v is allowed, Node is running only %v out of %v Pods.", podName(pod), node, len(pods)-1, info.Status.Capacity.Pods().Value())
	return true, nil
}

func NewResourceFitPredicate(info NodeInfo) algorithm.FitPredicate {
	fit := &ResourceFit{
		info: info,
	}
	return fit.PodFitsResources
}

func NewSelectorMatchPredicate(info NodeInfo) algorithm.FitPredicate {
	selector := &NodeSelector{
		info: info,
	}
	return selector.PodSelectorMatches
}

func PodMatchesNodeLabels(pod *api.Pod, node *api.Node) bool {
	if len(pod.Spec.NodeSelector) == 0 {
		return true
	}
	selector := labels.SelectorFromSet(pod.Spec.NodeSelector)
	return selector.Matches(labels.Set(node.Labels))
}

type NodeSelector struct {
	info NodeInfo
}

func (n *NodeSelector) PodSelectorMatches(pod *api.Pod, existingPods []*api.Pod, nodeID string) (bool, error) {
	node, err := n.info.GetNodeInfo(nodeID)
	if err != nil {
		return false, err
	}
	return PodMatchesNodeLabels(pod, node), nil
}

func PodFitsHost(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error) {
	if len(pod.Spec.NodeName) == 0 {
		return true, nil
	}
	return pod.Spec.NodeName == node, nil
}

type NodeLabelChecker struct {
	info     NodeInfo
	labels   []string
	presence bool
}

func NewNodeLabelPredicate(info NodeInfo, labels []string, presence bool) algorithm.FitPredicate {
	labelChecker := &NodeLabelChecker{
		info:     info,
		labels:   labels,
		presence: presence,
	}
	return labelChecker.CheckNodeLabelPresence
}

// CheckNodeLabelPresence checks whether all of the specified labels exists on a node or not, regardless of their value
// If "presence" is false, then returns false if any of the requested labels matches any of the node's labels,
// otherwise returns true.
// If "presence" is true, then returns false if any of the requested labels does not match any of the node's labels,
// otherwise returns true.
//
// Consider the cases where the nodes are placed in regions/zones/racks and these are identified by labels
// In some cases, it is required that only nodes that are part of ANY of the defined regions/zones/racks be selected
//
// Alternately, eliminating nodes that have a certain label, regardless of value, is also useful
// A node may have a label with "retiring" as key and the date as the value
// and it may be desirable to avoid scheduling new pods on this node
func (n *NodeLabelChecker) CheckNodeLabelPresence(pod *api.Pod, existingPods []*api.Pod, nodeID string) (bool, error) {
	var exists bool
	node, err := n.info.GetNodeInfo(nodeID)
	if err != nil {
		return false, err
	}
	nodeLabels := labels.Set(node.Labels)
	for _, label := range n.labels {
		exists = nodeLabels.Has(label)
		if (exists && !n.presence) || (!exists && n.presence) {
			return false, nil
		}
	}
	return true, nil
}

type ServiceAffinity struct {
	podLister     algorithm.PodLister
	serviceLister algorithm.ServiceLister
	nodeInfo      NodeInfo
	labels        []string
}

func NewServiceAffinityPredicate(podLister algorithm.PodLister, serviceLister algorithm.ServiceLister, nodeInfo NodeInfo, labels []string) algorithm.FitPredicate {
	affinity := &ServiceAffinity{
		podLister:     podLister,
		serviceLister: serviceLister,
		nodeInfo:      nodeInfo,
		labels:        labels,
	}
	return affinity.CheckServiceAffinity
}

// CheckServiceAffinity ensures that only the nodes that match the specified labels are considered for scheduling.
// The set of labels to be considered are provided to the struct (ServiceAffinity).
// The pod is checked for the labels and any missing labels are then checked in the node
// that hosts the service pods (peers) for the given pod.
//
// We add an implicit selector requiring some particular value V for label L to a pod, if:
// - L is listed in the ServiceAffinity object that is passed into the function
// - the pod does not have any NodeSelector for L
// - some other pod from the same service is already scheduled onto a node that has value V for label L
func (s *ServiceAffinity) CheckServiceAffinity(pod *api.Pod, existingPods []*api.Pod, nodeID string) (bool, error) {
	var affinitySelector labels.Selector

	// check if the pod being scheduled has the affinity labels specified in its NodeSelector
	affinityLabels := map[string]string{}
	nodeSelector := labels.Set(pod.Spec.NodeSelector)
	labelsExist := true
	for _, l := range s.labels {
		if nodeSelector.Has(l) {
			affinityLabels[l] = nodeSelector.Get(l)
		} else {
			// the current pod does not specify all the labels, look in the existing service pods
			labelsExist = false
		}
	}

	// skip looking at other pods in the service if the current pod defines all the required affinity labels
	if !labelsExist {
		services, err := s.serviceLister.GetPodServices(pod)
		if err == nil {
			// just use the first service and get the other pods within the service
			// TODO: a separate predicate can be created that tries to handle all services for the pod
			selector := labels.SelectorFromSet(services[0].Spec.Selector)
			servicePods, err := s.podLister.List(selector)
			if err != nil {
				return false, err
			}
			// consider only the pods that belong to the same namespace
			nsServicePods := []*api.Pod{}
			for _, nsPod := range servicePods {
				if nsPod.Namespace == pod.Namespace {
					nsServicePods = append(nsServicePods, nsPod)
				}
			}
			if len(nsServicePods) > 0 {
				// consider any service pod and fetch the node its hosted on
				otherNode, err := s.nodeInfo.GetNodeInfo(nsServicePods[0].Spec.NodeName)
				if err != nil {
					return false, err
				}
				for _, l := range s.labels {
					// If the pod being scheduled has the label value specified, do not override it
					if _, exists := affinityLabels[l]; exists {
						continue
					}
					if labels.Set(otherNode.Labels).Has(l) {
						affinityLabels[l] = labels.Set(otherNode.Labels).Get(l)
					}
				}
			}
		}
	}

	// if there are no existing pods in the service, consider all nodes
	if len(affinityLabels) == 0 {
		affinitySelector = labels.Everything()
	} else {
		affinitySelector = labels.Set(affinityLabels).AsSelector()
	}

	node, err := s.nodeInfo.GetNodeInfo(nodeID)
	if err != nil {
		return false, err
	}

	// check if the node matches the selector
	return affinitySelector.Matches(labels.Set(node.Labels)), nil
}

func PodFitsHostPorts(pod *api.Pod, existingPods []*api.Pod, node string) (bool, error) {
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

func getUsedPorts(pods ...*api.Pod) map[int]bool {
	ports := make(map[int]bool)
	for _, pod := range pods {
		for _, container := range pod.Spec.Containers {
			for _, podPort := range container.Ports {
				ports[podPort.HostPort] = true
			}
		}
	}
	return ports
}

func filterNonRunningPods(pods []*api.Pod) []*api.Pod {
	if len(pods) == 0 {
		return pods
	}
	result := []*api.Pod{}
	for _, pod := range pods {
		if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
			continue
		}
		result = append(result, pod)
	}
	return result
}

// MapPodsToMachines obtains a list of pods and pivots that list into a map where the keys are host names
// and the values are the list of pods running on that host.
func MapPodsToMachines(lister algorithm.PodLister) (map[string][]*api.Pod, error) {
	machineToPods := map[string][]*api.Pod{}
	// TODO: perform more targeted query...
	pods, err := lister.List(labels.Everything())
	if err != nil {
		return map[string][]*api.Pod{}, err
	}
	pods = filterNonRunningPods(pods)
	for _, scheduledPod := range pods {
		host := scheduledPod.Spec.NodeName
		machineToPods[host] = append(machineToPods[host], scheduledPod)
	}
	return machineToPods, nil
}
