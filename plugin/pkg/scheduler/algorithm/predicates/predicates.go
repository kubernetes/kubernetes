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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type NodeInfo interface {
	GetNodeInfo(nodeID string) (*api.Node, error)
}

type PersistentVolumeInfo interface {
	GetPersistentVolumeInfo(pvID string) (*api.PersistentVolume, error)
}

type PersistentVolumeClaimInfo interface {
	GetPersistentVolumeClaimInfo(namespace string, pvcID string) (*api.PersistentVolumeClaim, error)
}

type CachedNodeInfo struct {
	*cache.StoreToNodeLister
}

// GetNodeInfo returns cached data for the node 'id'.
func (c *CachedNodeInfo) GetNodeInfo(id string) (*api.Node, error) {
	node, exists, err := c.Get(&api.Node{ObjectMeta: api.ObjectMeta{Name: id}})

	if err != nil {
		return nil, fmt.Errorf("error retrieving node '%v' from cache: %v", id, err)
	}

	if !exists {
		return nil, fmt.Errorf("node '%v' is not in cache", id)
	}

	return node.(*api.Node), nil
}

func isVolumeConflict(volume api.Volume, pod *api.Pod) bool {
	// fast path if there is no conflict checking targets.
	if volume.GCEPersistentDisk == nil && volume.AWSElasticBlockStore == nil && volume.RBD == nil {
		return false
	}

	for _, existingVolume := range pod.Spec.Volumes {
		// Same GCE disk mounted by multiple pods conflicts unless all pods mount it read-only.
		if volume.GCEPersistentDisk != nil && existingVolume.GCEPersistentDisk != nil {
			disk, existingDisk := volume.GCEPersistentDisk, existingVolume.GCEPersistentDisk
			if disk.PDName == existingDisk.PDName && !(disk.ReadOnly && existingDisk.ReadOnly) {
				return true
			}
		}

		if volume.AWSElasticBlockStore != nil && existingVolume.AWSElasticBlockStore != nil {
			if volume.AWSElasticBlockStore.VolumeID == existingVolume.AWSElasticBlockStore.VolumeID {
				return true
			}
		}

		if volume.RBD != nil && existingVolume.RBD != nil {
			mon, pool, image := volume.RBD.CephMonitors, volume.RBD.RBDPool, volume.RBD.RBDImage
			emon, epool, eimage := existingVolume.RBD.CephMonitors, existingVolume.RBD.RBDPool, existingVolume.RBD.RBDImage
			if haveSame(mon, emon) && pool == epool && image == eimage {
				return true
			}
		}
	}

	return false
}

// NoDiskConflict evaluates if a pod can fit due to the volumes it requests, and those that
// are already mounted. If there is already a volume mounted on that node, another pod that uses the same volume
// can't be scheduled there.
// This is GCE, Amazon EBS, and Ceph RBD specific for now:
// - GCE PD allows multiple mounts as long as they're all read-only
// - AWS EBS forbids any two pods mounting the same volume ID
// - Ceph RBD forbids if any two pods share at least same monitor, and match pool and image.
// TODO: migrate this into some per-volume specific code?
func NoDiskConflict(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	for _, v := range pod.Spec.Volumes {
		for _, ev := range nodeInfo.Pods() {
			if isVolumeConflict(v, ev) {
				return false, ErrDiskConflict
			}
		}
	}
	return true, nil
}

type MaxPDVolumeCountChecker struct {
	filter     VolumeFilter
	maxVolumes int
	pvInfo     PersistentVolumeInfo
	pvcInfo    PersistentVolumeClaimInfo
}

// VolumeFilter contains information on how to filter PD Volumes when checking PD Volume caps
type VolumeFilter struct {
	// Filter normal volumes
	FilterVolume           func(vol *api.Volume) (id string, relevant bool)
	FilterPersistentVolume func(pv *api.PersistentVolume) (id string, relevant bool)
}

// NewMaxPDVolumeCountPredicate creates a predicate which evaluates whether a pod can fit based on the
// number of volumes which match a filter that it requests, and those that are already present.  The
// maximum number is configurable to accommodate different systems.
//
// The predicate looks for both volumes used directly, as well as PVC volumes that are backed by relevant volume
// types, counts the number of unique volumes, and rejects the new pod if it would place the total count over
// the maximum.
func NewMaxPDVolumeCountPredicate(filter VolumeFilter, maxVolumes int, pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo) algorithm.FitPredicate {
	c := &MaxPDVolumeCountChecker{
		filter:     filter,
		maxVolumes: maxVolumes,
		pvInfo:     pvInfo,
		pvcInfo:    pvcInfo,
	}

	return c.predicate
}

func (c *MaxPDVolumeCountChecker) filterVolumes(volumes []api.Volume, namespace string, filteredVolumes map[string]bool) error {
	for _, vol := range volumes {
		if id, ok := c.filter.FilterVolume(&vol); ok {
			filteredVolumes[id] = true
		} else if vol.PersistentVolumeClaim != nil {
			pvcName := vol.PersistentVolumeClaim.ClaimName
			if pvcName == "" {
				return fmt.Errorf("PersistentVolumeClaim had no name: %q", pvcName)
			}
			pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(namespace, pvcName)
			if err != nil {
				return err
			}

			pvName := pvc.Spec.VolumeName
			if pvName == "" {
				return fmt.Errorf("PersistentVolumeClaim is not bound: %q", pvcName)
			}

			pv, err := c.pvInfo.GetPersistentVolumeInfo(pvName)
			if err != nil {
				return err
			}

			if id, ok := c.filter.FilterPersistentVolume(pv); ok {
				filteredVolumes[id] = true
			}
		}
	}

	return nil
}

func (c *MaxPDVolumeCountChecker) predicate(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	newVolumes := make(map[string]bool)
	if err := c.filterVolumes(pod.Spec.Volumes, pod.Namespace, newVolumes); err != nil {
		return false, err
	}

	// quick return
	if len(newVolumes) == 0 {
		return true, nil
	}

	// count unique volumes
	existingVolumes := make(map[string]bool)
	for _, existingPod := range nodeInfo.Pods() {
		if err := c.filterVolumes(existingPod.Spec.Volumes, existingPod.Namespace, existingVolumes); err != nil {
			return false, err
		}
	}
	numExistingVolumes := len(existingVolumes)

	// filter out already-mounted volumes
	for k := range existingVolumes {
		if _, ok := newVolumes[k]; ok {
			delete(newVolumes, k)
		}
	}

	numNewVolumes := len(newVolumes)

	if numExistingVolumes+numNewVolumes > c.maxVolumes {
		// violates MaxEBSVolumeCount or MaxGCEPDVolumeCount
		return false, ErrMaxVolumeCountExceeded
	}

	return true, nil
}

// EBSVolumeFilter is a VolumeFilter for filtering AWS ElasticBlockStore Volumes
var EBSVolumeFilter VolumeFilter = VolumeFilter{
	FilterVolume: func(vol *api.Volume) (string, bool) {
		if vol.AWSElasticBlockStore != nil {
			return vol.AWSElasticBlockStore.VolumeID, true
		}
		return "", false
	},

	FilterPersistentVolume: func(pv *api.PersistentVolume) (string, bool) {
		if pv.Spec.AWSElasticBlockStore != nil {
			return pv.Spec.AWSElasticBlockStore.VolumeID, true
		}
		return "", false
	},
}

// GCEPDVolumeFilter is a VolumeFilter for filtering GCE PersistentDisk Volumes
var GCEPDVolumeFilter VolumeFilter = VolumeFilter{
	FilterVolume: func(vol *api.Volume) (string, bool) {
		if vol.GCEPersistentDisk != nil {
			return vol.GCEPersistentDisk.PDName, true
		}
		return "", false
	},

	FilterPersistentVolume: func(pv *api.PersistentVolume) (string, bool) {
		if pv.Spec.GCEPersistentDisk != nil {
			return pv.Spec.GCEPersistentDisk.PDName, true
		}
		return "", false
	},
}

type VolumeZoneChecker struct {
	pvInfo  PersistentVolumeInfo
	pvcInfo PersistentVolumeClaimInfo
}

// VolumeZonePredicate evaluates if a pod can fit due to the volumes it requests, given
// that some volumes may have zone scheduling constraints.  The requirement is that any
// volume zone-labels must match the equivalent zone-labels on the node.  It is OK for
// the node to have more zone-label constraints (for example, a hypothetical replicated
// volume might allow region-wide access)
//
// Currently this is only supported with PersistentVolumeClaims, and looks to the labels
// only on the bound PersistentVolume.
//
// Working with volumes declared inline in the pod specification (i.e. not
// using a PersistentVolume) is likely to be harder, as it would require
// determining the zone of a volume during scheduling, and that is likely to
// require calling out to the cloud provider.  It seems that we are moving away
// from inline volume declarations anyway.
func NewVolumeZonePredicate(pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo) algorithm.FitPredicate {
	c := &VolumeZoneChecker{
		pvInfo:  pvInfo,
		pvcInfo: pvcInfo,
	}
	return c.predicate
}

func (c *VolumeZoneChecker) predicate(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, fmt.Errorf("node not found")
	}

	nodeConstraints := make(map[string]string)
	for k, v := range node.ObjectMeta.Labels {
		if k != unversioned.LabelZoneFailureDomain && k != unversioned.LabelZoneRegion {
			continue
		}
		nodeConstraints[k] = v
	}

	if len(nodeConstraints) == 0 {
		// The node has no zone constraints, so we're OK to schedule.
		// In practice, when using zones, all nodes must be labeled with zone labels.
		// We want to fast-path this case though.
		return true, nil
	}

	namespace := pod.Namespace

	manifest := &(pod.Spec)
	for i := range manifest.Volumes {
		volume := &manifest.Volumes[i]
		if volume.PersistentVolumeClaim != nil {
			pvcName := volume.PersistentVolumeClaim.ClaimName
			if pvcName == "" {
				return false, fmt.Errorf("PersistentVolumeClaim had no name: %q", pvcName)
			}
			pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(namespace, pvcName)
			if err != nil {
				return false, err
			}

			if pvc == nil {
				return false, fmt.Errorf("PersistentVolumeClaim was not found: %q", pvcName)
			}

			pvName := pvc.Spec.VolumeName
			if pvName == "" {
				return false, fmt.Errorf("PersistentVolumeClaim is not bound: %q", pvcName)
			}

			pv, err := c.pvInfo.GetPersistentVolumeInfo(pvName)
			if err != nil {
				return false, err
			}

			if pv == nil {
				return false, fmt.Errorf("PersistentVolume not found: %q", pvName)
			}

			for k, v := range pv.ObjectMeta.Labels {
				if k != unversioned.LabelZoneFailureDomain && k != unversioned.LabelZoneRegion {
					continue
				}
				nodeV, _ := nodeConstraints[k]
				if v != nodeV {
					glog.V(2).Infof("Won't schedule pod %q onto node %q due to volume %q (mismatch on %q)", pod.Name, node.Name, pvName, k)
					return false, ErrVolumeZoneConflict
				}
			}
		}
	}

	return true, nil
}

type resourceRequest struct {
	milliCPU int64
	memory   int64
}

func getResourceRequest(pod *api.Pod) resourceRequest {
	result := resourceRequest{}
	for _, container := range pod.Spec.Containers {
		requests := container.Resources.Requests
		result.memory += requests.Memory().Value()
		result.milliCPU += requests.Cpu().MilliValue()
	}
	return result
}

func CheckPodsExceedingFreeResources(pods []*api.Pod, allocatable api.ResourceList) (fitting []*api.Pod, notFittingCPU, notFittingMemory []*api.Pod) {
	totalMilliCPU := allocatable.Cpu().MilliValue()
	totalMemory := allocatable.Memory().Value()
	milliCPURequested := int64(0)
	memoryRequested := int64(0)
	for _, pod := range pods {
		podRequest := getResourceRequest(pod)
		fitsCPU := (totalMilliCPU - milliCPURequested) >= podRequest.milliCPU
		fitsMemory := (totalMemory - memoryRequested) >= podRequest.memory
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

func PodFitsResources(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, fmt.Errorf("node not found")
	}
	allocatable := node.Status.Allocatable
	allowedPodNumber := allocatable.Pods().Value()
	if int64(len(nodeInfo.Pods()))+1 > allowedPodNumber {
		return false,
			newInsufficientResourceError(podCountResourceName, 1, int64(len(nodeInfo.Pods())), allowedPodNumber)
	}
	podRequest := getResourceRequest(pod)
	if podRequest.milliCPU == 0 && podRequest.memory == 0 {
		return true, nil
	}

	totalMilliCPU := allocatable.Cpu().MilliValue()
	totalMemory := allocatable.Memory().Value()

	if totalMilliCPU < podRequest.milliCPU+nodeInfo.RequestedResource().MilliCPU {
		return false,
			newInsufficientResourceError(cpuResourceName, podRequest.milliCPU, nodeInfo.RequestedResource().MilliCPU, totalMilliCPU)
	}
	if totalMemory < podRequest.memory+nodeInfo.RequestedResource().Memory {
		return false,
			newInsufficientResourceError(memoryResoureceName, podRequest.memory, nodeInfo.RequestedResource().Memory, totalMemory)
	}
	glog.V(10).Infof("Schedule Pod %+v on Node %+v is allowed, Node is running only %v out of %v Pods.",
		podName(pod), node.Name, len(nodeInfo.Pods()), allowedPodNumber)
	return true, nil
}

// nodeMatchesNodeSelectorTerms checks if a node's labels satisfy a list of node selector terms,
// terms are ORed, and an emtpy a list of terms will match nothing.
func nodeMatchesNodeSelectorTerms(node *api.Node, nodeSelectorTerms []api.NodeSelectorTerm) bool {
	for _, req := range nodeSelectorTerms {
		nodeSelector, err := api.NodeSelectorRequirementsAsSelector(req.MatchExpressions)
		if err != nil {
			glog.V(10).Infof("Failed to parse MatchExpressions: %+v, regarding as not match.", req.MatchExpressions)
			return false
		}
		if nodeSelector.Matches(labels.Set(node.Labels)) {
			return true
		}
	}
	return false
}

// The pod can only schedule onto nodes that satisfy requirements in both NodeAffinity and nodeSelector.
func PodMatchesNodeLabels(pod *api.Pod, node *api.Node) bool {
	// Check if node.Labels match pod.Spec.NodeSelector.
	if len(pod.Spec.NodeSelector) > 0 {
		selector := labels.SelectorFromSet(pod.Spec.NodeSelector)
		if !selector.Matches(labels.Set(node.Labels)) {
			return false
		}
	}

	// Parse required node affinity scheduling requirements
	// and check if the current node match the requirements.
	affinity, err := api.GetAffinityFromPodAnnotations(pod.Annotations)
	if err != nil {
		glog.V(10).Infof("Failed to get Affinity from Pod %+v, err: %+v", podName(pod), err)
		return false
	}

	// 1. nil NodeSelector matches all nodes (i.e. does not filter out any nodes)
	// 2. nil []NodeSelectorTerm (equivalent to non-nil empty NodeSelector) matches no nodes
	// 3. zero-length non-nil []NodeSelectorTerm matches no nodes also, just for simplicity
	// 4. nil []NodeSelectorRequirement (equivalent to non-nil empty NodeSelectorTerm) matches no nodes
	// 5. zero-length non-nil []NodeSelectorRequirement matches no nodes also, just for simplicity
	// 6. non-nil empty NodeSelectorRequirement is not allowed
	nodeAffinityMatches := true
	if affinity.NodeAffinity != nil {
		nodeAffinity := affinity.NodeAffinity
		// if no required NodeAffinity requirements, will do no-op, means select all nodes.
		// TODO: Replace next line with subsequent commented-out line when implement RequiredDuringSchedulingRequiredDuringExecution.
		if nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
			// if nodeAffinity.RequiredDuringSchedulingRequiredDuringExecution == nil && nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
			return true
		}

		// Match node selector for requiredDuringSchedulingRequiredDuringExecution.
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		// if nodeAffinity.RequiredDuringSchedulingRequiredDuringExecution != nil {
		// 	nodeSelectorTerms := nodeAffinity.RequiredDuringSchedulingRequiredDuringExecution.NodeSelectorTerms
		// 	glog.V(10).Infof("Match for RequiredDuringSchedulingRequiredDuringExecution node selector terms %+v", nodeSelectorTerms)
		// 	nodeAffinityMatches = nodeMatchesNodeSelectorTerms(node, nodeSelectorTerms)
		// }

		// Match node selector for requiredDuringSchedulingIgnoredDuringExecution.
		if nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
			nodeSelectorTerms := nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
			glog.V(10).Infof("Match for RequiredDuringSchedulingIgnoredDuringExecution node selector terms %+v", nodeSelectorTerms)
			nodeAffinityMatches = nodeAffinityMatches && nodeMatchesNodeSelectorTerms(node, nodeSelectorTerms)
		}

	}
	return nodeAffinityMatches
}

func PodSelectorMatches(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, fmt.Errorf("node not found")
	}
	if PodMatchesNodeLabels(pod, node) {
		return true, nil
	}
	return false, ErrNodeSelectorNotMatch
}

func PodFitsHost(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	if len(pod.Spec.NodeName) == 0 {
		return true, nil
	}
	node := nodeInfo.Node()
	if node == nil {
		return false, fmt.Errorf("node not found")
	}
	if pod.Spec.NodeName == node.Name {
		return true, nil
	}
	return false, ErrPodNotMatchHostName
}

type NodeLabelChecker struct {
	labels   []string
	presence bool
}

func NewNodeLabelPredicate(labels []string, presence bool) algorithm.FitPredicate {
	labelChecker := &NodeLabelChecker{
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
func (n *NodeLabelChecker) CheckNodeLabelPresence(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, fmt.Errorf("node not found")
	}

	var exists bool
	nodeLabels := labels.Set(node.Labels)
	for _, label := range n.labels {
		exists = nodeLabels.Has(label)
		if (exists && !n.presence) || (!exists && n.presence) {
			return false, ErrNodeLabelPresenceViolated
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
func (s *ServiceAffinity) CheckServiceAffinity(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
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

	node := nodeInfo.Node()
	if node == nil {
		return false, fmt.Errorf("node not found")
	}

	// check if the node matches the selector
	if affinitySelector.Matches(labels.Set(node.Labels)) {
		return true, nil
	}
	return false, ErrServiceAffinityViolated
}

func PodFitsHostPorts(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	wantPorts := getUsedPorts(pod)
	if len(wantPorts) == 0 {
		return true, nil
	}
	existingPorts := getUsedPorts(nodeInfo.Pods()...)
	for wport := range wantPorts {
		if wport == 0 {
			continue
		}
		if existingPorts[wport] {
			return false, ErrPodNotFitsHostPorts
		}
	}
	return true, nil
}

func getUsedPorts(pods ...*api.Pod) map[int]bool {
	// TODO: Aggregate it at the NodeInfo level.
	ports := make(map[int]bool)
	for _, pod := range pods {
		for _, container := range pod.Spec.Containers {
			for _, podPort := range container.Ports {
				// "0" is explicitly ignored in PodFitsHostPorts,
				// which is the only function that uses this value.
				if podPort.HostPort != 0 {
					ports[int(podPort.HostPort)] = true
				}
			}
		}
	}
	return ports
}

// search two arrays and return true if they have at least one common element; return false otherwise
func haveSame(a1, a2 []string) bool {
	for _, val1 := range a1 {
		for _, val2 := range a2 {
			if val1 == val2 {
				return true
			}
		}
	}
	return false
}

func GeneralPredicates(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	fit, err := PodFitsResources(pod, nodeInfo)
	if !fit {
		return fit, err
	}

	fit, err = PodFitsHost(pod, nodeInfo)
	if !fit {
		return fit, err
	}
	fit, err = PodFitsHostPorts(pod, nodeInfo)
	if !fit {
		return fit, err
	}
	fit, err = PodSelectorMatches(pod, nodeInfo)
	if !fit {
		return fit, err
	}
	return true, nil
}

type PodAffinityChecker struct {
	info           NodeInfo
	podLister      algorithm.PodLister
	failureDomains priorityutil.Topologies
}

func NewPodAffinityPredicate(info NodeInfo, podLister algorithm.PodLister, failureDomains []string) algorithm.FitPredicate {
	checker := &PodAffinityChecker{
		info:           info,
		podLister:      podLister,
		failureDomains: priorityutil.Topologies{DefaultKeys: failureDomains},
	}
	return checker.InterPodAffinityMatches
}

func (checker *PodAffinityChecker) InterPodAffinityMatches(pod *api.Pod, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, fmt.Errorf("node not found")
	}
	allPods, err := checker.podLister.List(labels.Everything())
	if err != nil {
		return false, err
	}
	if checker.NodeMatchPodAffinityAntiAffinity(pod, allPods, node) {
		return true, nil
	}
	return false, ErrPodAffinityNotMatch
}

// AnyPodMatchesPodAffinityTerm checks if any of given pods can match the specific podAffinityTerm.
func (checker *PodAffinityChecker) AnyPodMatchesPodAffinityTerm(pod *api.Pod, allPods []*api.Pod, node *api.Node, podAffinityTerm api.PodAffinityTerm) (bool, error) {
	for _, ep := range allPods {
		match, err := checker.failureDomains.CheckIfPodMatchPodAffinityTerm(ep, pod, podAffinityTerm,
			func(ep *api.Pod) (*api.Node, error) { return checker.info.GetNodeInfo(ep.Spec.NodeName) },
			func(pod *api.Pod) (*api.Node, error) { return node, nil },
		)
		if err != nil || match {
			return match, err
		}
	}
	return false, nil
}

// Checks whether the given node has pods which satisfy all the required pod affinity scheduling rules.
// If node has pods which satisfy all the required pod affinity scheduling rules then return true.
func (checker *PodAffinityChecker) NodeMatchesHardPodAffinity(pod *api.Pod, allPods []*api.Pod, node *api.Node, podAffinity *api.PodAffinity) bool {
	var podAffinityTerms []api.PodAffinityTerm
	if len(podAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 {
		podAffinityTerms = podAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	}
	// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
	//if len(podAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
	//	podAffinityTerms = append(podAffinityTerms, podAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
	//}

	for _, podAffinityTerm := range podAffinityTerms {
		podAffinityTermMatches, err := checker.AnyPodMatchesPodAffinityTerm(pod, allPods, node, podAffinityTerm)
		if err != nil {
			glog.V(10).Infof("Cannot schedule pod %+v onto node %v, an error ocurred when checking existing pods on the node for PodAffinityTerm %v err: %v",
				podName(pod), node.Name, podAffinityTerm, err)
			return false
		}

		if !podAffinityTermMatches {
			// TODO: Think about whether this can be simplified once we have controllerRef
			// Check if it is in special case that the requiredDuringScheduling affinity requirement can be disregarded.
			// If the requiredDuringScheduling affinity requirement matches a pod's own labels and namespace, and there are no other such pods
			// anywhere, then disregard the requirement.
			// This allows rules like "schedule all of the pods of this collection to the same zone" to not block forever
			// because the first pod of the collection can't be scheduled.
			names := priorityutil.GetNamespacesFromPodAffinityTerm(pod, podAffinityTerm)
			labelSelector, err := unversioned.LabelSelectorAsSelector(podAffinityTerm.LabelSelector)
			if err != nil || !names.Has(pod.Namespace) || !labelSelector.Matches(labels.Set(pod.Labels)) {
				glog.V(10).Infof("Cannot schedule pod %+v onto node %v, because none of the existing pods on this node satisfy the PodAffinityTerm %v, err: %+v",
					podName(pod), node.Name, podAffinityTerm, err)
				return false
			}

			// the affinity is to put the pod together with other pods from its same service or controller
			filteredPods := priorityutil.FilterPodsByNameSpaces(names, allPods)
			for _, filteredPod := range filteredPods {
				// if found an existing pod from same service or RC,
				// the affinity scheduling rules cannot be disregarded.
				if labelSelector.Matches(labels.Set(filteredPod.Labels)) {
					glog.V(10).Infof("Cannot schedule pod %+v onto node %v, because none of the existing pods on this node satisfy the PodAffinityTerm %v",
						podName(pod), node.Name, podAffinityTerm)
					return false
				}
			}
		}
	}
	// all the required pod affinity scheduling rules satisfied
	glog.V(10).Infof("All the required pod affinity scheduling rules are satisfied for Pod %+v, on node %v", podName(pod), node.Name)
	return true
}

// Checks whether the given node has pods which satisfy all the
// required pod anti-affinity scheduling rules.
// Also checks whether putting the pod onto the node would break
// any anti-affinity scheduling rules indicated by existing pods.
// If node has pods which satisfy all the required pod anti-affinity
// scheduling rules and scheduling the pod onto the node won't
// break any existing pods' anti-affinity rules, then return true.
func (checker *PodAffinityChecker) NodeMatchesHardPodAntiAffinity(pod *api.Pod, allPods []*api.Pod, node *api.Node, podAntiAffinity *api.PodAntiAffinity) bool {
	var podAntiAffinityTerms []api.PodAffinityTerm
	if len(podAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 {
		podAntiAffinityTerms = podAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution
	}
	// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
	//if len(podAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
	//	podAntiAffinityTerms = append(podAntiAffinityTerms, podAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
	//}

	// foreach element podAntiAffinityTerm of podAntiAffinityTerms
	// if the pod matches the term (breaks the anti-affinity),
	// don't schedule the pod onto this node.
	for _, podAntiAffinityTerm := range podAntiAffinityTerms {
		podAntiAffinityTermMatches, err := checker.AnyPodMatchesPodAffinityTerm(pod, allPods, node, podAntiAffinityTerm)
		if err != nil || podAntiAffinityTermMatches == true {
			glog.V(10).Infof("Cannot schedule pod %+v onto node %v, because not all the existing pods on this node satisfy the PodAntiAffinityTerm %v, err: %v",
				podName(pod), node.Name, podAntiAffinityTerm, err)
			return false
		}
	}

	// Check if scheduling the pod onto this node would break
	// any anti-affinity rules indicated by the existing pods on the node.
	// If it would break, system should not schedule pod onto this node.
	for _, ep := range allPods {
		epAffinity, err := api.GetAffinityFromPodAnnotations(ep.Annotations)
		if err != nil {
			glog.V(10).Infof("Failed to get Affinity from Pod %+v, err: %+v", podName(pod), err)
			return false
		}
		if epAffinity.PodAntiAffinity != nil {
			var epAntiAffinityTerms []api.PodAffinityTerm
			if len(epAffinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 {
				epAntiAffinityTerms = epAffinity.PodAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution
			}
			// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
			//if len(epAffinity.PodAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
			//	epAntiAffinityTerms = append(epAntiAffinityTerms, epAffinity.PodAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
			//}

			for _, epAntiAffinityTerm := range epAntiAffinityTerms {
				labelSelector, err := unversioned.LabelSelectorAsSelector(epAntiAffinityTerm.LabelSelector)
				if err != nil {
					glog.V(10).Infof("Failed to get label selector from anti-affinityterm %+v of existing pod %+v, err: %+v", epAntiAffinityTerm, podName(pod), err)
					return false
				}

				names := priorityutil.GetNamespacesFromPodAffinityTerm(ep, epAntiAffinityTerm)
				if (len(names) == 0 || names.Has(pod.Namespace)) && labelSelector.Matches(labels.Set(pod.Labels)) {
					epNode, err := checker.info.GetNodeInfo(ep.Spec.NodeName)
					if err != nil || checker.failureDomains.NodesHaveSameTopologyKey(node, epNode, epAntiAffinityTerm.TopologyKey) {
						glog.V(10).Infof("Cannot schedule Pod %+v, onto node %v because the pod would break the PodAntiAffinityTerm %+v, of existing pod %+v, err: %v",
							podName(pod), node.Name, epAntiAffinityTerm, podName(ep), err)
						return false
					}
				}
			}
		}
	}
	// all the required pod anti-affinity scheduling rules are satisfied
	glog.V(10).Infof("Can schedule Pod %+v, on node %v because all the required pod anti-affinity scheduling rules are satisfied", podName(pod), node.Name)
	return true
}

// NodeMatchPodAffinityAntiAffinity checks if the node matches
// the requiredDuringScheduling affinity/anti-affinity rules indicated by the pod.
func (checker *PodAffinityChecker) NodeMatchPodAffinityAntiAffinity(pod *api.Pod, allPods []*api.Pod, node *api.Node) bool {
	// Parse required affinity scheduling rules.
	affinity, err := api.GetAffinityFromPodAnnotations(pod.Annotations)
	if err != nil {
		glog.V(10).Infof("Failed to get Affinity from Pod %+v, err: %+v", podName(pod), err)
		return false
	}

	// check if the current node match the inter-pod affinity scheduling rules.
	if affinity.PodAffinity != nil {
		if !checker.NodeMatchesHardPodAffinity(pod, allPods, node, affinity.PodAffinity) {
			return false
		}
	}

	// check if the current node match the inter-pod anti-affinity scheduling rules.
	if affinity.PodAntiAffinity != nil {
		if !checker.NodeMatchesHardPodAntiAffinity(pod, allPods, node, affinity.PodAntiAffinity) {
			return false
		}
	}
	return true
}
