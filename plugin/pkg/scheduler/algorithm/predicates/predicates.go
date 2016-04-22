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
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/unversioned"
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
func NoDiskConflict(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
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

func (c *MaxPDVolumeCountChecker) predicate(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
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
	nodeInfo NodeInfo
	pvInfo   PersistentVolumeInfo
	pvcInfo  PersistentVolumeClaimInfo
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
func NewVolumeZonePredicate(nodeInfo NodeInfo, pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo) algorithm.FitPredicate {
	c := &VolumeZoneChecker{
		nodeInfo: nodeInfo,
		pvInfo:   pvInfo,
		pvcInfo:  pvcInfo,
	}
	return c.predicate
}

func (c *VolumeZoneChecker) predicate(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node, err := c.nodeInfo.GetNodeInfo(nodeName)
	if err != nil {
		return false, err
	}
	if node == nil {
		return false, fmt.Errorf("node not found: %q", nodeName)
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
					glog.V(2).Infof("Won't schedule pod %q onto node %q due to volume %q (mismatch on %q)", pod.Name, nodeName, pvName, k)
					return false, ErrVolumeZoneConflict
				}
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

func podFitsResourcesInternal(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo, info *api.Node) (bool, error) {
	allocatable := info.Status.Allocatable
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
		podName(pod), nodeName, len(nodeInfo.Pods()), allowedPodNumber)
	return true, nil
}

func (r *NodeStatus) PodFitsResources(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	info, err := r.info.GetNodeInfo(nodeName)
	if err != nil {
		return false, err
	}
	return podFitsResourcesInternal(pod, nodeName, nodeInfo, info)
}

func NewResourceFitPredicate(info NodeInfo) algorithm.FitPredicate {
	fit := &NodeStatus{
		info: info,
	}
	return fit.PodFitsResources
}

func NewSelectorMatchPredicate(info NodeInfo) algorithm.FitPredicate {
	selector := &NodeStatus{
		info: info,
	}
	return selector.PodSelectorMatches
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

type NodeSelector struct {
	info NodeInfo
}

func (n *NodeStatus) PodSelectorMatches(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node, err := n.info.GetNodeInfo(nodeName)
	if err != nil {
		return false, err
	}
	if PodMatchesNodeLabels(pod, node) {
		return true, nil
	}
	return false, ErrNodeSelectorNotMatch
}

func PodFitsHost(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	if len(pod.Spec.NodeName) == 0 {
		return true, nil
	}
	if pod.Spec.NodeName == nodeName {
		return true, nil
	}
	return false, ErrPodNotMatchHostName
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
func (n *NodeLabelChecker) CheckNodeLabelPresence(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	var exists bool
	node, err := n.info.GetNodeInfo(nodeName)
	if err != nil {
		return false, err
	}
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
func (s *ServiceAffinity) CheckServiceAffinity(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
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

	node, err := s.nodeInfo.GetNodeInfo(nodeName)
	if err != nil {
		return false, err
	}

	// check if the node matches the selector
	if affinitySelector.Matches(labels.Set(node.Labels)) {
		return true, nil
	}
	return false, ErrServiceAffinityViolated
}

func PodFitsHostPorts(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
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

type NodeStatus struct {
	info NodeInfo
}

func GeneralPredicates(info NodeInfo) algorithm.FitPredicate {
	node := &NodeStatus{
		info: info,
	}
	return node.SchedulerGeneralPredicates
}

func (n *NodeStatus) SchedulerGeneralPredicates(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo) (bool, error) {
	node, err := n.info.GetNodeInfo(nodeName)
	if err != nil {
		return false, err
	}
	return RunGeneralPredicates(pod, nodeName, nodeInfo, node)
}

func RunGeneralPredicates(pod *api.Pod, nodeName string, nodeInfo *schedulercache.NodeInfo, node *api.Node) (bool, error) {
	fit, err := podFitsResourcesInternal(pod, nodeName, nodeInfo, node)
	if !fit {
		return fit, err
	}

	fit, err = PodFitsHost(pod, nodeName, nodeInfo)
	if !fit {
		return fit, err
	}
	fit, err = PodFitsHostPorts(pod, nodeName, nodeInfo)
	if !fit {
		return fit, err
	}
	if !PodMatchesNodeLabels(pod, node) {
		return false, ErrNodeSelectorNotMatch
	}
	return true, nil
}
