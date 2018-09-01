/*
Copyright 2015 The Kubernetes Authors.

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

package cache

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

var (
	emptyResource = Resource{}
	generation    int64
)

// NodeInfo is node level aggregated information.
type NodeInfo struct {
	// Overall node information.
	node *v1.Node

	pods             []*v1.Pod
	podsWithAffinity []*v1.Pod
	usedPorts        util.HostPortInfo

	// Total requested resource of all pods on this node.
	// It includes assumed pods which scheduler sends binding to apiserver but
	// didn't get it as scheduled yet.
	requestedResource *Resource
	nonzeroRequest    *Resource
	// We store allocatedResources (which is Node.Status.Allocatable.*) explicitly
	// as int64, to avoid conversions and accessing map.
	allocatableResource *Resource

	// Cached taints of the node for faster lookup.
	taints    []v1.Taint
	taintsErr error

	// imageStates holds the entry of an image if and only if this image is on the node. The entry can be used for
	// checking an image's existence and advanced usage (e.g., image locality scheduling policy) based on the image
	// state information.
	imageStates map[string]*ImageStateSummary

	// TransientInfo holds the information pertaining to a scheduling cycle. This will be destructed at the end of
	// scheduling cycle.
	// TODO: @ravig. Remove this once we have a clear approach for message passing across predicates and priorities.
	TransientInfo *transientSchedulerInfo

	// Cached conditions of node for faster lookup.
	memoryPressureCondition v1.ConditionStatus
	diskPressureCondition   v1.ConditionStatus
	pidPressureCondition    v1.ConditionStatus

	// Whenever NodeInfo changes, generation is bumped.
	// This is used to avoid cloning it if the object didn't change.
	generation int64
}

//initializeNodeTransientInfo initializes transient information pertaining to node.
func initializeNodeTransientInfo() nodeTransientInfo {
	return nodeTransientInfo{AllocatableVolumesCount: 0, RequestedVolumes: 0}
}

// nextGeneration: Let's make sure history never forgets the name...
// Increments the generation number monotonically ensuring that generation numbers never collide.
// Collision of the generation numbers would be particularly problematic if a node was deleted and
// added back with the same name. See issue#63262.
func nextGeneration() int64 {
	return atomic.AddInt64(&generation, 1)
}

// nodeTransientInfo contains transient node information while scheduling.
type nodeTransientInfo struct {
	// AllocatableVolumesCount contains number of volumes that could be attached to node.
	AllocatableVolumesCount int
	// Requested number of volumes on a particular node.
	RequestedVolumes int
}

// transientSchedulerInfo is a transient structure which is destructed at the end of each scheduling cycle.
// It consists of items that are valid for a scheduling cycle and is used for message passing across predicates and
// priorities. Some examples which could be used as fields are number of volumes being used on node, current utilization
// on node etc.
// IMPORTANT NOTE: Make sure that each field in this structure is documented along with usage. Expand this structure
// only when absolutely needed as this data structure will be created and destroyed during every scheduling cycle.
type transientSchedulerInfo struct {
	TransientLock sync.Mutex
	// NodeTransInfo holds the information related to nodeTransientInformation. NodeName is the key here.
	TransNodeInfo nodeTransientInfo
}

// newTransientSchedulerInfo returns a new scheduler transient structure with initialized values.
func newTransientSchedulerInfo() *transientSchedulerInfo {
	tsi := &transientSchedulerInfo{
		TransNodeInfo: initializeNodeTransientInfo(),
	}
	return tsi
}

// resetTransientSchedulerInfo resets the transientSchedulerInfo.
func (transientSchedInfo *transientSchedulerInfo) resetTransientSchedulerInfo() {
	transientSchedInfo.TransientLock.Lock()
	defer transientSchedInfo.TransientLock.Unlock()
	// Reset TransientNodeInfo.
	transientSchedInfo.TransNodeInfo.AllocatableVolumesCount = 0
	transientSchedInfo.TransNodeInfo.RequestedVolumes = 0
}

// Resource is a collection of compute resource.
type Resource struct {
	MilliCPU         int64
	Memory           int64
	EphemeralStorage int64
	// We store allowedPodNumber (which is Node.Status.Allocatable.Pods().Value())
	// explicitly as int, to avoid conversions and improve performance.
	AllowedPodNumber int
	// ScalarResources
	ScalarResources map[v1.ResourceName]int64
}

// NewResource creates a Resource from ResourceList
func NewResource(rl v1.ResourceList) *Resource {
	r := &Resource{}
	r.Add(rl)
	return r
}

// Add adds ResourceList into Resource.
func (r *Resource) Add(rl v1.ResourceList) {
	if r == nil {
		return
	}

	for rName, rQuant := range rl {
		switch rName {
		case v1.ResourceCPU:
			r.MilliCPU += rQuant.MilliValue()
		case v1.ResourceMemory:
			r.Memory += rQuant.Value()
		case v1.ResourcePods:
			r.AllowedPodNumber += int(rQuant.Value())
		case v1.ResourceEphemeralStorage:
			r.EphemeralStorage += rQuant.Value()
		default:
			if v1helper.IsScalarResourceName(rName) {
				r.AddScalar(rName, rQuant.Value())
			}
		}
	}
}

// ResourceList returns a resource list of this resource.
func (r *Resource) ResourceList() v1.ResourceList {
	result := v1.ResourceList{
		v1.ResourceCPU:              *resource.NewMilliQuantity(r.MilliCPU, resource.DecimalSI),
		v1.ResourceMemory:           *resource.NewQuantity(r.Memory, resource.BinarySI),
		v1.ResourcePods:             *resource.NewQuantity(int64(r.AllowedPodNumber), resource.BinarySI),
		v1.ResourceEphemeralStorage: *resource.NewQuantity(r.EphemeralStorage, resource.BinarySI),
	}
	for rName, rQuant := range r.ScalarResources {
		if v1helper.IsHugePageResourceName(rName) {
			result[rName] = *resource.NewQuantity(rQuant, resource.BinarySI)
		} else {
			result[rName] = *resource.NewQuantity(rQuant, resource.DecimalSI)
		}
	}
	return result
}

// Clone returns a copy of this resource.
func (r *Resource) Clone() *Resource {
	res := &Resource{
		MilliCPU:         r.MilliCPU,
		Memory:           r.Memory,
		AllowedPodNumber: r.AllowedPodNumber,
		EphemeralStorage: r.EphemeralStorage,
	}
	if r.ScalarResources != nil {
		res.ScalarResources = make(map[v1.ResourceName]int64)
		for k, v := range r.ScalarResources {
			res.ScalarResources[k] = v
		}
	}
	return res
}

// AddScalar adds a resource by a scalar value of this resource.
func (r *Resource) AddScalar(name v1.ResourceName, quantity int64) {
	r.SetScalar(name, r.ScalarResources[name]+quantity)
}

// SetScalar sets a resource by a scalar value of this resource.
func (r *Resource) SetScalar(name v1.ResourceName, quantity int64) {
	// Lazily allocate scalar resource map.
	if r.ScalarResources == nil {
		r.ScalarResources = map[v1.ResourceName]int64{}
	}
	r.ScalarResources[name] = quantity
}

// SetMaxResource compares with ResourceList and takes max value for each Resource.
func (r *Resource) SetMaxResource(rl v1.ResourceList) {
	if r == nil {
		return
	}

	for rName, rQuantity := range rl {
		switch rName {
		case v1.ResourceMemory:
			if mem := rQuantity.Value(); mem > r.Memory {
				r.Memory = mem
			}
		case v1.ResourceCPU:
			if cpu := rQuantity.MilliValue(); cpu > r.MilliCPU {
				r.MilliCPU = cpu
			}
		case v1.ResourceEphemeralStorage:
			if ephemeralStorage := rQuantity.Value(); ephemeralStorage > r.EphemeralStorage {
				r.EphemeralStorage = ephemeralStorage
			}
		default:
			if v1helper.IsScalarResourceName(rName) {
				value := rQuantity.Value()
				if value > r.ScalarResources[rName] {
					r.SetScalar(rName, value)
				}
			}
		}
	}
}

// NewNodeInfo returns a ready to use empty NodeInfo object.
// If any pods are given in arguments, their information will be aggregated in
// the returned object.
func NewNodeInfo(pods ...*v1.Pod) *NodeInfo {
	ni := &NodeInfo{
		requestedResource:   &Resource{},
		nonzeroRequest:      &Resource{},
		allocatableResource: &Resource{},
		TransientInfo:       newTransientSchedulerInfo(),
		generation:          nextGeneration(),
		usedPorts:           make(util.HostPortInfo),
		imageStates:         make(map[string]*ImageStateSummary),
	}
	for _, pod := range pods {
		ni.AddPod(pod)
	}
	return ni
}

// Node returns overall information about this node.
func (n *NodeInfo) Node() *v1.Node {
	if n == nil {
		return nil
	}
	return n.node
}

// Pods return all pods scheduled (including assumed to be) on this node.
func (n *NodeInfo) Pods() []*v1.Pod {
	if n == nil {
		return nil
	}
	return n.pods
}

// UsedPorts returns used ports on this node.
func (n *NodeInfo) UsedPorts() util.HostPortInfo {
	if n == nil {
		return nil
	}
	return n.usedPorts
}

// ImageStates returns the state information of all images.
func (n *NodeInfo) ImageStates() map[string]*ImageStateSummary {
	if n == nil {
		return nil
	}
	return n.imageStates
}

// PodsWithAffinity return all pods with (anti)affinity constraints on this node.
func (n *NodeInfo) PodsWithAffinity() []*v1.Pod {
	if n == nil {
		return nil
	}
	return n.podsWithAffinity
}

// AllowedPodNumber returns the number of the allowed pods on this node.
func (n *NodeInfo) AllowedPodNumber() int {
	if n == nil || n.allocatableResource == nil {
		return 0
	}
	return n.allocatableResource.AllowedPodNumber
}

// Taints returns the taints list on this node.
func (n *NodeInfo) Taints() ([]v1.Taint, error) {
	if n == nil {
		return nil, nil
	}
	return n.taints, n.taintsErr
}

// MemoryPressureCondition returns the memory pressure condition status on this node.
func (n *NodeInfo) MemoryPressureCondition() v1.ConditionStatus {
	if n == nil {
		return v1.ConditionUnknown
	}
	return n.memoryPressureCondition
}

// DiskPressureCondition returns the disk pressure condition status on this node.
func (n *NodeInfo) DiskPressureCondition() v1.ConditionStatus {
	if n == nil {
		return v1.ConditionUnknown
	}
	return n.diskPressureCondition
}

// PIDPressureCondition returns the pid pressure condition status on this node.
func (n *NodeInfo) PIDPressureCondition() v1.ConditionStatus {
	if n == nil {
		return v1.ConditionUnknown
	}
	return n.pidPressureCondition
}

// RequestedResource returns aggregated resource request of pods on this node.
func (n *NodeInfo) RequestedResource() Resource {
	if n == nil {
		return emptyResource
	}
	return *n.requestedResource
}

// NonZeroRequest returns aggregated nonzero resource request of pods on this node.
func (n *NodeInfo) NonZeroRequest() Resource {
	if n == nil {
		return emptyResource
	}
	return *n.nonzeroRequest
}

// AllocatableResource returns allocatable resources on a given node.
func (n *NodeInfo) AllocatableResource() Resource {
	if n == nil {
		return emptyResource
	}
	return *n.allocatableResource
}

// SetAllocatableResource sets the allocatableResource information of given node.
func (n *NodeInfo) SetAllocatableResource(allocatableResource *Resource) {
	n.allocatableResource = allocatableResource
	n.generation = nextGeneration()
}

// Clone returns a copy of this node.
func (n *NodeInfo) Clone() *NodeInfo {
	clone := &NodeInfo{
		node:                    n.node,
		requestedResource:       n.requestedResource.Clone(),
		nonzeroRequest:          n.nonzeroRequest.Clone(),
		allocatableResource:     n.allocatableResource.Clone(),
		taintsErr:               n.taintsErr,
		TransientInfo:           n.TransientInfo,
		memoryPressureCondition: n.memoryPressureCondition,
		diskPressureCondition:   n.diskPressureCondition,
		pidPressureCondition:    n.pidPressureCondition,
		usedPorts:               make(util.HostPortInfo),
		imageStates:             n.imageStates,
		generation:              n.generation,
	}
	if len(n.pods) > 0 {
		clone.pods = append([]*v1.Pod(nil), n.pods...)
	}
	if len(n.usedPorts) > 0 {
		// util.HostPortInfo is a map-in-map struct
		// make sure it's deep copied
		for ip, portMap := range n.usedPorts {
			clone.usedPorts[ip] = make(map[util.ProtocolPort]struct{})
			for protocolPort, v := range portMap {
				clone.usedPorts[ip][protocolPort] = v
			}
		}
	}
	if len(n.podsWithAffinity) > 0 {
		clone.podsWithAffinity = append([]*v1.Pod(nil), n.podsWithAffinity...)
	}
	if len(n.taints) > 0 {
		clone.taints = append([]v1.Taint(nil), n.taints...)
	}
	return clone
}

// VolumeLimits returns volume limits associated with the node
func (n *NodeInfo) VolumeLimits() map[v1.ResourceName]int64 {
	volumeLimits := map[v1.ResourceName]int64{}
	for k, v := range n.AllocatableResource().ScalarResources {
		if v1helper.IsAttachableVolumeResourceName(k) {
			volumeLimits[k] = v
		}
	}
	return volumeLimits
}

// String returns representation of human readable format of this NodeInfo.
func (n *NodeInfo) String() string {
	podKeys := make([]string, len(n.pods))
	for i, pod := range n.pods {
		podKeys[i] = pod.Name
	}
	return fmt.Sprintf("&NodeInfo{Pods:%v, RequestedResource:%#v, NonZeroRequest: %#v, UsedPort: %#v, AllocatableResource:%#v}",
		podKeys, n.requestedResource, n.nonzeroRequest, n.usedPorts, n.allocatableResource)
}

func hasPodAffinityConstraints(pod *v1.Pod) bool {
	affinity := pod.Spec.Affinity
	return affinity != nil && (affinity.PodAffinity != nil || affinity.PodAntiAffinity != nil)
}

// AddPod adds pod information to this NodeInfo.
func (n *NodeInfo) AddPod(pod *v1.Pod) {
	res, non0CPU, non0Mem := calculateResource(pod)
	n.requestedResource.MilliCPU += res.MilliCPU
	n.requestedResource.Memory += res.Memory
	n.requestedResource.EphemeralStorage += res.EphemeralStorage
	if n.requestedResource.ScalarResources == nil && len(res.ScalarResources) > 0 {
		n.requestedResource.ScalarResources = map[v1.ResourceName]int64{}
	}
	for rName, rQuant := range res.ScalarResources {
		n.requestedResource.ScalarResources[rName] += rQuant
	}
	n.nonzeroRequest.MilliCPU += non0CPU
	n.nonzeroRequest.Memory += non0Mem
	n.pods = append(n.pods, pod)
	if hasPodAffinityConstraints(pod) {
		n.podsWithAffinity = append(n.podsWithAffinity, pod)
	}

	// Consume ports when pods added.
	n.updateUsedPorts(pod, true)

	n.generation = nextGeneration()
}

// RemovePod subtracts pod information from this NodeInfo.
func (n *NodeInfo) RemovePod(pod *v1.Pod) error {
	k1, err := getPodKey(pod)
	if err != nil {
		return err
	}

	for i := range n.podsWithAffinity {
		k2, err := getPodKey(n.podsWithAffinity[i])
		if err != nil {
			glog.Errorf("Cannot get pod key, err: %v", err)
			continue
		}
		if k1 == k2 {
			// delete the element
			n.podsWithAffinity[i] = n.podsWithAffinity[len(n.podsWithAffinity)-1]
			n.podsWithAffinity = n.podsWithAffinity[:len(n.podsWithAffinity)-1]
			break
		}
	}
	for i := range n.pods {
		k2, err := getPodKey(n.pods[i])
		if err != nil {
			glog.Errorf("Cannot get pod key, err: %v", err)
			continue
		}
		if k1 == k2 {
			// delete the element
			n.pods[i] = n.pods[len(n.pods)-1]
			n.pods = n.pods[:len(n.pods)-1]
			// reduce the resource data
			res, non0CPU, non0Mem := calculateResource(pod)

			n.requestedResource.MilliCPU -= res.MilliCPU
			n.requestedResource.Memory -= res.Memory
			n.requestedResource.EphemeralStorage -= res.EphemeralStorage
			if len(res.ScalarResources) > 0 && n.requestedResource.ScalarResources == nil {
				n.requestedResource.ScalarResources = map[v1.ResourceName]int64{}
			}
			for rName, rQuant := range res.ScalarResources {
				n.requestedResource.ScalarResources[rName] -= rQuant
			}
			n.nonzeroRequest.MilliCPU -= non0CPU
			n.nonzeroRequest.Memory -= non0Mem

			// Release ports when remove Pods.
			n.updateUsedPorts(pod, false)

			n.generation = nextGeneration()

			return nil
		}
	}
	return fmt.Errorf("no corresponding pod %s in pods of node %s", pod.Name, n.node.Name)
}

func calculateResource(pod *v1.Pod) (res Resource, non0CPU int64, non0Mem int64) {
	resPtr := &res
	for _, c := range pod.Spec.Containers {
		resPtr.Add(c.Resources.Requests)

		non0CPUReq, non0MemReq := priorityutil.GetNonzeroRequests(&c.Resources.Requests)
		non0CPU += non0CPUReq
		non0Mem += non0MemReq
		// No non-zero resources for GPUs or opaque resources.
	}

	return
}

func (n *NodeInfo) updateUsedPorts(pod *v1.Pod, add bool) {
	for j := range pod.Spec.Containers {
		container := &pod.Spec.Containers[j]
		for k := range container.Ports {
			podPort := &container.Ports[k]
			if add {
				n.usedPorts.Add(podPort.HostIP, string(podPort.Protocol), podPort.HostPort)
			} else {
				n.usedPorts.Remove(podPort.HostIP, string(podPort.Protocol), podPort.HostPort)
			}
		}
	}
}

// SetNode sets the overall node information.
func (n *NodeInfo) SetNode(node *v1.Node) error {
	n.node = node

	n.allocatableResource = NewResource(node.Status.Allocatable)

	n.taints = node.Spec.Taints
	for i := range node.Status.Conditions {
		cond := &node.Status.Conditions[i]
		switch cond.Type {
		case v1.NodeMemoryPressure:
			n.memoryPressureCondition = cond.Status
		case v1.NodeDiskPressure:
			n.diskPressureCondition = cond.Status
		case v1.NodePIDPressure:
			n.pidPressureCondition = cond.Status
		default:
			// We ignore other conditions.
		}
	}
	n.TransientInfo = newTransientSchedulerInfo()
	n.generation = nextGeneration()
	return nil
}

// RemoveNode removes the overall information about the node.
func (n *NodeInfo) RemoveNode(node *v1.Node) error {
	// We don't remove NodeInfo for because there can still be some pods on this node -
	// this is because notifications about pods are delivered in a different watch,
	// and thus can potentially be observed later, even though they happened before
	// node removal. This is handled correctly in cache.go file.
	n.node = nil
	n.allocatableResource = &Resource{}
	n.taints, n.taintsErr = nil, nil
	n.memoryPressureCondition = v1.ConditionUnknown
	n.diskPressureCondition = v1.ConditionUnknown
	n.pidPressureCondition = v1.ConditionUnknown
	n.imageStates = make(map[string]*ImageStateSummary)
	n.generation = nextGeneration()
	return nil
}

// FilterOutPods receives a list of pods and filters out those whose node names
// are equal to the node of this NodeInfo, but are not found in the pods of this NodeInfo.
//
// Preemption logic simulates removal of pods on a node by removing them from the
// corresponding NodeInfo. In order for the simulation to work, we call this method
// on the pods returned from SchedulerCache, so that predicate functions see
// only the pods that are not removed from the NodeInfo.
func (n *NodeInfo) FilterOutPods(pods []*v1.Pod) []*v1.Pod {
	node := n.Node()
	if node == nil {
		return pods
	}
	filtered := make([]*v1.Pod, 0, len(pods))
	for _, p := range pods {
		if p.Spec.NodeName != node.Name {
			filtered = append(filtered, p)
			continue
		}
		// If pod is on the given node, add it to 'filtered' only if it is present in nodeInfo.
		podKey, _ := getPodKey(p)
		for _, np := range n.Pods() {
			npodkey, _ := getPodKey(np)
			if npodkey == podKey {
				filtered = append(filtered, p)
				break
			}
		}
	}
	return filtered
}

// getPodKey returns the string key of a pod.
func getPodKey(pod *v1.Pod) (string, error) {
	uid := string(pod.UID)
	if len(uid) == 0 {
		return "", errors.New("Cannot get cache key for pod with empty UID")
	}
	return uid, nil
}

// Filter implements PodFilter interface. It returns false only if the pod node name
// matches NodeInfo.node and the pod is not found in the pods list. Otherwise,
// returns true.
func (n *NodeInfo) Filter(pod *v1.Pod) bool {
	if pod.Spec.NodeName != n.node.Name {
		return true
	}
	for _, p := range n.pods {
		if p.Name == pod.Name && p.Namespace == pod.Namespace {
			return true
		}
	}
	return false
}
