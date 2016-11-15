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

package schedulercache

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientcache "k8s.io/kubernetes/pkg/client/cache"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
)

var emptyResource = Resource{}

// NodeInfo is node level aggregated information.
type NodeInfo struct {
	// Overall node information.
	node *api.Node

	pods             []*api.Pod
	podsWithAffinity []*api.Pod

	// Total requested resource of all pods on this node.
	// It includes assumed pods which scheduler sends binding to apiserver but
	// didn't get it as scheduled yet.
	requestedResource *Resource
	nonzeroRequest    *Resource
	// We store allocatedResources (which is Node.Status.Allocatable.*) explicitly
	// as int64, to avoid conversions and accessing map.
	allocatableResource *Resource
	// We store allowedPodNumber (which is Node.Status.Allocatable.Pods().Value())
	// explicitly as int, to avoid conversions and improve performance.
	allowedPodNumber int

	// Whenever NodeInfo changes, generation is bumped.
	// This is used to avoid cloning it if the object didn't change.
	generation int64
}

// Resource is a collection of compute resource.
type Resource struct {
	MilliCPU           int64
	Memory             int64
	NvidiaGPU          int64
	OpaqueIntResources map[api.ResourceName]int64
}

func (r *Resource) ResourceList() api.ResourceList {
	result := api.ResourceList{
		api.ResourceCPU:       *resource.NewMilliQuantity(r.MilliCPU, resource.DecimalSI),
		api.ResourceMemory:    *resource.NewQuantity(r.Memory, resource.BinarySI),
		api.ResourceNvidiaGPU: *resource.NewQuantity(r.NvidiaGPU, resource.DecimalSI),
	}
	for rName, rQuant := range r.OpaqueIntResources {
		result[rName] = *resource.NewQuantity(rQuant, resource.DecimalSI)
	}
	return result
}

// NewNodeInfo returns a ready to use empty NodeInfo object.
// If any pods are given in arguments, their information will be aggregated in
// the returned object.
func NewNodeInfo(pods ...*api.Pod) *NodeInfo {
	ni := &NodeInfo{
		requestedResource:   &Resource{},
		nonzeroRequest:      &Resource{},
		allocatableResource: &Resource{},
		allowedPodNumber:    0,
		generation:          0,
	}
	for _, pod := range pods {
		ni.addPod(pod)
	}
	return ni
}

// Returns overall information about this node.
func (n *NodeInfo) Node() *api.Node {
	if n == nil {
		return nil
	}
	return n.node
}

// Pods return all pods scheduled (including assumed to be) on this node.
func (n *NodeInfo) Pods() []*api.Pod {
	if n == nil {
		return nil
	}
	return n.pods
}

// PodsWithAffinity return all pods with (anti)affinity constraints on this node.
func (n *NodeInfo) PodsWithAffinity() []*api.Pod {
	if n == nil {
		return nil
	}
	return n.podsWithAffinity
}

func (n *NodeInfo) AllowedPodNumber() int {
	if n == nil {
		return 0
	}
	return n.allowedPodNumber
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

func (n *NodeInfo) Clone() *NodeInfo {
	clone := &NodeInfo{
		node:                n.node,
		requestedResource:   &(*n.requestedResource),
		nonzeroRequest:      &(*n.nonzeroRequest),
		allocatableResource: &(*n.allocatableResource),
		allowedPodNumber:    n.allowedPodNumber,
		generation:          n.generation,
	}
	if len(n.pods) > 0 {
		clone.pods = append([]*api.Pod(nil), n.pods...)
	}
	if len(n.podsWithAffinity) > 0 {
		clone.podsWithAffinity = append([]*api.Pod(nil), n.podsWithAffinity...)
	}
	return clone
}

// String returns representation of human readable format of this NodeInfo.
func (n *NodeInfo) String() string {
	podKeys := make([]string, len(n.pods))
	for i, pod := range n.pods {
		podKeys[i] = pod.Name
	}
	return fmt.Sprintf("&NodeInfo{Pods:%v, RequestedResource:%#v, NonZeroRequest: %#v}", podKeys, n.requestedResource, n.nonzeroRequest)
}

func hasPodAffinityConstraints(pod *api.Pod) bool {
	affinity, err := api.GetAffinityFromPodAnnotations(pod.Annotations)
	if err != nil || affinity == nil {
		return false
	}
	return affinity.PodAffinity != nil || affinity.PodAntiAffinity != nil
}

// addPod adds pod information to this NodeInfo.
func (n *NodeInfo) addPod(pod *api.Pod) {
	// cpu, mem, nvidia_gpu, non0_cpu, non0_mem := calculateResource(pod)
	res, non0_cpu, non0_mem := calculateResource(pod)
	n.requestedResource.MilliCPU += res.MilliCPU
	n.requestedResource.Memory += res.Memory
	n.requestedResource.NvidiaGPU += res.NvidiaGPU
	if n.requestedResource.OpaqueIntResources == nil && len(res.OpaqueIntResources) > 0 {
		n.requestedResource.OpaqueIntResources = map[api.ResourceName]int64{}
	}
	for rName, rQuant := range res.OpaqueIntResources {
		n.requestedResource.OpaqueIntResources[rName] += rQuant
	}
	n.nonzeroRequest.MilliCPU += non0_cpu
	n.nonzeroRequest.Memory += non0_mem
	n.pods = append(n.pods, pod)
	if hasPodAffinityConstraints(pod) {
		n.podsWithAffinity = append(n.podsWithAffinity, pod)
	}
	n.generation++
}

// removePod subtracts pod information to this NodeInfo.
func (n *NodeInfo) removePod(pod *api.Pod) error {
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
			res, non0_cpu, non0_mem := calculateResource(pod)

			n.requestedResource.MilliCPU -= res.MilliCPU
			n.requestedResource.Memory -= res.Memory
			n.requestedResource.NvidiaGPU -= res.NvidiaGPU
			if len(res.OpaqueIntResources) > 0 && n.requestedResource.OpaqueIntResources == nil {
				n.requestedResource.OpaqueIntResources = map[api.ResourceName]int64{}
			}
			for rName, rQuant := range res.OpaqueIntResources {
				n.requestedResource.OpaqueIntResources[rName] -= rQuant
			}
			n.nonzeroRequest.MilliCPU -= non0_cpu
			n.nonzeroRequest.Memory -= non0_mem
			n.generation++
			return nil
		}
	}
	return fmt.Errorf("no corresponding pod %s in pods of node %s", pod.Name, n.node.Name)
}

func calculateResource(pod *api.Pod) (res Resource, non0_cpu int64, non0_mem int64) {
	for _, c := range pod.Spec.Containers {
		for rName, rQuant := range c.Resources.Requests {
			switch rName {
			case api.ResourceCPU:
				res.MilliCPU += rQuant.MilliValue()
			case api.ResourceMemory:
				res.Memory += rQuant.Value()
			case api.ResourceNvidiaGPU:
				res.NvidiaGPU += rQuant.Value()
			default:
				if api.IsOpaqueIntResourceName(rName) {
					// Lazily allocate opaque resource map.
					if res.OpaqueIntResources == nil {
						res.OpaqueIntResources = map[api.ResourceName]int64{}
					}
					res.OpaqueIntResources[rName] += rQuant.Value()
				}
			}
		}

		non0_cpu_req, non0_mem_req := priorityutil.GetNonzeroRequests(&c.Resources.Requests)
		non0_cpu += non0_cpu_req
		non0_mem += non0_mem_req
		// No non-zero resources for GPUs or opaque resources.
	}
	return
}

// Sets the overall node information.
func (n *NodeInfo) SetNode(node *api.Node) error {
	n.node = node
	for rName, rQuant := range node.Status.Allocatable {
		switch rName {
		case api.ResourceCPU:
			n.allocatableResource.MilliCPU = rQuant.MilliValue()
		case api.ResourceMemory:
			n.allocatableResource.Memory = rQuant.Value()
		case api.ResourceNvidiaGPU:
			n.allocatableResource.NvidiaGPU = rQuant.Value()
		case api.ResourcePods:
			n.allowedPodNumber = int(rQuant.Value())
		default:
			if api.IsOpaqueIntResourceName(rName) {
				// Lazily allocate opaque resource map.
				if n.allocatableResource.OpaqueIntResources == nil {
					n.allocatableResource.OpaqueIntResources = map[api.ResourceName]int64{}
				}
				n.allocatableResource.OpaqueIntResources[rName] = rQuant.Value()
			}
		}
	}
	n.generation++
	return nil
}

// Removes the overall information about the node.
func (n *NodeInfo) RemoveNode(node *api.Node) error {
	// We don't remove NodeInfo for because there can still be some pods on this node -
	// this is because notifications about pods are delivered in a different watch,
	// and thus can potentially be observed later, even though they happened before
	// node removal. This is handled correctly in cache.go file.
	n.node = nil
	n.allocatableResource = &Resource{}
	n.allowedPodNumber = 0
	n.generation++
	return nil
}

// getPodKey returns the string key of a pod.
func getPodKey(pod *api.Pod) (string, error) {
	return clientcache.MetaNamespaceKeyFunc(pod)
}
