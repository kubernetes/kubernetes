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

package schedulercache

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/apis/extensions"
	clientcache "k8s.io/kubernetes/pkg/client/cache"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
)

var emptyResource = Resource{}

// ClusterInfo is cluster level aggregated information.
type ClusterInfo struct {
	// Total requested resource of all replicaSets on this cluster.
	// It includes assumed replicaSets which scheduler sends binding to apiserver but
	// didn't get it as scheduled yet.
	requestedResource *Resource
	replicaSet        []*extensions.ReplicaSet
	nonzeroRequest    *Resource
}

// Resource is a collection of compute resource.
type Resource struct {
	MilliCPU int64
	Memory   int64
}

// NewClusterInfo returns a ready to use empty ClusterInfo object.
// If any replicaSets are given in arguments, their information will be aggregated in
// the returned object.
func NewClusterInfo(replicaSet ...*extensions.ReplicaSet) *ClusterInfo {
	ci := &ClusterInfo{
		requestedResource: &Resource{},
		nonzeroRequest:    &Resource{},
	}
	for _, rc := range replicaSet {
		ci.addReplicaSet(rc)
	}
	return ci
}

// ReplicaSets return all replicaSets scheduled (including assumed to be) on this cluster.
func (c *ClusterInfo) ReplicaSets() []*extensions.ReplicaSet {
	if c == nil {
		return nil
	}
	return c.replicaSet
}

// RequestedResource returns aggregated resource request of replicaSets on this cluster.
func (c *ClusterInfo) RequestedResource() Resource {
	if c == nil {
		return emptyResource
	}
	return *c.requestedResource
}

// NonZeroRequest returns aggregated nonzero resource request of replicaSets on this cluster.
func (c *ClusterInfo) NonZeroRequest() Resource {
	if c == nil {
		return emptyResource
	}
	return *c.nonzeroRequest
}

func (c *ClusterInfo) Clone() *ClusterInfo {
	replicaSets := append([]*extensions.ReplicaSet(nil), c.replicaSet...)
	clone := &ClusterInfo{
		requestedResource: &(*c.requestedResource),
		nonzeroRequest:    &(*c.nonzeroRequest),
		replicaSet:              replicaSets,
	}
	return clone
}

// String returns representation of human readable format of this ClusterInfo.
func (c *ClusterInfo) String() string {
	rcKeys := make([]string, len(c.replicaSet))
	for i, replicaSet := range c.replicaSet {
		rcKeys[i] = replicaSet.Name
	}
	return fmt.Sprintf("&ClusterInfo{ReplicaSets:%v, RequestedResource:%#v, NonZeroRequest: %#v}", rcKeys, c.requestedResource, c.nonzeroRequest)
}

// addReplicaSet adds replicaSet information to this ClusterInfo.
func (c *ClusterInfo) addReplicaSet(rc *extensions.ReplicaSet) {
	cpu, mem, non0_cpu, non0_mem := calculateResource(rc)
	c.requestedResource.MilliCPU += cpu
	c.requestedResource.Memory += mem
	c.nonzeroRequest.MilliCPU += non0_cpu
	c.nonzeroRequest.Memory += non0_mem
	c.replicaSet = append(c.replicaSet, rc)
}

// removeReplicaSet subtracts replicaSet information to this ClusterInfo.
func (c *ClusterInfo) removeReplicaSet(replicaSet *extensions.ReplicaSet) error {
	k1, err := getReplicaSetKey(replicaSet)
	if err != nil {
		return err
	}

	cpu, mem, non0_cpu, non0_mem := calculateResource(replicaSet)
	c.requestedResource.MilliCPU -= cpu
	c.requestedResource.Memory -= mem
	c.nonzeroRequest.MilliCPU -= non0_cpu
	c.nonzeroRequest.Memory -= non0_mem

	for i := range c.replicaSet {
		k2, err := getReplicaSetKey(c.replicaSet[i])
		if err != nil {
			glog.Errorf("Cannot get replicaSet key, err: %v", err)
			continue
		}
		if k1 == k2 {
			// delete the element
			c.replicaSet[i] = c.replicaSet[len(c.replicaSet)-1]
			c.replicaSet = c.replicaSet[:len(c.replicaSet)-1]
			return nil
		}
	}
	return fmt.Errorf("no corresponding replicaSet in replicaSets")
}

func calculateResource(rs *extensions.ReplicaSet) (cpu int64, mem int64, non0_cpu int64, non0_mem int64) {
	for _, c := range rs.Spec.Template.Spec.Containers {
		req := c.Resources.Requests
		cpu += req.Cpu().MilliValue()
		mem += req.Memory().Value()

		non0_cpu_req, non0_mem_req := priorityutil.GetNonzeroRequests(&req)
		non0_cpu += non0_cpu_req
		non0_mem += non0_mem_req
	}
	return
}

// getReplicaSetKey returns the string key of a replicaSet.
func getReplicaSetKey(replicaSet *extensions.ReplicaSet) (string, error) {
	return clientcache.MetaNamespaceKeyFunc(replicaSet)
}
