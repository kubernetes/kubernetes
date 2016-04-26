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

	clientcache "k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// ClusterInfo is cluster level aggregated information.
type ClusterInfo struct {
	// It includes assumed replicaSets which federated-scheduler sends binding to apiserver but
	// didn't get it as scheduled yet.
	replicaSets []*v1beta1.ReplicaSet
}

// NewClusterInfo returns a ready to use empty ClusterInfo object.
// If any replicaSets are given in arguments, their information will be aggregated in
// the returned object.
func NewClusterInfo(replicaSet ...*v1beta1.ReplicaSet) *ClusterInfo {
	ci := &ClusterInfo{}
	for _, rc := range replicaSet {
		ci.addReplicaSet(rc)
	}
	return ci
}

// ReplicaSets return all replicaSets scheduled (including assumed to be) on this cluster.
func (c *ClusterInfo) SubReplicaSets() []*v1beta1.ReplicaSet {
	if c == nil {
		return nil
	}
	return c.replicaSets
}

func (c *ClusterInfo) Clone() *ClusterInfo {
	replicaSets := append([]*v1beta1.ReplicaSet(nil), c.replicaSets...)
	clone := &ClusterInfo{
		replicaSets:              replicaSets,
	}
	return clone
}

// String returns representation of human readable format of this ClusterInfo.
func (c *ClusterInfo) String() string {
	rcKeys := make([]string, len(c.replicaSets))
	for i, replicaSet := range c.replicaSets {
		rcKeys[i] = replicaSet.Name
	}
	return fmt.Sprintf("&ClusterInfo{ReplicaSets:%v}", rcKeys)
}

// addReplicaSet adds replicaSet information to this ClusterInfo.
func (c *ClusterInfo) addReplicaSet(replicaSet *v1beta1.ReplicaSet) {
	c.replicaSets = append(c.replicaSets, replicaSet)
}

// removeReplicaSet subtracts replicaSet information to this ClusterInfo.
func (c *ClusterInfo) removeReplicaSet(replicaSet *v1beta1.ReplicaSet) error {
	k1, err := getReplicaSetKey(replicaSet)
	if err != nil {
		return err
	}
	for i := range c.replicaSets {
		k2, err := getReplicaSetKey(c.replicaSets[i])
		if err != nil {
			glog.Errorf("Cannot get replicaSet key, err: %v", err)
			continue
		}
		if k1 == k2 {
			// delete the element
			c.replicaSets[i] = c.replicaSets[len(c.replicaSets)-1]
			c.replicaSets = c.replicaSets[:len(c.replicaSets)-1]
			return nil
		}
	}
	return fmt.Errorf("no corresponding replicaSet in replicaSets")
}

// getReplicaSetKey returns the string key of a replicaSet.
func getReplicaSetKey(replicaSet *v1beta1.ReplicaSet) (string, error) {
	return clientcache.MetaNamespaceKeyFunc(replicaSet)
}
