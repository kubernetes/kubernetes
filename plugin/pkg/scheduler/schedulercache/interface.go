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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
)

// Cache collects pods' information and provides node-level aggregated information.
// It's intended for generic scheduler to do efficient lookup.
type Cache interface {
	// GetNodeNameToInfoMap returns a map of node names to node info. The node info contains
	// aggregated information of pods scheduled (including assumed to be) on this node.
	GetNodeNameToInfoMap() map[string]*NodeInfo

	// List lists all pods added (including assumed) in this cache
	List(labels.Selector) []*api.Pod
}

// PodLister is a clone of algorithm.PodLister. There is important cycle issue if we use that one.
// TODO: move algorithm.PodLister into a separate standalone package.
type PodLister interface {
	List(labels.Selector) ([]*api.Pod, error)
}

// CacheToPodLister make a Cache have the List method required by algorithm.PodLister
type CacheToPodLister struct {
	Cache Cache
}

func (c2p *CacheToPodLister) List(selector labels.Selector) ([]*api.Pod, error) {
	return c2p.Cache.List(selector), nil
}
