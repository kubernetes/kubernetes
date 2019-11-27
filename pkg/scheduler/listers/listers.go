/*
Copyright 2019 The Kubernetes Authors.

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

package listers

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// PodFilter is a function to filter a pod. If pod passed return true else return false.
type PodFilter func(*v1.Pod) bool

// PodLister interface represents anything that can list pods for a scheduler.
type PodLister interface {
	// Returns the list of pods.
	List(labels.Selector) ([]*v1.Pod, error)
	// This is similar to "List()", but the returned slice does not
	// contain pods that don't pass `podFilter`.
	FilteredList(podFilter PodFilter, selector labels.Selector) ([]*v1.Pod, error)
}

// NodeInfoLister interface represents anything that can list/get NodeInfo objects from node name.
type NodeInfoLister interface {
	// Returns the list of NodeInfos.
	List() ([]*schedulernodeinfo.NodeInfo, error)
	// Returns the list of NodeInfos of nodes with pods with affinity terms.
	HavePodsWithAffinityList() ([]*schedulernodeinfo.NodeInfo, error)
	// Returns the NodeInfo of the given node name.
	Get(nodeName string) (*schedulernodeinfo.NodeInfo, error)
}

// SharedLister groups scheduler-specific listers.
type SharedLister interface {
	Pods() PodLister
	NodeInfos() NodeInfoLister
}
