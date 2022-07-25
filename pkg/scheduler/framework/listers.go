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

package framework

// NodeInfoLister interface represents anything that can list/get NodeInfo objects from node name.
type NodeInfoLister interface {
	// List returns the list of NodeInfos.
	List() ([]*NodeInfo, error)
	// HavePodsWithAffinityList returns the list of NodeInfos of nodes with pods with affinity terms.
	HavePodsWithAffinityList() ([]*NodeInfo, error)
	// HavePodsWithRequiredAntiAffinityList returns the list of NodeInfos of nodes with pods with required anti-affinity terms.
	HavePodsWithRequiredAntiAffinityList() ([]*NodeInfo, error)
	// Get returns the NodeInfo of the given node name.
	Get(nodeName string) (*NodeInfo, error)
}

// StorageInfoLister interface represents anything that handles storage-related operations and resources.
type StorageInfoLister interface {
	// IsPVCUsedByPods returns true/false on whether the PVC is used by one or more scheduled pods,
	// keyed in the format "namespace/name".
	IsPVCUsedByPods(key string) bool
}

// SharedLister groups scheduler-specific listers.
type SharedLister interface {
	NodeInfos() NodeInfoLister
	StorageInfos() StorageInfoLister
}
