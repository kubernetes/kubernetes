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

package nodeinfo

import (
	v1 "k8s.io/api/core/v1"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// TODO(#89528): This file defines temporary aliases of types used by kubelet.
// Those will be removed and the underlying types defined in scheduler/types will be used directly.

// NodeInfo is node level aggregated information.
type NodeInfo = framework.NodeInfo

// Resource is a collection of compute resource.
type Resource = framework.Resource

// NewResource creates a Resource from ResourceList
func NewResource(rl v1.ResourceList) *Resource {
	return framework.NewResource(rl)
}

// NewNodeInfo returns a ready to use empty NodeInfo object.
// If any pods are given in arguments, their information will be aggregated in
// the returned object.
func NewNodeInfo(pods ...*v1.Pod) *NodeInfo {
	return framework.NewNodeInfo(pods...)
}
