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

package api

import (
	api "k8s.io/kubernetes/pkg/apis/core"
)

const (
	// TaintNodeNotReady will be added when node is not ready
	// and feature-gate for TaintBasedEvictions flag is enabled,
	// and removed when node becomes ready.
	TaintNodeNotReady = "node.kubernetes.io/not-ready"

	// TaintNodeUnreachable will be added when node becomes unreachable
	// (corresponding to NodeReady status ConditionUnknown)
	// and feature-gate for TaintBasedEvictions flag is enabled,
	// and removed when node becomes reachable (NodeReady status ConditionTrue).
	TaintNodeUnreachable = "node.kubernetes.io/unreachable"

	// TaintNodeUnschedulable will be added when node becomes unschedulable
	// and feature-gate for TaintNodesByCondition flag is enabled,
	// and removed when node becomes scheduable.
	TaintNodeUnschedulable = "node.kubernetes.io/unschedulable"

	// TaintNodeMemoryPressure will be added when node has memory pressure
	// and feature-gate for TaintNodesByCondition flag is enabled,
	// and removed when node has enough memory.
	TaintNodeMemoryPressure = "node.kubernetes.io/memory-pressure"

	// TaintNodeDiskPressure will be added when node has disk pressure
	// and feature-gate for TaintNodesByCondition flag is enabled,
	// and removed when node has enough disk.
	TaintNodeDiskPressure = "node.kubernetes.io/disk-pressure"

	// TaintNodeNetworkUnavailable will be added when node's network is unavailable
	// and feature-gate for TaintNodesByCondition flag is enabled,
	// and removed when network becomes ready.
	TaintNodeNetworkUnavailable = "node.kubernetes.io/network-unavailable"

	// TaintNodePIDPressure will be added when node has pid pressure
	// and feature-gate for TaintNodesByCondition flag is enabled,
	// and removed when node has enough disk.
	TaintNodePIDPressure = "node.kubernetes.io/pid-pressure"

	// TaintExternalCloudProvider sets this taint on a node to mark it as unusable,
	// when kubelet is started with the "external" cloud provider, until a controller
	// from the cloud-controller-manager intitializes this node, and then removes
	// the taint
	TaintExternalCloudProvider = "node.cloudprovider.kubernetes.io/uninitialized"

	// TaintNodeShutdown when node is shutdown in external cloud provider
	TaintNodeShutdown = "node.cloudprovider.kubernetes.io/shutdown"

	// NodeFieldSelectorKeyNodeName ('metadata.name') uses this as node field selector key
	// when selecting node by node's name.
	NodeFieldSelectorKeyNodeName = api.ObjectNameField
)
