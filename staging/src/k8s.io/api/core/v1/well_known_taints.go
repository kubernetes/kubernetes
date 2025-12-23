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

package v1

import (
	"k8s.io/constants/taints"
)

// Well-known taints re-exported from k8s.io/constants/taints for backwards compatibility.
const (
	// TaintNodeNotReady will be added when node is not ready
	// and removed when node becomes ready.
	TaintNodeNotReady = taints.TaintNodeNotReady

	// TaintNodeUnreachable will be added when node becomes unreachable
	// (corresponding to NodeReady status ConditionUnknown)
	// and removed when node becomes reachable (NodeReady status ConditionTrue).
	TaintNodeUnreachable = taints.TaintNodeUnreachable

	// TaintNodeUnschedulable will be added when node becomes unschedulable
	// and removed when node becomes schedulable.
	TaintNodeUnschedulable = taints.TaintNodeUnschedulable

	// TaintNodeMemoryPressure will be added when node has memory pressure
	// and removed when node has enough memory.
	TaintNodeMemoryPressure = taints.TaintNodeMemoryPressure

	// TaintNodeDiskPressure will be added when node has disk pressure
	// and removed when node has enough disk.
	TaintNodeDiskPressure = taints.TaintNodeDiskPressure

	// TaintNodeNetworkUnavailable will be added when node's network is unavailable
	// and removed when network becomes ready.
	TaintNodeNetworkUnavailable = taints.TaintNodeNetworkUnavailable

	// TaintNodePIDPressure will be added when node has pid pressure
	// and removed when node has enough pid.
	TaintNodePIDPressure = taints.TaintNodePIDPressure

	// TaintNodeOutOfService can be added when node is out of service in case of
	// a non-graceful shutdown
	TaintNodeOutOfService = taints.TaintNodeOutOfService
)
