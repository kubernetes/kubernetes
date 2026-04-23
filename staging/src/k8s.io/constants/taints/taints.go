/*
Copyright 2025 The Kubernetes Authors.

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

// Package taints contains well-known Kubernetes taint keys.
// These are taints that have special meaning to Kubernetes components.
//
// Identifiers in this package intentionally omit a "Taint" prefix so that
// callers read naturally as "taints.NodeNotReady" rather than
// "taints.TaintNodeNotReady".
package taints

// Node condition taints. These taints are automatically added and removed
// by the node controller based on node conditions.
const (
	// NodeNotReady is added when node is not ready and removed when
	// node becomes ready.
	NodeNotReady = "node.kubernetes.io/not-ready"

	// NodeUnreachable is added when node becomes unreachable (corresponding
	// to NodeReady status ConditionUnknown) and removed when node becomes
	// reachable (NodeReady status ConditionTrue).
	NodeUnreachable = "node.kubernetes.io/unreachable"

	// NodeUnschedulable is added when node becomes unschedulable and removed
	// when node becomes schedulable.
	NodeUnschedulable = "node.kubernetes.io/unschedulable"

	// NodeMemoryPressure is added when node has memory pressure and removed
	// when node has enough memory.
	NodeMemoryPressure = "node.kubernetes.io/memory-pressure"

	// NodeDiskPressure is added when node has disk pressure and removed when
	// node has enough disk.
	NodeDiskPressure = "node.kubernetes.io/disk-pressure"

	// NodeNetworkUnavailable is added when node's network is unavailable and
	// removed when network becomes ready.
	NodeNetworkUnavailable = "node.kubernetes.io/network-unavailable"

	// NodePIDPressure is added when node has pid pressure and removed when
	// node has enough pid.
	NodePIDPressure = "node.kubernetes.io/pid-pressure"

	// NodeOutOfService can be added when node is out of service in case of a
	// non-graceful shutdown.
	NodeOutOfService = "node.kubernetes.io/out-of-service"
)
