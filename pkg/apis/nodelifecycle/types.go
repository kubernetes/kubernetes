/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package nodelifecycle

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

type NodeMaintenance struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// Specification of the desired behavior of the NodeMaintenance.
	Spec NodeMaintenanceSpec `json:"spec,omitempty"`

	// Most recently observed status of the NodeMaintenance.
	Status NodeMaintenanceStatus `json:"status,omitempty"`
}

type NodeMaintenanceSpec struct {
	// Nodes this NodeMaintenance should operate on.
	//
	// We could also use a NodeSelector, but it gets more confusing as nodes are added with the
	// new selector; should we operate on them?  It's also nice that this is the same type as
	// NodesPendingOperation, NodesUndergoingOperation, and NodesPostOperation, because they all
	// represent the same set of objects.
	Nodes NodeList `json:"nodes,omitempty"`

	// The operation to do on nodes.
	Operation NodeMaintenanceOperation `json:"strategy,omitempty"`

	// Minimum number of seconds for which a node, having undergone some operation, should be
	// ready for it to be considered post-operation.  Defaults to 0 (node will be considered
	// available as soon as it is ready).
	MinReadySeconds int `json:"minReadySeconds,omitempty"`

	// Indicates that the deployment is paused and will not be processed by the
	// deployment controller.
	Paused bool `json:"paused,omitempty"`

	// The config this deployment is rolling back to. Will be cleared after rollback is done.
	RollingBack bool `json:"rollingBack,omitempty"`
}

// This is significantly different from Maintenance's conception of Strategy, which is why I changed it to Operation.
type NodeMaintenanceOperation struct {
	// Type of NodeMaintenance. Can be "Recreate" or "RollingUpdate". Default is RollingUpdate.
	Type NodeMaintenanceStrategyType string

	// Whether or not to drain the node of its pods before performing the operation.
	DrainBeforeOperating bool `json:"drainBeforeOperating,omitempty"`

	// Type of operation to perform. Can be "Recreate" or "InPlaceNodeUpgrade". Default is Recreate.
	Operation NodeMaintenanceOperationType `json:"type,omitempty"`

	// Rolling operation config params.
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be.
	RollingUpdate *RollingUpdateNodeMaintenance `json:"rollingUpdate,omitempty"`
}

type NodeMaintenanceStrategyType string

const (
	// Kill all existing nodes before creating new ones.
	RecreateNodeMaintenanceStrategyType NodeMaintenanceStrategyType = "Recreate"

	// Operate on the nodes one by one.
	RollingUpdateNodeMaintenanceStrategyType NodeMaintenanceStrategyType = "RollingUpdate"
)

type NodeMaintenanceOperationType string

const (
	// Recreate nodes one at a time all existing nodes before creating new ones.
	RecreateNodeMaintenanceStrategyType NodeMaintenanceOperationType = "RecreateNode"

	// Do an in-place upgrade of the node, including kernel, which generally involves triggering
	// an in-place update, copying metadata and rebooting the machine.
	InPlaceNodeUpgradeNodeMaintenanceStrategyType NodeMaintenanceOperationType = "InPlaceNodeUpgrade"
)

// Spec to control the desired behavior of rolling operation.
type RollingUpdateNodeMaintenance struct {
	// The maximum number of nodes that can be unavailable during the rolling operation.
	// Value can be an absolute number (ex: 5) or a percentage of total pods at the start of update (ex: 10%).
	// Absolute number is calculated from percentage by rounding up.
	// By default, a fixed value of 1 is used.
	// Example: when this is set to 30%, we can immediately operate on 30%
	// of the nodes when the rolling update starts. Once some nodes are post-operation, other
	// nodes can be operated on.
	MaxUnavailable intstr.IntOrString `json:"maxUnavailable,omitempty"`
}

type NodeMaintenanceStatus struct {
	// The generation observed by the node deployment controller.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Nodes waiting to be operated on.
	NodesPendingOperation NodeList `json:"nodePendingOperation,omitempty"`

	// Nodes currently being operated on.
	NodesUndergoingOperation NodeList `json:"nodePendingOperation,omitempty"`

	// Nodes that have already been operated on.
	NodesPostOperation NodeList `json:"nodePendingOperation,omitempty"`
}
