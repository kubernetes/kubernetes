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

type NodeDeployment struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// Specification of the desired behavior of the NodeDeployment.
	Spec NodeDeploymentSpec `json:"spec,omitempty"`

	// Most recently observed status of the NodeDeployment.
	Status NodeDeploymentStatus `json:"status,omitempty"`
}

type NodeDeploymentSpec struct {
	// Nodes this NodeDeployment should operate on.
	//
	// We could also use a NodeSelector, but it gets more confusing as nodes are added with the
	// new selector; should we operate on them?  It's also nice that this is the same type as
	// NodesPendingOperation, NodesUndergoingOperation, and NodesPostOperation, because they all
	// represent the same set of objects.
	Nodes []string `json:"nodes,omitempty"`

	// The operation to do on nodes.
	Operation NodeDeploymentOperation `json:"strategy,omitempty"`

	// TODO: maybe add this?
	//
	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	//
	// MinReadySeconds int `json:"minReadySeconds,omitempty"`

	// Indicates that the deployment is paused and will not be processed by the
	// deployment controller.
	Paused bool `json:"paused,omitempty"`

	// The config this deployment is rolling back to. Will be cleared after rollback is done.
	RollingBack bool `json:"rollingBack,omitempty"`
}

// This is significantly different from Deployment's conception of Strategy, which is why I changed it to Operation.
type NodeDeploymentOperation struct {
	// Type of NodeDeployment. Can be "Recreate" or "RollingUpdate". Default is RollingUpdate.
	Type NodeDeploymentStrategyType string

	// Type of operation to perform. Can be "Recreate" or "InPlaceNodeUpgrade". Default is Recreate.
	Operation NodeDeploymentOperationType `json:"type,omitempty"`

	// Rolling operation config params.
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be.
	RollingUpdate *RollingUpdateNodeDeployment `json:"rollingUpdate,omitempty"`
}

type NodeDeploymentStrategyType string

const (
	// Kill all existing nodes before creating new ones.
	RecreateNodeDeploymentStrategyType NodeDeploymentStrategyType = "Recreate"

	// Operate on the nodes one by one.
	RollingUpdateNodeDeploymentStrategyType NodeDeploymentStrategyType = "RollingUpdate"
)

type NodeDeploymentOperationType string

const (
	// Recreate nodes one at a time all existing nodes before creating new ones.
	RecreateNodeDeploymentStrategyType NodeDeploymentOperationType = "RecreateNode"

	// Do an in-place upgrade of the node, including kernel, which generally involves triggering
	// an in-place update, copying metadata and rebooting the machine.
	InPlaceNodeUpgradeNodeDeploymentStrategyType NodeDeploymentOperationType = "InPlaceNodeUpgrade"
)

// Spec to control the desired behavior of rolling operation.
type RollingUpdateNodeDeployment struct {
	// The maximum number of nodes that can be unavailable during the rolling operation.
	// Value can be an absolute number (ex: 5) or a percentage of total pods at the start of update (ex: 10%).
	// Absolute number is calculated from percentage by rounding up.
	// (TODO This can not be 0 if MaxSurge is 0.)
	// By default, a fixed value of 1 is used.
	// Example: when this is set to 30%, we can immediately operate on 30%
	// of the nodes when the rolling update starts. Once some nodes are post-operation, other
	// nodes can be operated on.
	MaxUnavailable intstr.IntOrString `json:"maxUnavailable,omitempty"`

	// TODO: MaxSurge intstr.IntOrString `json:"maxSurge,omitempty"` (only for Recreate)
}

type NodeDeploymentStatus struct {
	// The generation observed by the node deployment controller.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Nodes waiting to be operated on.
	NodesPendingOperation []string `json:"nodePendingOperation,omitempty"`

	// Nodes currently being operated on.
	NodesUndergoingOperation []string `json:"nodePendingOperation,omitempty"`

	// Nodes that have already been operated on.
	NodesPostOperation []string `json:"nodePendingOperation,omitempty"`
}
