/*
Copyright 2026 The Kubernetes Authors.

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

// Package events contains the reasons of events emitted by the cloud provider
// controllers. Reasons are part of the observable API, so existing values must
// not be changed.
package events

// Service controller
const (
	// SyncLoadBalancerFailed is emitted when syncing a Service's load balancer fails.
	SyncLoadBalancerFailed = "SyncLoadBalancerFailed"

	// EnsuringLoadBalancer is emitted when creating or updating a Service's load balancer starts.
	EnsuringLoadBalancer = "EnsuringLoadBalancer"

	// EnsuredLoadBalancer is emitted when a Service's load balancer was created or updated.
	EnsuredLoadBalancer = "EnsuredLoadBalancer"

	// UnAvailableLoadBalancer is emitted when no nodes are available to back a Service's load balancer.
	UnAvailableLoadBalancer = "UnAvailableLoadBalancer"

	// UpdatedLoadBalancer is emitted when a Service's load balancer hosts were updated.
	UpdatedLoadBalancer = "UpdatedLoadBalancer"

	// UpdateLoadBalancerFailed is emitted when updating a Service's load balancer hosts fails.
	UpdateLoadBalancerFailed = "UpdateLoadBalancerFailed"

	// DeletingLoadBalancer is emitted when deleting a Service's load balancer starts.
	DeletingLoadBalancer = "DeletingLoadBalancer"

	// DeletedLoadBalancer is emitted when a Service's load balancer was deleted.
	DeletedLoadBalancer = "DeletedLoadBalancer"

	// DeleteLoadBalancerFailed is emitted when deleting a Service's load balancer fails.
	DeleteLoadBalancerFailed = "DeleteLoadBalancerFailed"
)

// Node controller
const (
	// NodeSynced is emitted with reason "Synced" when a Node was initialized with cloud provider data.
	NodeSynced = "Synced"
)

// Node lifecycle controller
const (
	// DeletingNode is emitted when a Node is deleted because its instance no longer exists in the cloud provider.
	DeletingNode = "DeletingNode"

	// DeletingNodeFailed is emitted when deleting a Node whose instance no longer exists fails.
	DeletingNodeFailed = "DeletingNodeFailed"
)

// Route controller
const (
	// FailedToCreateRoute is emitted when creating a cloud route for a Node's pod CIDR fails.
	FailedToCreateRoute = "FailedToCreateRoute"
)
