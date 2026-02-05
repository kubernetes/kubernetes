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

package lifecycle

import (
	v1 "k8s.io/api/core/v1"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
)

// NodeInfoProviderStub is a NodeInfoProvider that returns a configurable NodeInfo.
// Used for testing when the NodeInfo cache behavior is not being tested.
type NodeInfoProviderStub struct {
	pods []*v1.Pod
}

var _ NodeInfoProvider = &NodeInfoProviderStub{}

// NewNodeInfoProviderStub returns an instance of NodeInfoProviderStub with an empty NodeInfo.
func NewNodeInfoProviderStub() *NodeInfoProviderStub {
	return &NodeInfoProviderStub{}
}

// NewNodeInfoProviderStubWithPods returns an instance of NodeInfoProviderStub
// pre-populated with the given pods.
func NewNodeInfoProviderStubWithPods(pods []*v1.Pod) *NodeInfoProviderStub {
	return &NodeInfoProviderStub{pods: pods}
}

// Snapshot returns a NodeInfo snapshot containing the configured pods.
func (n *NodeInfoProviderStub) Snapshot() *schedulerframework.NodeInfo {
	return schedulerframework.NewNodeInfo(n.pods...)
}
