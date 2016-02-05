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

package bootstrap

import (
	"k8s.io/kubernetes/pkg/api"
)

// NodeConfig holds information about how the kubelet was launched for container management.
type NodeConfig struct {
	DockerDaemonContainerName string
	SystemContainerName       string
	KubeletContainerName      string
}

// BootstrapManager performs cgroup bootstrapping during node setup.
type BootstrapManager interface {
	// Start performs initial cgroup bootstrapping of the node.
	Start(NodeConfig) error
	// SystemContainers is the list of non-user containers managed during bootstrapping.
	SystemContainers() []SystemContainer
}

// SystemContainer describes a system container
type SystemContainer interface {
	// Absolute name of the container.
	Name() string
	// Limits is the set of resources allocated to the system container.
	Limits() api.ResourceList
}
