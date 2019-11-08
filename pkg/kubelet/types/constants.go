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

package types

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// Defaults.
const (
	// ResolvConfDefault defines the default DNS resolver configuration.
	ResolvConfDefault = "/etc/resolv.conf"

	// NamespaceDefault defines the default namespace.
	NamespaceDefault = metav1.NamespaceDefault
)

// Container runtimes.
const (
	// DockerContainerRuntime defines the docker container runtime.
	DockerContainerRuntime = "docker"
	// RemoteContainerRuntime defines the remote container runtime.
	RemoteContainerRuntime = "remote"
)

// User visible keys for managing node allocatable enforcement on the node.
const (
	NodeAllocatableEnforcementKey = "pods"
	SystemReservedEnforcementKey  = "system-reserved"
	KubeReservedEnforcementKey    = "kube-reserved"
	NodeAllocatableNoneKey        = "none"
)
