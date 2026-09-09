/*
Copyright 2022 The Kubernetes Authors.

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

package dra

import (
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ContainerInfo contains information required by the runtime to consume prepared resources.
type ContainerInfo struct {
	// CDI Devices for the container
	CDIDevices []kubecontainer.CDIDevice
}
