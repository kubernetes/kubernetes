/*
Copyright 2017 The Kubernetes Authors.

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

package gpu

import "k8s.io/api/core/v1"

// GPUManager manages GPUs on a local node.
// Implementations are expected to be thread safe.
type GPUManager interface {
	// Start logically initializes GPUManager
	Start() error
	// Capacity returns the total number of GPUs on the node.
	Capacity() v1.ResourceList
	// AllocateGPU attempts to allocate GPUs for input container.
	// Returns paths to allocated GPUs and nil on success.
	// Returns an error on failure.
	AllocateGPU(*v1.Pod, *v1.Container) ([]string, error)
}
