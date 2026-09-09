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

package qos

// ResourceIsolationLevel defines the level of isolation for a resource.
type ResourceIsolationLevel string

const (
	// ResourceIsolationHost implies the resource is shared with other containers on the host.
	ResourceIsolationHost ResourceIsolationLevel = "host"
	// ResourceIsolationPod implies the resource is isolated from other pods but shared within the pod.
	ResourceIsolationPod ResourceIsolationLevel = "pod"
	// ResourceIsolationContainer implies the resource is exclusive to the container.
	ResourceIsolationContainer ResourceIsolationLevel = "container"
)
