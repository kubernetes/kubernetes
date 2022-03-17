//go:build windows
// +build windows

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

package kubelet

// shouldMountHostsFile checks if the nodes /etc/hosts should be mounted
// Kubernetes only mounts on /etc/hosts if:
// - container is not an infrastructure (pause) container
// - container is not already mounting on /etc/hosts
// Kubernetes will not mount /etc/hosts if:
// - when the Pod sandbox is being created, its IP is still unknown. Hence, PodIP will not have been set.
// - Windows pod contains a hostProcess container
func shouldMountHostsFile(pod *v1.Pod, podIPs []string) bool {
	shouldMount := len(podIPs) > 0
	if utilfeature.DefaultFeatureGate.Enabled(features.WindowsHostProcessContainers) {
		return shouldMount && !kubecontainer.HasWindowsHostProcessContainer(pod)
	}
	return shouldMount
}
