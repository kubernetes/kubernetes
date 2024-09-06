/*
Copyright 2024 The Kubernetes Authors.

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

package v1

const (
	// Version means current version of the API supported by kubelet
	Version = "v1"
	// PodResourcesPath is the folder the Pod Resources is expecting sockets to be on
	// Only privileged pods have access to this path
	// Note: Placeholder until we find a "standard path"
	PodResourcesPath = "/var/lib/kubelet/pod-resources/"
	// PodResourcesSocket is the path of the Kubelet registry socket
	PodResourcesSocket = PodResourcesPath + "kubelet.sock"

	// PodResourcesPathWindows Avoid failed to run Kubelet: bad socketPath,
	// must be an absolute path: /var/lib/kubelet/pod-resources/kubelet.sock
	PodResourcesPathWindows = "\\var\\lib\\kubelet\\pod-resources\\"
	// PodResourcesSocketWindows is the path of the Kubelet registry socket on windows
	PodResourcesSocketWindows = PodResourcesPathWindows + "kubelet.sock"
)
