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

package v1alpha

const (
	// Healthy means that the device is healthy
	Healthy = "Healthy"
	// Unhealthy means that the device is unhealthy
	Unhealthy = "Unhealthy"

	// Version is the current version of the API supported by kubelet
	Version = "v1alpha2"
	// DevicePluginPath is the folder the Device Plugin is expecting sockets to be on
	// Only privileged pods have access to this path
	// Note: Placeholder until we find a "standard path"
	DevicePluginPath = "/var/lib/kubelet/device-plugins/"
	// KubeletSocket is the path of the Kubelet registry socket
	KubeletSocket = DevicePluginPath + "kubelet.sock"
)
