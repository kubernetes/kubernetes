/*
Copyright 2023 The Kubernetes Authors.

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

package e2enode

import (
	v1 "k8s.io/api/core/v1"
)

const (
	// SampleDevicePluginDSYAML is the path of the daemonset template of the sample device plugin. // TODO: Parametrize it by making it a feature in TestFramework.
	SampleDevicePluginDSYAML                    = "test/e2e/testing-manifests/sample-device-plugin/sample-device-plugin.yaml"
	SampleDevicePluginControlRegistrationDSYAML = "test/e2e/testing-manifests/sample-device-plugin/sample-device-plugin-control-registration.yaml"

	// SampleDevicePluginName is the name of the device plugin pod
	SampleDevicePluginName = "sample-device-plugin"

	// SampleDeviceResourceName is the name of the resource provided by the sample device plugin
	SampleDeviceResourceName = "example.com/resource"

	SampleDeviceEnvVarNamePluginSockDir = "PLUGIN_SOCK_DIR"
)

// CountSampleDeviceCapacity returns the number of devices of SampleDeviceResourceName advertised by a node capacity
func CountSampleDeviceCapacity(node *v1.Node) int64 {
	val, ok := node.Status.Capacity[v1.ResourceName(SampleDeviceResourceName)]
	if !ok {
		return 0
	}
	return val.Value()
}

// CountSampleDeviceAllocatable returns the number of devices of SampleDeviceResourceName advertised by a node allocatable
func CountSampleDeviceAllocatable(node *v1.Node) int64 {
	val, ok := node.Status.Allocatable[v1.ResourceName(SampleDeviceResourceName)]
	if !ok {
		return 0
	}
	return val.Value()
}
