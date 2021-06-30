/*
Copyright 2021 The Kubernetes Authors.

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

const (
	// SampleResourceName is the name of the example resource which is used in the e2e test
	SampleResourceName = "example.com/resource"
	// SampleDevicePluginDSYAML is the path of the daemonset template of the sample device plugin. // TODO: Parametrize it by making it a feature in TestFramework.
	SampleDevicePluginDSYAML = "test/e2e/testing-manifests/sample-device-plugin.yaml"
	// SampleDevicePluginName is the name of the device plugin pod
	SampleDevicePluginName = "sample-device-plugin"
)
