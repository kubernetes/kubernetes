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
	KubeVirtDevicePluginDSYAML = "test/e2e_node/testing-manifests/kubevirt-kvm-ds.yaml"

	// KubeVirtDevicePluginName is the name of the device plugin pod
	KubeVirtDevicePluginName = "kubevirt-device-plugin"

	// KubeVirtResourceName is the name of the resource provided by kubevirt device plugin
	KubeVirtResourceName = "devices.kubevirt.io/kvm"
)
