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

const (
	// NVIDIAGPUResourceName is the extended name of the GPU resource since v1.8
	// this uses the device plugin mechanism
	NVIDIAGPUResourceName = "nvidia.com/gpu"

	// GPUDevicePluginDSYAML is the official Google Device Plugin Daemonset NVIDIA GPU manifest for GKE
	// TODO: Parametrize it by making it a feature in TestFramework.
	// so we can override the daemonset in other setups (non COS).
	GPUDevicePluginDSYAML = "https://raw.githubusercontent.com/kubernetes/kubernetes/master/cluster/addons/device-plugins/nvidia-gpu/daemonset.yaml"
)
