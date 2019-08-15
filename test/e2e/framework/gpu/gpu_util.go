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

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// NVIDIAGPUResourceName is the extended name of the GPU resource since v1.8
	// this uses the device plugin mechanism
	NVIDIAGPUResourceName = "nvidia.com/gpu"

	// GPUDevicePluginDSYAML is the official Google Device Plugin Daemonset NVIDIA GPU manifest for GKE
	// TODO: Parametrize it by making it a feature in TestFramework.
	// so we can override the daemonset in other setups (non COS).
	GPUDevicePluginDSYAML = "https://raw.githubusercontent.com/kubernetes/kubernetes/master/cluster/addons/device-plugins/nvidia-gpu/daemonset.yaml"
)

// NumberOfNVIDIAGPUs returns the number of GPUs advertised by a node
// This is based on the Device Plugin system and expected to run on a COS based node
// After the NVIDIA drivers were installed
// TODO make this generic and not linked to COS only
func NumberOfNVIDIAGPUs(node *v1.Node) int64 {
	val, ok := node.Status.Capacity[NVIDIAGPUResourceName]
	if !ok {
		return 0
	}
	return val.Value()
}

// NVIDIADevicePlugin returns the official Google Device Plugin pod for NVIDIA GPU in GKE
func NVIDIADevicePlugin() *v1.Pod {
	ds, err := framework.DsFromManifest(GPUDevicePluginDSYAML)
	framework.ExpectNoError(err)
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "device-plugin-nvidia-gpu-" + string(uuid.NewUUID()),
			Namespace: metav1.NamespaceSystem,
		},
		Spec: ds.Spec.Template.Spec,
	}
	// Remove node affinity
	p.Spec.Affinity = nil
	return p
}
