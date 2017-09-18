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

package framework

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"

	. "github.com/onsi/gomega"
)

const (
	// GPUResourceName is the extended name of the GPU resource since v1.8
	// this uses the device plugin mechanism
	NVIDIAGPUResourceName = "nvidia.com/gpu"

	// TODO: Parametrize it by making it a feature in TestFramework.
	// so we can override the daemonset in other setups (non COS).
	// GPUDevicePluginDSYAML is the official Google Device Plugin Daemonset NVIDIA GPU manifest for GKE
	GPUDevicePluginDSYAML = "https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/device-plugin-daemonset.yaml"
)

// TODO make this generic and not linked to COS only
// NumberOfGPUs returs the number of GPUs advertised by a node
// This is based on the Device Plugin system and expected to run on a COS based node
// After the NVIDIA drivers were installed
func NumberOfNVIDIAGPUs(node *v1.Node) int64 {
	val, ok := node.Status.Capacity[NVIDIAGPUResourceName]

	if !ok {
		return 0
	}

	return val.Value()
}

// NVIDIADevicePlugin returns the official Google Device Plugin pod for NVIDIA GPU in GKE
func NVIDIADevicePlugin(ns string) *v1.Pod {
	ds, err := DsFromManifest(GPUDevicePluginDSYAML)
	Expect(err).NotTo(HaveOccurred())
	p := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "device-plugin-nvidia-gpu-" + string(uuid.NewUUID()),
			Namespace: ns,
		},

		Spec: ds.Spec.Template.Spec,
	}
	// Remove NVIDIA drivers installation
	p.Spec.InitContainers = []v1.Container{}

	return p
}

func GetGPUDevicePluginImage() string {
	ds, err := DsFromManifest(GPUDevicePluginDSYAML)
	if err != nil || ds == nil || len(ds.Spec.Template.Spec.Containers) < 1 {
		return ""
	}
	return ds.Spec.Template.Spec.Containers[0].Image
}
