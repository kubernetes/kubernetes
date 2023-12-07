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

package v1beta4

import (
	"k8s.io/apimachinery/pkg/conversion"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// Convert_kubeadm_InitConfiguration_To_v1beta4_InitConfiguration converts a private InitConfiguration to a public InitConfiguration.
func Convert_kubeadm_InitConfiguration_To_v1beta4_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	return autoConvert_kubeadm_InitConfiguration_To_v1beta4_InitConfiguration(in, out, s)
}

// Convert_v1beta4_InitConfiguration_To_kubeadm_InitConfiguration converts a public InitConfiguration to a private InitConfiguration.
func Convert_v1beta4_InitConfiguration_To_kubeadm_InitConfiguration(in *InitConfiguration, out *kubeadm.InitConfiguration, s conversion.Scope) error {
	err := autoConvert_v1beta4_InitConfiguration_To_kubeadm_InitConfiguration(in, out, s)
	if err != nil {
		return err
	}
	err = Convert_v1beta4_ClusterConfiguration_To_kubeadm_ClusterConfiguration(&ClusterConfiguration{}, &out.ClusterConfiguration, s)
	return err
}
