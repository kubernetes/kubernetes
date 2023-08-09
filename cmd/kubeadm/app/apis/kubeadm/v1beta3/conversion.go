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

package v1beta3

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/conversion"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

func Convert_kubeadm_InitConfiguration_To_v1beta3_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	return autoConvert_kubeadm_InitConfiguration_To_v1beta3_InitConfiguration(in, out, s)
}

func Convert_v1beta3_InitConfiguration_To_kubeadm_InitConfiguration(in *InitConfiguration, out *kubeadm.InitConfiguration, s conversion.Scope) error {
	err := autoConvert_v1beta3_InitConfiguration_To_kubeadm_InitConfiguration(in, out, s)
	if err != nil {
		return err
	}
	err = Convert_v1beta3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(&ClusterConfiguration{}, &out.ClusterConfiguration, s)
	return err
}

// Convert_v1beta3_ControlPlaneComponent_To_kubeadm_ControlPlaneComponent is required due to the missing ControlPlaneComponent.ExtraEnvs in v1beta3.
func Convert_v1beta3_ControlPlaneComponent_To_kubeadm_ControlPlaneComponent(in *ControlPlaneComponent, out *kubeadm.ControlPlaneComponent, s conversion.Scope) error {
	out.ExtraEnvs = []v1.EnvVar{}
	return autoConvert_v1beta3_ControlPlaneComponent_To_kubeadm_ControlPlaneComponent(in, out, s)
}

func Convert_kubeadm_ControlPlaneComponent_To_v1beta3_ControlPlaneComponent(in *kubeadm.ControlPlaneComponent, out *ControlPlaneComponent, s conversion.Scope) error {
	return autoConvert_kubeadm_ControlPlaneComponent_To_v1beta3_ControlPlaneComponent(in, out, s)
}

// Convert_v1beta3_LocalEtcd_To_kubeadm_LocalEtcd is required due to the missing LocalEtcd.ExtraEnvs in v1beta3.
func Convert_v1beta3_LocalEtcd_To_kubeadm_LocalEtcd(in *LocalEtcd, out *kubeadm.LocalEtcd, s conversion.Scope) error {
	out.ExtraEnvs = []v1.EnvVar{}
	return autoConvert_v1beta3_LocalEtcd_To_kubeadm_LocalEtcd(in, out, s)
}

func Convert_kubeadm_LocalEtcd_To_v1beta3_LocalEtcd(in *kubeadm.LocalEtcd, out *LocalEtcd, s conversion.Scope) error {
	return autoConvert_kubeadm_LocalEtcd_To_v1beta3_LocalEtcd(in, out, s)
}
