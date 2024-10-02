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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// Convert_kubeadm_InitConfiguration_To_v1beta4_InitConfiguration converts a private InitConfiguration to a public InitConfiguration.
func Convert_kubeadm_InitConfiguration_To_v1beta4_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	err := autoConvert_kubeadm_InitConfiguration_To_v1beta4_InitConfiguration(in, out, s)
	timeoutControlPlane := kubeadm.GetConversionTimeoutControlPlane() // Remove with v1beta3.
	if timeoutControlPlane != nil {
		out.Timeouts.ControlPlaneComponentHealthCheck = timeoutControlPlane
	}
	return err
}

// Convert_v1beta4_InitConfiguration_To_kubeadm_InitConfiguration converts a public InitConfiguration to a private InitConfiguration.
func Convert_v1beta4_InitConfiguration_To_kubeadm_InitConfiguration(in *InitConfiguration, out *kubeadm.InitConfiguration, s conversion.Scope) error {
	err := autoConvert_v1beta4_InitConfiguration_To_kubeadm_InitConfiguration(in, out, s)
	if err != nil {
		return err
	}
	err = Convert_v1beta4_ClusterConfiguration_To_kubeadm_ClusterConfiguration(&ClusterConfiguration{}, &out.ClusterConfiguration, s)
	out.ClusterConfiguration.APIServer.TimeoutForControlPlane = nil
	return err
}

// Convert_kubeadm_APIServer_To_v1beta4_APIServer is required due to missing APIServer.TimeoutForControlPlane in v1beta4.
func Convert_kubeadm_APIServer_To_v1beta4_APIServer(in *kubeadm.APIServer, out *APIServer, s conversion.Scope) error {
	return autoConvert_kubeadm_APIServer_To_v1beta4_APIServer(in, out, s)
}

// Convert_v1beta4_ClusterConfiguration_To_kubeadm_ClusterConfiguration is required due to missing TimeoutForControlPlane in v1beta4
func Convert_v1beta4_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in *ClusterConfiguration, out *kubeadm.ClusterConfiguration, s conversion.Scope) error {
	err := autoConvert_v1beta4_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in, out, s)
	out.APIServer.TimeoutForControlPlane = &metav1.Duration{}
	return err
}

// Convert_v1beta4_JoinConfiguration_To_kubeadm_JoinConfiguration converts a public JoinConfiguration to a private JoinConfiguration.
func Convert_v1beta4_JoinConfiguration_To_kubeadm_JoinConfiguration(in *JoinConfiguration, out *kubeadm.JoinConfiguration, s conversion.Scope) error {
	err := autoConvert_v1beta4_JoinConfiguration_To_kubeadm_JoinConfiguration(in, out, s)
	out.Discovery.Timeout = in.Timeouts.Discovery.DeepCopy()
	return err
}

// Convert_kubeadm_JoinConfiguration_To_v1beta4_JoinConfiguration converts a private JoinConfinguration to a public JoinCOnfiguration.
func Convert_kubeadm_JoinConfiguration_To_v1beta4_JoinConfiguration(in *kubeadm.JoinConfiguration, out *JoinConfiguration, s conversion.Scope) error {
	err := autoConvert_kubeadm_JoinConfiguration_To_v1beta4_JoinConfiguration(in, out, s)
	timeoutControlPlane := kubeadm.GetConversionTimeoutControlPlane() // Remove with v1beta3.
	if timeoutControlPlane != nil {
		out.Timeouts.ControlPlaneComponentHealthCheck = timeoutControlPlane
	}
	return err
}

// Convert_kubeadm_Discovery_To_v1beta4_Discovery is required because there is no Discovery.Timeout in v1beta4
func Convert_kubeadm_Discovery_To_v1beta4_Discovery(in *kubeadm.Discovery, out *Discovery, s conversion.Scope) error {
	return autoConvert_kubeadm_Discovery_To_v1beta4_Discovery(in, out, s)
}

// Convert_v1beta4_Discovery_To_kubeadm_Discovery is required because there is no Discovery.Timeout in v1beta4
func Convert_v1beta4_Discovery_To_kubeadm_Discovery(in *Discovery, out *kubeadm.Discovery, s conversion.Scope) error {
	err := autoConvert_v1beta4_Discovery_To_kubeadm_Discovery(in, out, s)
	out.Timeout = &metav1.Duration{}
	return err
}
