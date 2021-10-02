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
	conversion "k8s.io/apimachinery/pkg/conversion"

	kubeadm "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
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
	// Make roundtrip / fuzzers happy
	// TODO: Remove with v1beta2 https://github.com/kubernetes/kubeadm/issues/2459
	out.DNS.Type = ""
	return err
}

// Convert_kubeadm_DNS_To_v1beta3_DNS is required since Type does not exist in the DNS struct
// TODO: Remove with v1beta2 https://github.com/kubernetes/kubeadm/issues/2459
func Convert_kubeadm_DNS_To_v1beta3_DNS(in *kubeadm.DNS, out *DNS, s conversion.Scope) error {
	return autoConvert_kubeadm_DNS_To_v1beta3_DNS(in, out, s)
}

// Convert_v1beta3_ClusterConfiguration_To_kubeadm_ClusterConfiguration is required due to the missing
// DNS.Type in v1beta3. TODO: Remove with v1beta2 https://github.com/kubernetes/kubeadm/issues/2459
func Convert_v1beta3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in *ClusterConfiguration, out *kubeadm.ClusterConfiguration, s conversion.Scope) error {
	out.DNS.Type = kubeadm.CoreDNS
	return autoConvert_v1beta3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in, out, s)
}
