/*
Copyright 2019 The Kubernetes Authors.

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

package v1beta2

import (
	"github.com/pkg/errors"

	conversion "k8s.io/apimachinery/pkg/conversion"
	kubeadm "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func Convert_kubeadm_InitConfiguration_To_v1beta2_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	return autoConvert_kubeadm_InitConfiguration_To_v1beta2_InitConfiguration(in, out, s)
}

func Convert_v1beta2_InitConfiguration_To_kubeadm_InitConfiguration(in *InitConfiguration, out *kubeadm.InitConfiguration, s conversion.Scope) error {
	err := autoConvert_v1beta2_InitConfiguration_To_kubeadm_InitConfiguration(in, out, s)
	if err != nil {
		return err
	}

	// Keep the fuzzer test happy by setting out.ClusterConfiguration to defaults
	clusterCfg := &ClusterConfiguration{}
	SetDefaults_ClusterConfiguration(clusterCfg)
	return Convert_v1beta2_ClusterConfiguration_To_kubeadm_ClusterConfiguration(clusterCfg, &out.ClusterConfiguration, s)
}

func Convert_kubeadm_ClusterConfiguration_To_v1beta2_ClusterConfiguration(in *kubeadm.ClusterConfiguration, out *ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_ClusterConfiguration_To_v1beta2_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	out.ClusterName = in.Name

	if addon, ok := in.AddOns[constants.CoreDNS]; ok {
		out.DNS.Type = CoreDNS
		if err := Convert_kubeadm_ImageMeta_To_v1beta2_ImageMeta(&addon.ImageMeta, &out.DNS.ImageMeta, s); err != nil {
			return err
		}
	}

	if addon, ok := in.AddOns[constants.KubeDNS]; ok {
		out.DNS.Type = KubeDNS
		if err := Convert_kubeadm_ImageMeta_To_v1beta2_ImageMeta(&addon.ImageMeta, &out.DNS.ImageMeta, s); err != nil {
			return err
		}
	}

	return nil
}

func Convert_v1beta2_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in *ClusterConfiguration, out *kubeadm.ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1beta2_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	out.Name = in.ClusterName

	if out.AddOns == nil {
		out.AddOns = map[string]kubeadm.AddOn{}
	}

	out.AddOns[constants.KubeProxy] = kubeadm.AddOn{
		Kind: constants.KubeProxy,
	}

	var dnsAddOn kubeadm.AddOn
	switch in.DNS.Type {
	case CoreDNS:
		dnsAddOn.Kind = constants.CoreDNS
	case KubeDNS:
		dnsAddOn.Kind = constants.KubeDNS
	default:
		return errors.Errorf("unexpected DNS addon type %q", in.DNS.Type)
	}

	if err := Convert_v1beta2_ImageMeta_To_kubeadm_ImageMeta(&in.DNS.ImageMeta, &dnsAddOn.ImageMeta, s); err != nil {
		return err
	}
	out.AddOns[dnsAddOn.Kind] = dnsAddOn

	return nil
}
