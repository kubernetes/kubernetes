/*
Copyright 2018 The Kubernetes Authors.

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

package v1alpha3

import (
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletconfigscheme "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/scheme"
	kubeletconfigv1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1beta1"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig"
	kubeproxyconfigscheme "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/scheme"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1"
)

func Convert_v1alpha3_MasterConfiguration_To_kubeadm_MasterConfiguration(in *MasterConfiguration, out *kubeadm.MasterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_MasterConfiguration_To_kubeadm_MasterConfiguration(in, out, s); err != nil {
		return err
	}

	// TODO: Remove this conversion code ASAP, as the ComponentConfig structs should not be in the external version of the kubeadm API, but be marshalled as
	// different YAML documents
	if in.KubeProxy.Config != nil {
		if out.ComponentConfigs.KubeProxy == nil {
			out.ComponentConfigs.KubeProxy = &kubeproxyconfig.KubeProxyConfiguration{}
		}

		if err := kubeproxyconfigscheme.Scheme.Convert(in.KubeProxy.Config, out.ComponentConfigs.KubeProxy, nil); err != nil {
			return err
		}
	}
	if in.KubeletConfiguration.BaseConfig != nil {
		if out.ComponentConfigs.Kubelet == nil {
			out.ComponentConfigs.Kubelet = &kubeletconfig.KubeletConfiguration{}
		}

		scheme, _, err := kubeletconfigscheme.NewSchemeAndCodecs()
		if err != nil {
			return err
		}

		if err := scheme.Convert(in.KubeletConfiguration.BaseConfig, out.ComponentConfigs.Kubelet, nil); err != nil {
			return err
		}
	}

	return nil
}

func Convert_kubeadm_MasterConfiguration_To_v1alpha3_MasterConfiguration(in *kubeadm.MasterConfiguration, out *MasterConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_MasterConfiguration_To_v1alpha3_MasterConfiguration(in, out, s); err != nil {
		return err
	}

	// TODO: Remove this conversion code ASAP, as the ComponentConfig structs should not be in the external version of the kubeadm API, but be marshalled as
	// different YAML documents
	if in.ComponentConfigs.KubeProxy != nil {
		if out.KubeProxy.Config == nil {
			out.KubeProxy.Config = &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}
		}

		if err := kubeproxyconfigscheme.Scheme.Convert(in.ComponentConfigs.KubeProxy, out.KubeProxy.Config, nil); err != nil {
			return err
		}
	}
	if in.ComponentConfigs.Kubelet != nil {
		if out.KubeletConfiguration.BaseConfig == nil {
			out.KubeletConfiguration.BaseConfig = &kubeletconfigv1beta1.KubeletConfiguration{}
		}

		scheme, _, err := kubeletconfigscheme.NewSchemeAndCodecs()
		if err != nil {
			return err
		}

		if err := scheme.Convert(in.ComponentConfigs.Kubelet, out.KubeletConfiguration.BaseConfig, nil); err != nil {
			return err
		}
	}
	return nil
}
