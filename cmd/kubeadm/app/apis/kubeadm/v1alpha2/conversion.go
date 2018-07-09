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

package v1alpha2

import (
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubeletconfigv1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1beta1"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1"
)

func Convert_v1alpha2_InitConfiguration_To_kubeadm_InitConfiguration(in *InitConfiguration, out *kubeadm.InitConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha2_InitConfiguration_To_kubeadm_InitConfiguration(in, out, s); err != nil {
		return err
	}

	if in.KubeProxy.Config != nil {
		if out.ComponentConfigs.KubeProxy == nil {
			out.ComponentConfigs.KubeProxy = &kubeproxyconfig.KubeProxyConfiguration{}
		}

		if err := componentconfigs.Scheme.Convert(in.KubeProxy.Config, out.ComponentConfigs.KubeProxy, nil); err != nil {
			return err
		}
	}
	if in.KubeletConfiguration.BaseConfig != nil {
		if out.ComponentConfigs.Kubelet == nil {
			out.ComponentConfigs.Kubelet = &kubeletconfig.KubeletConfiguration{}
		}

		if err := componentconfigs.Scheme.Convert(in.KubeletConfiguration.BaseConfig, out.ComponentConfigs.Kubelet, nil); err != nil {
			return err
		}
	}

	return nil
}

func Convert_kubeadm_InitConfiguration_To_v1alpha2_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_InitConfiguration_To_v1alpha2_InitConfiguration(in, out, s); err != nil {
		return err
	}

	if in.ComponentConfigs.KubeProxy != nil {
		if out.KubeProxy.Config == nil {
			out.KubeProxy.Config = &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}
		}

		if err := componentconfigs.Scheme.Convert(in.ComponentConfigs.KubeProxy, out.KubeProxy.Config, nil); err != nil {
			return err
		}
	}
	if in.ComponentConfigs.Kubelet != nil {
		if out.KubeletConfiguration.BaseConfig == nil {
			out.KubeletConfiguration.BaseConfig = &kubeletconfigv1beta1.KubeletConfiguration{}
		}

		if err := componentconfigs.Scheme.Convert(in.ComponentConfigs.Kubelet, out.KubeletConfiguration.BaseConfig, nil); err != nil {
			return err
		}
	}
	return nil
}
