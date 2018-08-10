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
	"unsafe"

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

	if err := Convert_v1alpha2_API_To_kubeadm_API(&in.API, &out.ClusterConfiguration.API, s); err != nil {
		return err
	}
	if err := Convert_v1alpha2_Etcd_To_kubeadm_Etcd(&in.Etcd, &out.ClusterConfiguration.Etcd, s); err != nil {
		return err
	}
	if err := Convert_v1alpha2_Networking_To_kubeadm_Networking(&in.Networking, &out.ClusterConfiguration.Networking, s); err != nil {
		return err
	}
	out.ClusterConfiguration.KubernetesVersion = in.KubernetesVersion
	out.ClusterConfiguration.APIServerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.APIServerExtraArgs))
	out.ClusterConfiguration.ControllerManagerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.ControllerManagerExtraArgs))
	out.ClusterConfiguration.SchedulerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.SchedulerExtraArgs))
	out.ClusterConfiguration.APIServerExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.APIServerExtraVolumes))
	out.ClusterConfiguration.ControllerManagerExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.ControllerManagerExtraVolumes))
	out.ClusterConfiguration.SchedulerExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.SchedulerExtraVolumes))
	out.ClusterConfiguration.APIServerCertSANs = *(*[]string)(unsafe.Pointer(&in.APIServerCertSANs))
	out.ClusterConfiguration.CertificatesDir = in.CertificatesDir
	out.ClusterConfiguration.ImageRepository = in.ImageRepository
	out.ClusterConfiguration.UnifiedControlPlaneImage = in.UnifiedControlPlaneImage
	if err := Convert_v1alpha2_AuditPolicyConfiguration_To_kubeadm_AuditPolicyConfiguration(&in.AuditPolicyConfiguration, &out.ClusterConfiguration.AuditPolicyConfiguration, s); err != nil {
		return err
	}
	out.ClusterConfiguration.FeatureGates = *(*map[string]bool)(unsafe.Pointer(&in.FeatureGates))
	out.ClusterConfiguration.ClusterName = in.ClusterName
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

	if err := Convert_kubeadm_API_To_v1alpha2_API(&in.ClusterConfiguration.API, &out.API, s); err != nil {
		return err
	}
	if err := Convert_kubeadm_Etcd_To_v1alpha2_Etcd(&in.ClusterConfiguration.Etcd, &out.Etcd, s); err != nil {
		return err
	}
	if err := Convert_kubeadm_Networking_To_v1alpha2_Networking(&in.ClusterConfiguration.Networking, &out.Networking, s); err != nil {
		return err
	}
	out.KubernetesVersion = in.ClusterConfiguration.KubernetesVersion
	out.APIServerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.ClusterConfiguration.APIServerExtraArgs))
	out.ControllerManagerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.ClusterConfiguration.ControllerManagerExtraArgs))
	out.SchedulerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.ClusterConfiguration.SchedulerExtraArgs))
	out.APIServerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.ClusterConfiguration.APIServerExtraVolumes))
	out.ControllerManagerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.ClusterConfiguration.ControllerManagerExtraVolumes))
	out.SchedulerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.ClusterConfiguration.SchedulerExtraVolumes))
	out.APIServerCertSANs = *(*[]string)(unsafe.Pointer(&in.ClusterConfiguration.APIServerCertSANs))
	out.CertificatesDir = in.ClusterConfiguration.CertificatesDir
	out.ImageRepository = in.ClusterConfiguration.ImageRepository
	out.UnifiedControlPlaneImage = in.ClusterConfiguration.UnifiedControlPlaneImage
	if err := Convert_kubeadm_AuditPolicyConfiguration_To_v1alpha2_AuditPolicyConfiguration(&in.ClusterConfiguration.AuditPolicyConfiguration, &out.AuditPolicyConfiguration, s); err != nil {
		return err
	}
	out.FeatureGates = *(*map[string]bool)(unsafe.Pointer(&in.ClusterConfiguration.FeatureGates))
	out.ClusterName = in.ClusterConfiguration.ClusterName

	return nil
}
