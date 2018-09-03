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
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
)

func Convert_v1alpha2_InitConfiguration_To_kubeadm_InitConfiguration(in *InitConfiguration, out *kubeadm.InitConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha2_InitConfiguration_To_kubeadm_InitConfiguration(in, out, s); err != nil {
		return err
	}
	if err := split_v1alpha2_InitConfiguration_into_kubeadm_ClusterConfiguration(in, &out.ClusterConfiguration, s); err != nil {
		return err
	}
	if err := split_v1alpha2_InitConfiguration_into_kubeadm_APIEndpoint(in, &out.APIEndpoint, s); err != nil {
		return err
	}
	return nil
}

func split_v1alpha2_InitConfiguration_into_kubeadm_APIEndpoint(in *InitConfiguration, out *kubeadm.APIEndpoint, s conversion.Scope) error {
	out.AdvertiseAddress = in.API.AdvertiseAddress
	out.BindPort = in.API.BindPort
	// in.API.ControlPlaneEndpoint will be splitted into ClusterConfiguration
	return nil
}

func split_v1alpha2_InitConfiguration_into_kubeadm_ClusterConfiguration(in *InitConfiguration, out *kubeadm.ClusterConfiguration, s conversion.Scope) error {
	if err := split_v1alpha2_InitConfiguration_into_kubeadm_ComponentConfigs(in, &out.ComponentConfigs, s); err != nil {
		return err
	}
	if err := Convert_v1alpha2_Networking_To_kubeadm_Networking(&in.Networking, &out.Networking, s); err != nil {
		return err
	}
	if err := Convert_v1alpha2_Etcd_To_kubeadm_Etcd(&in.Etcd, &out.Etcd, s); err != nil {
		return err
	}
	if err := Convert_v1alpha2_AuditPolicyConfiguration_To_kubeadm_AuditPolicyConfiguration(&in.AuditPolicyConfiguration, &out.AuditPolicyConfiguration, s); err != nil {
		return err
	}
	out.KubernetesVersion = in.KubernetesVersion
	out.ControlPlaneEndpoint = in.API.ControlPlaneEndpoint
	out.APIServerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.APIServerExtraArgs))
	out.ControllerManagerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.ControllerManagerExtraArgs))
	out.SchedulerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.SchedulerExtraArgs))
	out.APIServerExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.APIServerExtraVolumes))
	out.ControllerManagerExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.ControllerManagerExtraVolumes))
	out.SchedulerExtraVolumes = *(*[]kubeadm.HostPathMount)(unsafe.Pointer(&in.SchedulerExtraVolumes))
	out.APIServerCertSANs = *(*[]string)(unsafe.Pointer(&in.APIServerCertSANs))
	out.CertificatesDir = in.CertificatesDir
	out.ImageRepository = in.ImageRepository
	out.UnifiedControlPlaneImage = in.UnifiedControlPlaneImage
	out.FeatureGates = *(*map[string]bool)(unsafe.Pointer(&in.FeatureGates))
	out.ClusterName = in.ClusterName
	return nil
}

func split_v1alpha2_InitConfiguration_into_kubeadm_ComponentConfigs(in *InitConfiguration, out *kubeadm.ComponentConfigs, s conversion.Scope) error {
	if in.KubeProxy.Config != nil {
		if out.KubeProxy == nil {
			out.KubeProxy = &kubeproxyconfig.KubeProxyConfiguration{}
		}

		if err := componentconfigs.Scheme.Convert(in.KubeProxy.Config, out.KubeProxy, nil); err != nil {
			return err
		}
	}
	if in.KubeletConfiguration.BaseConfig != nil {
		if out.Kubelet == nil {
			out.Kubelet = &kubeletconfig.KubeletConfiguration{}
		}

		if err := componentconfigs.Scheme.Convert(in.KubeletConfiguration.BaseConfig, out.Kubelet, nil); err != nil {
			return err
		}
	}
	return nil
}

func Convert_v1alpha2_JoinConfiguration_To_kubeadm_JoinConfiguration(in *JoinConfiguration, out *kubeadm.JoinConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha2_JoinConfiguration_To_kubeadm_JoinConfiguration(in, out, s); err != nil {
		return err
	}
	out.APIEndpoint.AdvertiseAddress = in.AdvertiseAddress
	out.APIEndpoint.BindPort = in.BindPort
	return nil
}

func Convert_kubeadm_InitConfiguration_To_v1alpha2_InitConfiguration(in *kubeadm.InitConfiguration, out *InitConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_InitConfiguration_To_v1alpha2_InitConfiguration(in, out, s); err != nil {
		return err
	}
	if err := join_kubeadm_ClusterConfiguration_into_v1alpha2_InitConfiguration(&in.ClusterConfiguration, out, s); err != nil {
		return err
	}
	if err := join_kubeadm_APIEndpoint_into_v1alpha2_InitConfiguration(&in.APIEndpoint, out, s); err != nil {
		return err
	}
	return nil
}

func join_kubeadm_ClusterConfiguration_into_v1alpha2_InitConfiguration(in *kubeadm.ClusterConfiguration, out *InitConfiguration, s conversion.Scope) error {
	if err := join_kubeadm_ComponentConfigs_into_v1alpha2_InitConfiguration(&in.ComponentConfigs, out, s); err != nil {
		return err
	}
	if err := Convert_kubeadm_Etcd_To_v1alpha2_Etcd(&in.Etcd, &out.Etcd, s); err != nil {
		return err
	}
	if err := Convert_kubeadm_Networking_To_v1alpha2_Networking(&in.Networking, &out.Networking, s); err != nil {
		return err
	}
	if err := Convert_kubeadm_AuditPolicyConfiguration_To_v1alpha2_AuditPolicyConfiguration(&in.AuditPolicyConfiguration, &out.AuditPolicyConfiguration, s); err != nil {
		return err
	}
	out.KubernetesVersion = in.KubernetesVersion
	out.API.ControlPlaneEndpoint = in.ControlPlaneEndpoint
	out.APIServerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.APIServerExtraArgs))
	out.ControllerManagerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.ControllerManagerExtraArgs))
	out.SchedulerExtraArgs = *(*map[string]string)(unsafe.Pointer(&in.SchedulerExtraArgs))
	out.APIServerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.APIServerExtraVolumes))
	out.ControllerManagerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.ControllerManagerExtraVolumes))
	out.SchedulerExtraVolumes = *(*[]HostPathMount)(unsafe.Pointer(&in.SchedulerExtraVolumes))
	out.APIServerCertSANs = *(*[]string)(unsafe.Pointer(&in.APIServerCertSANs))
	out.CertificatesDir = in.CertificatesDir
	out.ImageRepository = in.ImageRepository
	out.UnifiedControlPlaneImage = in.UnifiedControlPlaneImage
	out.FeatureGates = *(*map[string]bool)(unsafe.Pointer(&in.FeatureGates))
	out.ClusterName = in.ClusterName
	return nil
}

func join_kubeadm_APIEndpoint_into_v1alpha2_InitConfiguration(in *kubeadm.APIEndpoint, out *InitConfiguration, s conversion.Scope) error {
	out.API.AdvertiseAddress = in.AdvertiseAddress
	out.API.BindPort = in.BindPort
	// out.API.ControlPlaneEndpoint will join from ClusterConfiguration
	return nil
}

func join_kubeadm_ComponentConfigs_into_v1alpha2_InitConfiguration(in *kubeadm.ComponentConfigs, out *InitConfiguration, s conversion.Scope) error {
	if in.KubeProxy != nil {
		if out.KubeProxy.Config == nil {
			out.KubeProxy.Config = &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}
		}

		if err := componentconfigs.Scheme.Convert(in.KubeProxy, out.KubeProxy.Config, nil); err != nil {
			return err
		}
	}
	if in.Kubelet != nil {
		if out.KubeletConfiguration.BaseConfig == nil {
			out.KubeletConfiguration.BaseConfig = &kubeletconfigv1beta1.KubeletConfiguration{}
		}

		if err := componentconfigs.Scheme.Convert(in.Kubelet, out.KubeletConfiguration.BaseConfig, nil); err != nil {
			return err
		}
	}
	return nil
}

func Convert_kubeadm_JoinConfiguration_To_v1alpha2_JoinConfiguration(in *kubeadm.JoinConfiguration, out *JoinConfiguration, s conversion.Scope) error {
	if err := autoConvert_kubeadm_JoinConfiguration_To_v1alpha2_JoinConfiguration(in, out, s); err != nil {
		return err
	}
	out.AdvertiseAddress = in.APIEndpoint.AdvertiseAddress
	out.BindPort = in.APIEndpoint.BindPort
	return nil
}
