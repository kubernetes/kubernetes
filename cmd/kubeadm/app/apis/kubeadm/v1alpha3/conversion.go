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
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletconfigscheme "k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	kubeproxyconfigscheme "k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
)

func Convert_v1alpha3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in *ClusterConfiguration, out *kubeadm.ClusterConfiguration, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_ClusterConfiguration_To_kubeadm_ClusterConfiguration(in, out, s); err != nil {
		return err
	}

	// TODO: This conversion code is here ONLY for fuzzing tests. When we remove the v1alpha2 API, we can remove this (unnecessary)
	// code. Right now this defaulting code has to be kept in sync with the defaulting code in cmd/kubeadm/app/apis/kubeadm/v1alpha2 and cmd/kubeadm/app/componentconfig
	if out.ComponentConfigs.Kubelet == nil {
		// Set the Kubelet ComponentConfig to an empty, defaulted struct
		out.ComponentConfigs.Kubelet = &kubeletconfig.KubeletConfiguration{}
		extkubeletconfig := &kubeletconfigv1beta1.KubeletConfiguration{}

		scheme, _, err := kubeletconfigscheme.NewSchemeAndCodecs()
		if err != nil {
			return err
		}

		scheme.Default(extkubeletconfig)
		scheme.Convert(extkubeletconfig, out.ComponentConfigs.Kubelet, nil)
		defaultKubeletConfiguration(in, out.ComponentConfigs.Kubelet)
	}
	if out.ComponentConfigs.KubeProxy == nil {
		// Set the KubeProxy ComponentConfig to an empty, defaulted struct
		out.ComponentConfigs.KubeProxy = &kubeproxyconfig.KubeProxyConfiguration{}
		extkubeproxyconfig := &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}
		kubeproxyconfigscheme.Scheme.Default(extkubeproxyconfig)
		kubeproxyconfigscheme.Scheme.Convert(extkubeproxyconfig, out.ComponentConfigs.KubeProxy, nil)
		defaultKubeProxyConfiguration(in, out.ComponentConfigs.KubeProxy)
	}
	return nil
}

func defaultKubeProxyConfiguration(internalcfg *ClusterConfiguration, obj *kubeproxyconfig.KubeProxyConfiguration) {
	// NOTE: This code should be mirrored from cmd/kubeadm/app/apis/kubeadm/v1alpha2/defaults.go and cmd/kubeadm/app/componentconfig/defaults.go
	if obj.ClusterCIDR == "" && internalcfg.Networking.PodSubnet != "" {
		obj.ClusterCIDR = internalcfg.Networking.PodSubnet
	}

	if obj.ClientConnection.Kubeconfig == "" {
		obj.ClientConnection.Kubeconfig = "/var/lib/kube-proxy/kubeconfig.conf"
	}
}

func defaultKubeletConfiguration(internalcfg *ClusterConfiguration, obj *kubeletconfig.KubeletConfiguration) {
	// NOTE: This code should be mirrored from cmd/kubeadm/app/apis/kubeadm/v1alpha2/defaults.go and cmd/kubeadm/app/componentconfig/defaults.go
	if obj.StaticPodPath == "" {
		obj.StaticPodPath = DefaultManifestsDir
	}
	if obj.ClusterDNS == nil {
		dnsIP, err := constants.GetDNSIP(internalcfg.Networking.ServiceSubnet)
		if err != nil {
			obj.ClusterDNS = []string{DefaultClusterDNSIP}
		} else {
			obj.ClusterDNS = []string{dnsIP.String()}
		}
	}
	if obj.ClusterDomain == "" {
		obj.ClusterDomain = internalcfg.Networking.DNSDomain
	}
	// Enforce security-related kubelet options

	// Require all clients to the kubelet API to have client certs signed by the cluster CA
	obj.Authentication.X509.ClientCAFile = DefaultCACertPath
	obj.Authentication.Anonymous.Enabled = false

	// On every client request to the kubelet API, execute a webhook (SubjectAccessReview request) to the API server
	// and ask it whether the client is authorized to access the kubelet API
	obj.Authorization.Mode = kubeletconfig.KubeletAuthorizationModeWebhook

	// Let clients using other authentication methods like ServiceAccount tokens also access the kubelet API
	obj.Authentication.Webhook.Enabled = true

	// Disable the readonly port of the kubelet, in order to not expose unnecessary information
	obj.ReadOnlyPort = 0

	// Enables client certificate rotation for the kubelet
	obj.RotateCertificates = true

	// Serve a /healthz webserver on localhost:10248 that kubeadm can talk to
	obj.HealthzBindAddress = "127.0.0.1"
	obj.HealthzPort = constants.KubeletHealthzPort
}
