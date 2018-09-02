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

package componentconfigs

import (
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	utilpointer "k8s.io/utils/pointer"
)

const (
	// KubeproxyKubeConfigFileName defines the file name for the kube-proxy's KubeConfig file
	KubeproxyKubeConfigFileName = "/var/lib/kube-proxy/kubeconfig.conf"
)

// DefaultKubeProxyConfiguration assigns default values for the kube-proxy ComponentConfig
func DefaultKubeProxyConfiguration(internalcfg *kubeadmapi.ClusterConfiguration) {
	// IMPORTANT NOTE: If you're changing this code you should mirror it to cmd/kubeadm/app/apis/kubeadm/v1alpha2/defaults.go
	// and cmd/kubeadm/app/apis/kubeadm/v1alpha3/conversion.go. TODO: Remove this requirement when v1alpha2 is removed.
	externalproxycfg := &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}

	// Do a roundtrip to the external version for defaulting
	Scheme.Convert(internalcfg.ComponentConfigs.KubeProxy, externalproxycfg, nil)

	if externalproxycfg.ClusterCIDR == "" && internalcfg.Networking.PodSubnet != "" {
		externalproxycfg.ClusterCIDR = internalcfg.Networking.PodSubnet
	}

	if externalproxycfg.ClientConnection.Kubeconfig == "" {
		externalproxycfg.ClientConnection.Kubeconfig = KubeproxyKubeConfigFileName
	}

	// Run the rest of the kube-proxy defaulting code
	Scheme.Default(externalproxycfg)

	if internalcfg.ComponentConfigs.KubeProxy == nil {
		internalcfg.ComponentConfigs.KubeProxy = &kubeproxyconfig.KubeProxyConfiguration{}
	}

	// TODO: Figure out how to handle errors in defaulting code
	// Go back to the internal version
	Scheme.Convert(externalproxycfg, internalcfg.ComponentConfigs.KubeProxy, nil)
}

// DefaultKubeletConfiguration assigns default values for the kubelet ComponentConfig
func DefaultKubeletConfiguration(internalcfg *kubeadmapi.ClusterConfiguration) {
	// IMPORTANT NOTE: If you're changing this code you should mirror it to cmd/kubeadm/app/apis/kubeadm/v1alpha2/defaults.go
	// and cmd/kubeadm/app/apis/kubeadm/v1alpha3/conversion.go. TODO: Remove this requirement when v1alpha2 is removed.
	externalkubeletcfg := &kubeletconfigv1beta1.KubeletConfiguration{}

	// Do a roundtrip to the external version for defaulting
	Scheme.Convert(internalcfg.ComponentConfigs.Kubelet, externalkubeletcfg, nil)

	if externalkubeletcfg.StaticPodPath == "" {
		externalkubeletcfg.StaticPodPath = kubeadmapiv1alpha3.DefaultManifestsDir
	}
	if externalkubeletcfg.ClusterDNS == nil {
		dnsIP, err := constants.GetDNSIP(internalcfg.Networking.ServiceSubnet)
		if err != nil {
			externalkubeletcfg.ClusterDNS = []string{kubeadmapiv1alpha3.DefaultClusterDNSIP}
		} else {
			externalkubeletcfg.ClusterDNS = []string{dnsIP.String()}
		}
	}
	if externalkubeletcfg.ClusterDomain == "" {
		externalkubeletcfg.ClusterDomain = internalcfg.Networking.DNSDomain
	}

	// Enforce security-related kubelet options

	// Require all clients to the kubelet API to have client certs signed by the cluster CA
	externalkubeletcfg.Authentication.X509.ClientCAFile = kubeadmapiv1alpha3.DefaultCACertPath
	externalkubeletcfg.Authentication.Anonymous.Enabled = utilpointer.BoolPtr(false)

	// On every client request to the kubelet API, execute a webhook (SubjectAccessReview request) to the API server
	// and ask it whether the client is authorized to access the kubelet API
	externalkubeletcfg.Authorization.Mode = kubeletconfigv1beta1.KubeletAuthorizationModeWebhook

	// Let clients using other authentication methods like ServiceAccount tokens also access the kubelet API
	externalkubeletcfg.Authentication.Webhook.Enabled = utilpointer.BoolPtr(true)

	// Disable the readonly port of the kubelet, in order to not expose unnecessary information
	externalkubeletcfg.ReadOnlyPort = 0

	// Enables client certificate rotation for the kubelet
	externalkubeletcfg.RotateCertificates = true

	// Serve a /healthz webserver on localhost:10248 that kubeadm can talk to
	externalkubeletcfg.HealthzBindAddress = "127.0.0.1"
	externalkubeletcfg.HealthzPort = utilpointer.Int32Ptr(constants.KubeletHealthzPort)

	Scheme.Default(externalkubeletcfg)

	if internalcfg.ComponentConfigs.Kubelet == nil {
		internalcfg.ComponentConfigs.Kubelet = &kubeletconfig.KubeletConfiguration{}
	}

	// TODO: Figure out how to handle errors in defaulting code
	// Go back to the internal version
	Scheme.Convert(externalkubeletcfg, internalcfg.ComponentConfigs.Kubelet, nil)
}
