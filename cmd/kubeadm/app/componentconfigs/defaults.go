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
	"path/filepath"

	"k8s.io/klog"

	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	utilpointer "k8s.io/utils/pointer"
)

const (
	// kubeproxyKubeConfigFileName defines the file name for the kube-proxy's kubeconfig file
	kubeproxyKubeConfigFileName = "/var/lib/kube-proxy/kubeconfig.conf"

	// kubeletReadOnlyPort specifies the default insecure http server port
	// 0 will disable insecure http server.
	kubeletReadOnlyPort int32 = 0

	// kubeletRotateCertificates specifies the default value to enable certificate rotation
	kubeletRotateCertificates = true

	// kubeletAuthenticationAnonymousEnabled specifies the default value to disable anonymous access
	kubeletAuthenticationAnonymousEnabled = false

	// kubeletAuthorizationMode specifies the default authorization mode
	kubeletAuthorizationMode = kubeletconfigv1beta1.KubeletAuthorizationModeWebhook

	// kubeletAuthenticationWebhookEnabled set the default value to enable authentication webhook
	kubeletAuthenticationWebhookEnabled = true

	// kubeletHealthzBindAddress specifies the default healthz bind address
	kubeletHealthzBindAddress = "127.0.0.1"
)

// DefaultKubeProxyConfiguration assigns default values for the kube-proxy ComponentConfig
func DefaultKubeProxyConfiguration(internalcfg *kubeadmapi.ClusterConfiguration) {
	externalproxycfg := &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}
	kind := "KubeProxyConfiguration"

	// Do a roundtrip to the external version for defaulting
	if internalcfg.ComponentConfigs.KubeProxy != nil {
		Scheme.Convert(internalcfg.ComponentConfigs.KubeProxy, externalproxycfg, nil)
	}

	if externalproxycfg.ClusterCIDR == "" && internalcfg.Networking.PodSubnet != "" {
		externalproxycfg.ClusterCIDR = internalcfg.Networking.PodSubnet
	} else if internalcfg.Networking.PodSubnet != "" && externalproxycfg.ClusterCIDR != internalcfg.Networking.PodSubnet {
		warnDefaultComponentConfigValue(kind, "clusterCIDR", internalcfg.Networking.PodSubnet, externalproxycfg.ClusterCIDR)
	}

	if externalproxycfg.ClientConnection.Kubeconfig == "" {
		externalproxycfg.ClientConnection.Kubeconfig = kubeproxyKubeConfigFileName
	} else if externalproxycfg.ClientConnection.Kubeconfig != kubeproxyKubeConfigFileName {
		warnDefaultComponentConfigValue(kind, "clientConnection.kubeconfig", kubeproxyKubeConfigFileName, externalproxycfg.ClientConnection.Kubeconfig)
	}

	// TODO: The following code should be remvoved after dual-stack is GA.
	// Note: The user still retains the ability to explicitly set feature-gates and that value will overwrite this base value.
	if enabled, present := internalcfg.FeatureGates[features.IPv6DualStack]; present {
		externalproxycfg.FeatureGates[features.IPv6DualStack] = enabled
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
	externalkubeletcfg := &kubeletconfigv1beta1.KubeletConfiguration{}
	kind := "KubeletConfiguration"

	// Do a roundtrip to the external version for defaulting
	if internalcfg.ComponentConfigs.Kubelet != nil {
		Scheme.Convert(internalcfg.ComponentConfigs.Kubelet, externalkubeletcfg, nil)
	}

	if externalkubeletcfg.StaticPodPath == "" {
		externalkubeletcfg.StaticPodPath = kubeadmapiv1beta2.DefaultManifestsDir
	} else if externalkubeletcfg.StaticPodPath != kubeadmapiv1beta2.DefaultManifestsDir {
		warnDefaultComponentConfigValue(kind, "staticPodPath", kubeadmapiv1beta2.DefaultManifestsDir, externalkubeletcfg.StaticPodPath)
	}

	clusterDNS := ""
	dnsIP, err := constants.GetDNSIP(internalcfg.Networking.ServiceSubnet)
	if err != nil {
		clusterDNS = kubeadmapiv1beta2.DefaultClusterDNSIP
	} else {
		clusterDNS = dnsIP.String()
	}

	if externalkubeletcfg.ClusterDNS == nil {
		externalkubeletcfg.ClusterDNS = []string{clusterDNS}
	} else if len(externalkubeletcfg.ClusterDNS) != 1 || externalkubeletcfg.ClusterDNS[0] != clusterDNS {
		warnDefaultComponentConfigValue(kind, "clusterDNS", []string{clusterDNS}, externalkubeletcfg.ClusterDNS)
	}

	if externalkubeletcfg.ClusterDomain == "" {
		externalkubeletcfg.ClusterDomain = internalcfg.Networking.DNSDomain
	} else if internalcfg.Networking.DNSDomain != "" && externalkubeletcfg.ClusterDomain != internalcfg.Networking.DNSDomain {
		warnDefaultComponentConfigValue(kind, "clusterDomain", internalcfg.Networking.DNSDomain, externalkubeletcfg.ClusterDomain)
	}

	// Require all clients to the kubelet API to have client certs signed by the cluster CA
	clientCAFile := filepath.Join(internalcfg.CertificatesDir, constants.CACertName)
	if externalkubeletcfg.Authentication.X509.ClientCAFile == "" {
		externalkubeletcfg.Authentication.X509.ClientCAFile = clientCAFile
	} else if externalkubeletcfg.Authentication.X509.ClientCAFile != clientCAFile {
		warnDefaultComponentConfigValue(kind, "authentication.x509.clientCAFile", clientCAFile, externalkubeletcfg.Authentication.X509.ClientCAFile)
	}

	if externalkubeletcfg.Authentication.Anonymous.Enabled == nil {
		externalkubeletcfg.Authentication.Anonymous.Enabled = utilpointer.BoolPtr(kubeletAuthenticationAnonymousEnabled)
	} else if *externalkubeletcfg.Authentication.Anonymous.Enabled != kubeletAuthenticationAnonymousEnabled {
		warnDefaultComponentConfigValue(kind, "authentication.anonymous.enabled", kubeletAuthenticationAnonymousEnabled, *externalkubeletcfg.Authentication.Anonymous.Enabled)
	}

	// On every client request to the kubelet API, execute a webhook (SubjectAccessReview request) to the API server
	// and ask it whether the client is authorized to access the kubelet API
	if externalkubeletcfg.Authorization.Mode == "" {
		externalkubeletcfg.Authorization.Mode = kubeletAuthorizationMode
	} else if externalkubeletcfg.Authorization.Mode != kubeletAuthorizationMode {
		warnDefaultComponentConfigValue(kind, "authorization.mode", kubeletAuthorizationMode, externalkubeletcfg.Authorization.Mode)
	}

	// Let clients using other authentication methods like ServiceAccount tokens also access the kubelet API
	if externalkubeletcfg.Authentication.Webhook.Enabled == nil {
		externalkubeletcfg.Authentication.Webhook.Enabled = utilpointer.BoolPtr(kubeletAuthenticationWebhookEnabled)
	} else if *externalkubeletcfg.Authentication.Webhook.Enabled != kubeletAuthenticationWebhookEnabled {
		warnDefaultComponentConfigValue(kind, "authentication.webhook.enabled", kubeletAuthenticationWebhookEnabled, *externalkubeletcfg.Authentication.Webhook.Enabled)
	}

	// Serve a /healthz webserver on localhost:10248 that kubeadm can talk to
	if externalkubeletcfg.HealthzBindAddress == "" {
		externalkubeletcfg.HealthzBindAddress = kubeletHealthzBindAddress
	} else if externalkubeletcfg.HealthzBindAddress != kubeletHealthzBindAddress {
		warnDefaultComponentConfigValue(kind, "healthzBindAddress", kubeletHealthzBindAddress, externalkubeletcfg.HealthzBindAddress)
	}

	if externalkubeletcfg.HealthzPort == nil {
		externalkubeletcfg.HealthzPort = utilpointer.Int32Ptr(constants.KubeletHealthzPort)
	} else if *externalkubeletcfg.HealthzPort != constants.KubeletHealthzPort {
		warnDefaultComponentConfigValue(kind, "healthzPort", constants.KubeletHealthzPort, *externalkubeletcfg.HealthzPort)
	}

	if externalkubeletcfg.ReadOnlyPort != kubeletReadOnlyPort {
		warnDefaultComponentConfigValue(kind, "readOnlyPort", kubeletReadOnlyPort, externalkubeletcfg.ReadOnlyPort)
	}

	// We cannot show a warning for RotateCertificates==false and we must hardcode it to true.
	// There is no way to determine if the user has set this or not, given the field is a non-pointer.
	externalkubeletcfg.RotateCertificates = kubeletRotateCertificates

	Scheme.Default(externalkubeletcfg)

	if internalcfg.ComponentConfigs.Kubelet == nil {
		internalcfg.ComponentConfigs.Kubelet = &kubeletconfig.KubeletConfiguration{}
	}

	// TODO: Figure out how to handle errors in defaulting code
	// Go back to the internal version
	Scheme.Convert(externalkubeletcfg, internalcfg.ComponentConfigs.Kubelet, nil)
}

// warnDefaultComponentConfigValue prints a warning if the user modified a field in a certain
// CompomentConfig from the default recommended value in kubeadm.
func warnDefaultComponentConfigValue(componentConfigKind, paramName string, defaultValue, userValue interface{}) {
	klog.Warningf("The recommended value for %q in %q is: %v; the provided value is: %v",
		paramName, componentConfigKind, defaultValue, userValue)
}
