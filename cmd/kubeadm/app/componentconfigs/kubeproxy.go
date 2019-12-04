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

package componentconfigs

import (
	"net"

	clientset "k8s.io/client-go/kubernetes"
	kubeproxyconfig "k8s.io/kube-proxy/config/v1alpha1"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
)

const (
	// KubeProxyGroup is a pointer to the used API group name for the kube-proxy config
	KubeProxyGroup = kubeproxyconfig.GroupName

	// kubeproxyKubeConfigFileName is used during defaulting. It's here so it can be accessed from the tests.
	kubeproxyKubeConfigFileName = "/var/lib/kube-proxy/kubeconfig.conf"
)

// kubeProxyHandler is the handler instance for the kube-proxy component config
var kubeProxyHandler = handler{
	GroupVersion: kubeproxyconfig.SchemeGroupVersion,
	AddToScheme:  kubeproxyconfig.AddToScheme,
	CreateEmpty: func() kubeadmapi.ComponentConfig {
		return &kubeProxyConfig{}
	},
	fromCluster: kubeProxyConfigFromCluster,
}

func kubeProxyConfigFromCluster(h *handler, clientset clientset.Interface, _ *kubeadmapi.ClusterConfiguration) (kubeadmapi.ComponentConfig, error) {
	return h.fromConfigMap(clientset, kubeadmconstants.KubeProxyConfigMap, kubeadmconstants.KubeProxyConfigMapKey, false)
}

// kubeProxyConfig implements the kubeadmapi.ComponentConfig interface for kube-proxy
type kubeProxyConfig struct {
	config kubeproxyconfig.KubeProxyConfiguration
}

func (kp *kubeProxyConfig) DeepCopy() kubeadmapi.ComponentConfig {
	result := &kubeProxyConfig{}
	kp.config.DeepCopyInto(&result.config)
	return result
}

func (kp *kubeProxyConfig) Marshal() ([]byte, error) {
	return kubeProxyHandler.Marshal(&kp.config)
}

func (kp *kubeProxyConfig) Unmarshal(docmap kubeadmapi.DocumentMap) error {
	return kubeProxyHandler.Unmarshal(docmap, &kp.config)
}

func kubeProxyDefaultBindAddress(localAdvertiseAddress string) string {
	ip := net.ParseIP(localAdvertiseAddress)
	if ip.To4() != nil {
		return kubeadmapiv1beta2.DefaultProxyBindAddressv4
	}
	return kubeadmapiv1beta2.DefaultProxyBindAddressv6
}

func (kp *kubeProxyConfig) Default(cfg *kubeadmapi.ClusterConfiguration, localAPIEndpoint *kubeadmapi.APIEndpoint) {
	const kind = "KubeProxyConfiguration"

	// The below code is necessary because while KubeProxy may be defined, the user may not
	// have defined any feature-gates, thus FeatureGates will be nil and the later insertion
	// of any feature-gates (e.g. IPv6DualStack) will cause a panic.
	if kp.config.FeatureGates == nil {
		kp.config.FeatureGates = map[string]bool{}
	}

	defaultBindAddress := kubeProxyDefaultBindAddress(localAPIEndpoint.AdvertiseAddress)
	if kp.config.BindAddress == "" {
		kp.config.BindAddress = defaultBindAddress
	} else if kp.config.BindAddress != defaultBindAddress {
		warnDefaultComponentConfigValue(kind, "bindAddress", kp.config.BindAddress, defaultBindAddress)
	}

	if kp.config.ClusterCIDR == "" && cfg.Networking.PodSubnet != "" {
		kp.config.ClusterCIDR = cfg.Networking.PodSubnet
	} else if cfg.Networking.PodSubnet != "" && kp.config.ClusterCIDR != cfg.Networking.PodSubnet {
		warnDefaultComponentConfigValue(kind, "clusterCIDR", cfg.Networking.PodSubnet, kp.config.ClusterCIDR)
	}

	if kp.config.ClientConnection.Kubeconfig == "" {
		kp.config.ClientConnection.Kubeconfig = kubeproxyKubeConfigFileName
	} else if kp.config.ClientConnection.Kubeconfig != kubeproxyKubeConfigFileName {
		warnDefaultComponentConfigValue(kind, "clientConnection.kubeconfig", kubeproxyKubeConfigFileName, kp.config.ClientConnection.Kubeconfig)
	}

	// TODO: The following code should be removed after dual-stack is GA.
	// Note: The user still retains the ability to explicitly set feature-gates and that value will overwrite this base value.
	if enabled, present := cfg.FeatureGates[features.IPv6DualStack]; present {
		kp.config.FeatureGates[features.IPv6DualStack] = enabled
	}
}
