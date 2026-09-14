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
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	kubeproxyconfig "k8s.io/kube-proxy/config/v1alpha1"
	netutils "k8s.io/utils/net"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
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
		return &kubeProxyConfig{
			configBase: configBase{
				GroupVersion: kubeproxyconfig.SchemeGroupVersion,
			},
		}
	},
	fromCluster: kubeProxyConfigFromCluster,
}

func kubeProxyConfigFromCluster(h *handler, clientset clientset.Interface, _ *kubeadmapi.ClusterConfiguration) (kubeadmapi.ComponentConfig, error) {
	configMapName := kubeadmconstants.KubeProxyConfigMap
	klog.V(1).Infof("attempting to download the KubeProxyConfiguration from ConfigMap %q", configMapName)
	cm, err := h.fromConfigMap(clientset, configMapName, kubeadmconstants.KubeProxyConfigMapKey, false)
	if err != nil {
		return nil, errors.Wrapf(err, "could not download the kube-proxy configuration from ConfigMap %q",
			configMapName)
	}
	return cm, nil
}

// kubeProxyConfig implements the kubeadmapi.ComponentConfig interface for kube-proxy
type kubeProxyConfig struct {
	configBase
	config kubeproxyconfig.KubeProxyConfiguration
}

func (kp *kubeProxyConfig) DeepCopy() kubeadmapi.ComponentConfig {
	result := &kubeProxyConfig{}
	kp.configBase.DeepCopyInto(&result.configBase)
	kp.config.DeepCopyInto(&result.config)
	return result
}

func (kp *kubeProxyConfig) Marshal() ([]byte, error) {
	return kp.configBase.Marshal(&kp.config)
}

func (kp *kubeProxyConfig) Unmarshal(docmap kubeadmapi.DocumentMap) error {
	return kp.configBase.Unmarshal(docmap, &kp.config)
}

func kubeProxyDefaultBindAddress(address string) string {
	if ip := netutils.ParseIPSloppy(address); ip != nil && ip.To4() != nil {
		return kubeadmapiv1.DefaultProxyBindAddressv4
	}
	return kubeadmapiv1.DefaultProxyBindAddressv6
}

// isWildcardBindAddress reports whether the given bindAddress is one of the
// recommended wildcard defaults (0.0.0.0 for IPv4 or :: for IPv6).
func isWildcardBindAddress(bindAddress string) bool {
	return bindAddress == kubeadmapiv1.DefaultProxyBindAddressv4 ||
		bindAddress == kubeadmapiv1.DefaultProxyBindAddressv6
}

func (kp *kubeProxyConfig) Get() interface{} {
	return &kp.config
}

func (kp *kubeProxyConfig) Set(cfg interface{}) {
	kp.config = *cfg.(*kubeproxyconfig.KubeProxyConfiguration)
}

func (kp *kubeProxyConfig) Default(cfg *kubeadmapi.ClusterConfiguration, localAPIEndpoint *kubeadmapi.APIEndpoint, _ *kubeadmapi.NodeRegistrationOptions) {
	const kind = "KubeProxyConfiguration"

	// The below code is necessary because while KubeProxy may be defined, the user may not
	// have defined any feature-gates, thus FeatureGates will be nil and the later insertion
	// of any feature-gates will cause a panic.
	if kp.config.FeatureGates == nil {
		kp.config.FeatureGates = map[string]bool{}
	}

	switch {
	case kp.config.BindAddress == "":
		kp.config.BindAddress = kubeProxyDefaultBindAddress(localAPIEndpoint.AdvertiseAddress)
	case isWildcardBindAddress(kp.config.BindAddress):
		// 0.0.0.0 and :: are both valid explicit wildcard binds.
	case netutils.ParseIPSloppy(kp.config.BindAddress) == nil:
		klog.Warningf("The bindAddress %q in %q is not a valid IP address", kp.config.BindAddress, kind)
	default:
		// Warn if the bindAddress is not the recommended wildcard default for its
		// own IP family (0.0.0.0 for IPv4 or :: for IPv6).
		warnDefaultComponentConfigValue(kind, "bindAddress",
			kubeProxyDefaultBindAddress(kp.config.BindAddress), kp.config.BindAddress)
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

	if kp.config.Mode == "ipvs" {
		klog.Warningf("The %q field \"mode\" is set to \"ipvs\" which has been deprecated since version v1.35. "+
			"For newer Linux kernel versions, we recommend using the \"nftables\" mode, which has been GA since v1.33. "+
			"For older Linux kernel versions, you can use the \"iptables\" mode which is still the default one.", kind)
	}

	// kube-proxy throws warnings if the user has not explicitly set the mode.
	// In a future release the default mode will switch from "iptables" to "nftables" too.
	// TODO: apply the required changes in future releases:
	// - https://github.com/kubernetes/kubeadm/issues/3309
	if kp.config.Mode == "" {
		kp.config.Mode = "iptables"
	}
}

// Mutate is NOP for the kube-proxy config
func (kp *kubeProxyConfig) Mutate() error {
	return nil
}
