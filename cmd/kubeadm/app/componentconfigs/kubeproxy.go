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
	"github.com/pkg/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	kubeproxyconfig "k8s.io/kube-proxy/config/v1alpha1"
	netutils "k8s.io/utils/net"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
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

func kubeProxyDefaultBindAddress(localAdvertiseAddress string) string {
	ip := netutils.ParseIPSloppy(localAdvertiseAddress)
	if ip.To4() != nil {
		return kubeadmapiv1.DefaultProxyBindAddressv4
	}
	return kubeadmapiv1.DefaultProxyBindAddressv6
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

	defaultBindAddress := kubeProxyDefaultBindAddress(localAPIEndpoint.AdvertiseAddress)
	if kp.config.BindAddress == "" {
		kp.config.BindAddress = defaultBindAddress
	} else if kp.config.BindAddress != defaultBindAddress {
		warnDefaultComponentConfigValue(kind, "bindAddress", defaultBindAddress, kp.config.BindAddress)
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
}

// Mutate is NOP for the kube-proxy config
func (kp *kubeProxyConfig) Mutate() error {
	return nil
}
