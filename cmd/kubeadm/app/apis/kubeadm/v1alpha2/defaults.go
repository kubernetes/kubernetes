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
	"net/url"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kubeproxyconfigv1alpha1 "k8s.io/kube-proxy/config/v1alpha1"
	kubeletconfigv1beta1 "k8s.io/kubelet/config/v1beta1"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeletscheme "k8s.io/kubernetes/pkg/kubelet/apis/config/scheme"
	kubeproxyscheme "k8s.io/kubernetes/pkg/proxy/apis/config/scheme"
	utilpointer "k8s.io/utils/pointer"
)

const (
	// DefaultServiceDNSDomain defines default cluster-internal domain name for Services and Pods
	DefaultServiceDNSDomain = "cluster.local"
	// DefaultServicesSubnet defines default service subnet range
	DefaultServicesSubnet = "10.96.0.0/12"
	// DefaultClusterDNSIP defines default DNS IP
	DefaultClusterDNSIP = "10.96.0.10"
	// DefaultKubernetesVersion defines default kubernetes version
	DefaultKubernetesVersion = "stable-1"
	// DefaultAPIBindPort defines default API port
	DefaultAPIBindPort = 6443
	// DefaultCertificatesDir defines default certificate directory
	DefaultCertificatesDir = "/etc/kubernetes/pki"
	// DefaultImageRepository defines default image registry
	DefaultImageRepository = "k8s.gcr.io"
	// DefaultManifestsDir defines default manifests directory
	DefaultManifestsDir = "/etc/kubernetes/manifests"
	// DefaultClusterName defines the default cluster name
	DefaultClusterName = "kubernetes"

	// DefaultEtcdDataDir defines default location of etcd where static pods will save data to
	DefaultEtcdDataDir = "/var/lib/etcd"
	// DefaultProxyBindAddressv4 is the default bind address when the advertise address is v4
	DefaultProxyBindAddressv4 = "0.0.0.0"
	// DefaultProxyBindAddressv6 is the default bind address when the advertise address is v6
	DefaultProxyBindAddressv6 = "::"
	// KubeproxyKubeConfigFileName defines the file name for the kube-proxy's KubeConfig file
	KubeproxyKubeConfigFileName = "/var/lib/kube-proxy/kubeconfig.conf"

	// DefaultDiscoveryTimeout specifies the default discovery timeout for kubeadm (used unless one is specified in the JoinConfiguration)
	DefaultDiscoveryTimeout = 5 * time.Minute
)

var (
	// DefaultAuditPolicyLogMaxAge is defined as a var so its address can be taken
	// It is the number of days to store audit logs
	DefaultAuditPolicyLogMaxAge = int32(2)
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_InitConfiguration assigns default values to Master node
func SetDefaults_InitConfiguration(obj *InitConfiguration) {
	if obj.KubernetesVersion == "" {
		obj.KubernetesVersion = DefaultKubernetesVersion
	}

	if obj.API.BindPort == 0 {
		obj.API.BindPort = DefaultAPIBindPort
	}

	if obj.Networking.ServiceSubnet == "" {
		obj.Networking.ServiceSubnet = DefaultServicesSubnet
	}

	if obj.Networking.DNSDomain == "" {
		obj.Networking.DNSDomain = DefaultServiceDNSDomain
	}

	if obj.CertificatesDir == "" {
		obj.CertificatesDir = DefaultCertificatesDir
	}

	if obj.ImageRepository == "" {
		obj.ImageRepository = DefaultImageRepository
	}

	if obj.ClusterName == "" {
		obj.ClusterName = DefaultClusterName
	}

	SetDefaults_NodeRegistrationOptions(&obj.NodeRegistration)
	SetDefaults_BootstrapTokens(obj)
	SetDefaults_KubeletConfiguration(obj)
	SetDefaults_Etcd(obj)
	SetDefaults_ProxyConfiguration(obj)
	SetDefaults_AuditPolicyConfiguration(obj)
}

// SetDefaults_Etcd assigns default values for the Proxy
func SetDefaults_Etcd(obj *InitConfiguration) {
	if obj.Etcd.External == nil && obj.Etcd.Local == nil {
		obj.Etcd.Local = &LocalEtcd{}
	}
	if obj.Etcd.Local != nil {
		if obj.Etcd.Local.DataDir == "" {
			obj.Etcd.Local.DataDir = DefaultEtcdDataDir
		}
	}
}

// SetDefaults_ProxyConfiguration assigns default values for the Proxy
func SetDefaults_ProxyConfiguration(obj *InitConfiguration) {
	// IMPORTANT NOTE: If you're changing this code you should mirror it to cmd/kubeadm/app/componentconfig/defaults.go
	// and cmd/kubeadm/app/apis/kubeadm/v1alpha3/conversion.go.
	if obj.KubeProxy.Config == nil {
		obj.KubeProxy.Config = &kubeproxyconfigv1alpha1.KubeProxyConfiguration{}
	}
	if obj.KubeProxy.Config.ClusterCIDR == "" && obj.Networking.PodSubnet != "" {
		obj.KubeProxy.Config.ClusterCIDR = obj.Networking.PodSubnet
	}

	if obj.KubeProxy.Config.ClientConnection.Kubeconfig == "" {
		obj.KubeProxy.Config.ClientConnection.Kubeconfig = KubeproxyKubeConfigFileName
	}

	kubeproxyscheme.Scheme.Default(obj.KubeProxy.Config)
}

// SetDefaults_JoinConfiguration assigns default values to a regular node
func SetDefaults_JoinConfiguration(obj *JoinConfiguration) {
	if obj.CACertPath == "" {
		obj.CACertPath = DefaultCACertPath
	}
	if len(obj.TLSBootstrapToken) == 0 {
		obj.TLSBootstrapToken = obj.Token
	}
	if len(obj.DiscoveryToken) == 0 && len(obj.DiscoveryFile) == 0 {
		obj.DiscoveryToken = obj.Token
	}
	// Make sure file URLs become paths
	if len(obj.DiscoveryFile) != 0 {
		u, err := url.Parse(obj.DiscoveryFile)
		if err == nil && u.Scheme == "file" {
			obj.DiscoveryFile = u.Path
		}
	}
	if obj.DiscoveryTimeout == nil {
		obj.DiscoveryTimeout = &metav1.Duration{
			Duration: DefaultDiscoveryTimeout,
		}
	}
	if obj.ClusterName == "" {
		obj.ClusterName = DefaultClusterName
	}

	if obj.BindPort == 0 {
		obj.BindPort = DefaultAPIBindPort
	}

	SetDefaults_NodeRegistrationOptions(&obj.NodeRegistration)
}

// SetDefaults_KubeletConfiguration assigns default values to kubelet
func SetDefaults_KubeletConfiguration(obj *InitConfiguration) {
	// IMPORTANT NOTE: If you're changing this code you should mirror it to cmd/kubeadm/app/componentconfig/defaults.go
	// and cmd/kubeadm/app/apis/kubeadm/v1alpha3/conversion.go.
	if obj.KubeletConfiguration.BaseConfig == nil {
		obj.KubeletConfiguration.BaseConfig = &kubeletconfigv1beta1.KubeletConfiguration{}
	}
	if obj.KubeletConfiguration.BaseConfig.StaticPodPath == "" {
		obj.KubeletConfiguration.BaseConfig.StaticPodPath = DefaultManifestsDir
	}
	if obj.KubeletConfiguration.BaseConfig.ClusterDNS == nil {
		dnsIP, err := constants.GetDNSIP(obj.Networking.ServiceSubnet)
		if err != nil {
			obj.KubeletConfiguration.BaseConfig.ClusterDNS = []string{DefaultClusterDNSIP}
		} else {
			obj.KubeletConfiguration.BaseConfig.ClusterDNS = []string{dnsIP.String()}
		}
	}
	if obj.KubeletConfiguration.BaseConfig.ClusterDomain == "" {
		obj.KubeletConfiguration.BaseConfig.ClusterDomain = obj.Networking.DNSDomain
	}

	// Enforce security-related kubelet options

	// Require all clients to the kubelet API to have client certs signed by the cluster CA
	obj.KubeletConfiguration.BaseConfig.Authentication.X509.ClientCAFile = DefaultCACertPath
	obj.KubeletConfiguration.BaseConfig.Authentication.Anonymous.Enabled = utilpointer.BoolPtr(false)

	// On every client request to the kubelet API, execute a webhook (SubjectAccessReview request) to the API server
	// and ask it whether the client is authorized to access the kubelet API
	obj.KubeletConfiguration.BaseConfig.Authorization.Mode = kubeletconfigv1beta1.KubeletAuthorizationModeWebhook

	// Let clients using other authentication methods like ServiceAccount tokens also access the kubelet API
	obj.KubeletConfiguration.BaseConfig.Authentication.Webhook.Enabled = utilpointer.BoolPtr(true)

	// Disable the readonly port of the kubelet, in order to not expose unnecessary information
	obj.KubeletConfiguration.BaseConfig.ReadOnlyPort = 0

	// Enables client certificate rotation for the kubelet
	obj.KubeletConfiguration.BaseConfig.RotateCertificates = true

	// Serve a /healthz webserver on localhost:10248 that kubeadm can talk to
	obj.KubeletConfiguration.BaseConfig.HealthzBindAddress = "127.0.0.1"
	obj.KubeletConfiguration.BaseConfig.HealthzPort = utilpointer.Int32Ptr(constants.KubeletHealthzPort)

	scheme, _, _ := kubeletscheme.NewSchemeAndCodecs()
	if scheme != nil {
		scheme.Default(obj.KubeletConfiguration.BaseConfig)
	}
}

func SetDefaults_NodeRegistrationOptions(obj *NodeRegistrationOptions) {
	if obj.CRISocket == "" {
		obj.CRISocket = DefaultCRISocket
	}
}

// SetDefaults_AuditPolicyConfiguration sets default values for the AuditPolicyConfiguration
func SetDefaults_AuditPolicyConfiguration(obj *InitConfiguration) {
	if obj.AuditPolicyConfiguration.LogDir == "" {
		obj.AuditPolicyConfiguration.LogDir = constants.StaticPodAuditPolicyLogDir
	}
	if obj.AuditPolicyConfiguration.LogMaxAge == nil {
		obj.AuditPolicyConfiguration.LogMaxAge = &DefaultAuditPolicyLogMaxAge
	}
}

// SetDefaults_BootstrapTokens sets the defaults for the .BootstrapTokens field
// If the slice is empty, it's defaulted with one token. Otherwise it just loops
// through the slice and sets the defaults for the omitempty fields that are TTL,
// Usages and Groups. Token is NOT defaulted with a random one in the API defaulting
// layer, but set to a random value later at runtime if not set before.
func SetDefaults_BootstrapTokens(obj *InitConfiguration) {

	if obj.BootstrapTokens == nil || len(obj.BootstrapTokens) == 0 {
		obj.BootstrapTokens = []BootstrapToken{{}}
	}

	for i := range obj.BootstrapTokens {
		SetDefaults_BootstrapToken(&obj.BootstrapTokens[i])
	}
}

// SetDefaults_BootstrapToken sets the defaults for an individual Bootstrap Token
func SetDefaults_BootstrapToken(bt *BootstrapToken) {
	if bt.TTL == nil {
		bt.TTL = &metav1.Duration{
			Duration: constants.DefaultTokenDuration,
		}
	}
	if len(bt.Usages) == 0 {
		bt.Usages = constants.DefaultTokenUsages
	}

	if len(bt.Groups) == 0 {
		bt.Groups = constants.DefaultTokenGroups
	}
}
