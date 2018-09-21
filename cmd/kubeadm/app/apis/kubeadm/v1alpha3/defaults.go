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
	"net/url"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
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

// SetDefaults_InitConfiguration assigns default values for the InitConfiguration
func SetDefaults_InitConfiguration(obj *InitConfiguration) {
	SetDefaults_ClusterConfiguration(&obj.ClusterConfiguration)
	SetDefaults_NodeRegistrationOptions(&obj.NodeRegistration)
	SetDefaults_BootstrapTokens(obj)
	SetDefaults_APIEndpoint(&obj.APIEndpoint)
}

// SetDefaults_ClusterConfiguration assigns default values for the ClusterConfiguration
func SetDefaults_ClusterConfiguration(obj *ClusterConfiguration) {
	if obj.KubernetesVersion == "" {
		obj.KubernetesVersion = DefaultKubernetesVersion
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

	SetDefaults_Etcd(obj)
	SetDefaults_AuditPolicyConfiguration(obj)
}

// SetDefaults_Etcd assigns default values for the Proxy
func SetDefaults_Etcd(obj *ClusterConfiguration) {
	if obj.Etcd.External == nil && obj.Etcd.Local == nil {
		obj.Etcd.Local = &LocalEtcd{}
	}
	if obj.Etcd.Local != nil {
		if obj.Etcd.Local.DataDir == "" {
			obj.Etcd.Local.DataDir = DefaultEtcdDataDir
		}
	}
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

	SetDefaults_NodeRegistrationOptions(&obj.NodeRegistration)
	SetDefaults_APIEndpoint(&obj.APIEndpoint)
}

func SetDefaults_NodeRegistrationOptions(obj *NodeRegistrationOptions) {
	if obj.CRISocket == "" {
		obj.CRISocket = DefaultCRISocket
	}
}

// SetDefaults_AuditPolicyConfiguration sets default values for the AuditPolicyConfiguration
func SetDefaults_AuditPolicyConfiguration(obj *ClusterConfiguration) {
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

// SetDefaults_APIEndpoint sets the defaults for the API server instance deployed on a node.
func SetDefaults_APIEndpoint(obj *APIEndpoint) {
	if obj.BindPort == 0 {
		obj.BindPort = DefaultAPIBindPort
	}
}
