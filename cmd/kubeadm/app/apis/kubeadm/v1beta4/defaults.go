/*
Copyright 2023 The Kubernetes Authors.

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

package v1beta4

import (
	"net/url"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
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
	// DefaultCertificatesDir defines default certificate directory
	DefaultCertificatesDir = "/etc/kubernetes/pki"
	// DefaultImageRepository defines default image registry
	// (previously this defaulted to k8s.gcr.io)
	DefaultImageRepository = "registry.k8s.io"
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

	// DefaultImagePullPolicy is the default image pull policy in kubeadm
	DefaultImagePullPolicy = corev1.PullIfNotPresent

	// DefaultEncryptionAlgorithm is the default encryption algorithm.
	DefaultEncryptionAlgorithm = EncryptionAlgorithmRSA2048
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_InitConfiguration assigns default values for the InitConfiguration
func SetDefaults_InitConfiguration(obj *InitConfiguration) {
	SetDefaults_BootstrapTokens(obj)
	SetDefaults_APIEndpoint(&obj.LocalAPIEndpoint)
	SetDefaults_NodeRegistration(&obj.NodeRegistration)
	if obj.Timeouts == nil {
		obj.Timeouts = &Timeouts{}
	}
	SetDefaults_Timeouts(obj.Timeouts)
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

	if obj.EncryptionAlgorithm == "" {
		obj.EncryptionAlgorithm = DefaultEncryptionAlgorithm
	}

	if obj.CertificateValidityPeriod == nil {
		obj.CertificateValidityPeriod = &metav1.Duration{
			Duration: constants.CertificateValidityPeriod,
		}
	}
	if obj.CACertificateValidityPeriod == nil {
		obj.CACertificateValidityPeriod = &metav1.Duration{
			Duration: constants.CACertificateValidityPeriod,
		}
	}

	SetDefaults_Etcd(obj)
}

// SetDefaults_Etcd assigns default values for the proxy
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

	SetDefaults_JoinControlPlane(obj.ControlPlane)
	SetDefaults_Discovery(&obj.Discovery)
	SetDefaults_NodeRegistration(&obj.NodeRegistration)
	if obj.Timeouts == nil {
		obj.Timeouts = &Timeouts{}
	}
	SetDefaults_Timeouts(obj.Timeouts)
}

// SetDefaults_JoinControlPlane assigns default values for a joining control plane node
func SetDefaults_JoinControlPlane(obj *JoinControlPlane) {
	if obj != nil {
		SetDefaults_APIEndpoint(&obj.LocalAPIEndpoint)
	}
}

// SetDefaults_Discovery assigns default values for the discovery process
func SetDefaults_Discovery(obj *Discovery) {
	if len(obj.TLSBootstrapToken) == 0 && obj.BootstrapToken != nil {
		obj.TLSBootstrapToken = obj.BootstrapToken.Token
	}

	if obj.File != nil {
		SetDefaults_FileDiscovery(obj.File)
	}
}

// SetDefaults_FileDiscovery assigns default values for file based discovery
func SetDefaults_FileDiscovery(obj *FileDiscovery) {
	// Make sure file URL becomes path
	if len(obj.KubeConfigPath) != 0 {
		u, err := url.Parse(obj.KubeConfigPath)
		if err == nil && u.Scheme == "file" {
			obj.KubeConfigPath = u.Path
		}
	}
}

// SetDefaults_BootstrapTokens sets the defaults for the .BootstrapTokens field
// If the slice is empty, it's defaulted with one token. Otherwise it just loops
// through the slice and sets the defaults for the omitempty fields that are TTL,
// Usages and Groups. Token is NOT defaulted with a random one in the API defaulting
// layer, but set to a random value later at runtime if not set before.
func SetDefaults_BootstrapTokens(obj *InitConfiguration) {

	if len(obj.BootstrapTokens) == 0 {
		obj.BootstrapTokens = []bootstraptokenv1.BootstrapToken{{}}
	}

	for i := range obj.BootstrapTokens {
		bootstraptokenv1.SetDefaults_BootstrapToken(&obj.BootstrapTokens[i])
	}
}

// SetDefaults_APIEndpoint sets the defaults for the API server instance deployed on a node.
func SetDefaults_APIEndpoint(obj *APIEndpoint) {
	if obj.BindPort == 0 {
		obj.BindPort = constants.KubeAPIServerPort
	}
}

// SetDefaults_NodeRegistration sets the defaults for the NodeRegistrationOptions object
func SetDefaults_NodeRegistration(obj *NodeRegistrationOptions) {
	if len(obj.ImagePullPolicy) == 0 {
		obj.ImagePullPolicy = DefaultImagePullPolicy
	}
	if obj.ImagePullSerial == nil {
		obj.ImagePullSerial = ptr.To(true)
	}
}

// SetDefaults_ResetConfiguration assigns default values for the ResetConfiguration object
func SetDefaults_ResetConfiguration(obj *ResetConfiguration) {
	if obj.CertificatesDir == "" {
		obj.CertificatesDir = DefaultCertificatesDir
	}
	if obj.Timeouts == nil {
		obj.Timeouts = &Timeouts{}
	}
	SetDefaults_Timeouts(obj.Timeouts)
}

// SetDefaults_EnvVar assigns default values for EnvVar.
// +k8s:defaulter-gen=covers
func SetDefaults_EnvVar(obj *EnvVar) {
	if obj.ValueFrom != nil {
		if obj.ValueFrom.FieldRef != nil {
			if obj.ValueFrom.FieldRef.APIVersion == "" {
				obj.ValueFrom.FieldRef.APIVersion = "v1"
			}
		}
	}
}

// SetDefaults_Timeouts assigns default values for timeouts.
func SetDefaults_Timeouts(obj *Timeouts) {
	if obj.ControlPlaneComponentHealthCheck == nil {
		obj.ControlPlaneComponentHealthCheck = &metav1.Duration{
			Duration: constants.ControlPlaneComponentHealthCheckTimeout,
		}
	}
	if obj.KubeletHealthCheck == nil {
		obj.KubeletHealthCheck = &metav1.Duration{
			Duration: constants.KubeletHealthCheckTimeout,
		}
	}
	if obj.KubernetesAPICall == nil {
		obj.KubernetesAPICall = &metav1.Duration{
			Duration: constants.KubernetesAPICallTimeout,
		}
	}
	if obj.EtcdAPICall == nil {
		obj.EtcdAPICall = &metav1.Duration{
			Duration: constants.EtcdAPICallTimeout,
		}
	}
	if obj.TLSBootstrap == nil {
		obj.TLSBootstrap = &metav1.Duration{
			Duration: constants.TLSBootstrapTimeout,
		}
	}
	if obj.Discovery == nil {
		obj.Discovery = &metav1.Duration{
			Duration: constants.DiscoveryTimeout,
		}
	}
	if obj.UpgradeManifests == nil {
		obj.UpgradeManifests = &metav1.Duration{
			Duration: constants.UpgradeManifestsTimeout,
		}
	}
}

// SetDefaults_UpgradeConfiguration assigns default values for the UpgradeConfiguration
func SetDefaults_UpgradeConfiguration(obj *UpgradeConfiguration) {
	if obj.Node.EtcdUpgrade == nil {
		obj.Node.EtcdUpgrade = ptr.To(true)
	}
	if obj.Node.CertificateRenewal == nil {
		obj.Node.CertificateRenewal = ptr.To(true)
	}
	if len(obj.Node.ImagePullPolicy) == 0 {
		obj.Node.ImagePullPolicy = DefaultImagePullPolicy
	}
	if obj.Node.ImagePullSerial == nil {
		obj.Node.ImagePullSerial = ptr.To(true)
	}

	if obj.Apply.EtcdUpgrade == nil {
		obj.Apply.EtcdUpgrade = ptr.To(true)
	}
	if obj.Apply.CertificateRenewal == nil {
		obj.Apply.CertificateRenewal = ptr.To(true)
	}
	if len(obj.Apply.ImagePullPolicy) == 0 {
		obj.Apply.ImagePullPolicy = DefaultImagePullPolicy
	}
	if obj.Apply.ImagePullSerial == nil {
		obj.Apply.ImagePullSerial = ptr.To(true)
	}

	if obj.Plan.EtcdUpgrade == nil {
		obj.Plan.EtcdUpgrade = ptr.To(true)
	}

	if obj.Timeouts == nil {
		obj.Timeouts = &Timeouts{}
	}
	SetDefaults_Timeouts(obj.Timeouts)
}
