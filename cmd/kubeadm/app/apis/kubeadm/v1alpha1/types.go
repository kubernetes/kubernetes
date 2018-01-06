/*
Copyright 2016 The Kubernetes Authors.

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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfigv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1alpha1"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MasterConfiguration contains a list of elements which make up master's
// configuration object.
type MasterConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	API                  API                  `json:"api"`
	KubeProxy            KubeProxy            `json:"kubeProxy"`
	Etcd                 Etcd                 `json:"etcd"`
	KubeletConfiguration KubeletConfiguration `json:"kubeletConfiguration"`
	Networking           Networking           `json:"networking"`
	KubernetesVersion    string               `json:"kubernetesVersion"`
	CloudProvider        string               `json:"cloudProvider"`
	NodeName             string               `json:"nodeName"`
	AuthorizationModes   []string             `json:"authorizationModes,omitempty"`

	Token    string           `json:"token"`
	TokenTTL *metav1.Duration `json:"tokenTTL,omitempty"`

	APIServerExtraArgs         map[string]string `json:"apiServerExtraArgs,omitempty"`
	ControllerManagerExtraArgs map[string]string `json:"controllerManagerExtraArgs,omitempty"`
	SchedulerExtraArgs         map[string]string `json:"schedulerExtraArgs,omitempty"`

	APIServerExtraVolumes         []HostPathMount `json:"apiServerExtraVolumes,omitempty"`
	ControllerManagerExtraVolumes []HostPathMount `json:"controllerManagerExtraVolumes,omitempty"`
	SchedulerExtraVolumes         []HostPathMount `json:"schedulerExtraVolumes,omitempty"`

	// APIServerCertSANs sets extra Subject Alternative Names for the API Server signing cert
	APIServerCertSANs []string `json:"apiServerCertSANs,omitempty"`
	// CertificatesDir specifies where to store or look for all required certificates
	CertificatesDir string `json:"certificatesDir"`

	// ImageRepository what container registry to pull control plane images from
	ImageRepository string `json:"imageRepository"`
	// UnifiedControlPlaneImage specifies if a specific container image should be used for all control plane components
	UnifiedControlPlaneImage string `json:"unifiedControlPlaneImage"`

	// FeatureGates enabled by the user
	FeatureGates map[string]bool `json:"featureGates,omitempty"`
}

// API struct contains elements of API server address.
type API struct {
	// AdvertiseAddress sets the address for the API server to advertise.
	AdvertiseAddress string `json:"advertiseAddress"`
	// BindPort sets the secure port for the API Server to bind to
	BindPort int32 `json:"bindPort"`
}

// TokenDiscovery contains elements needed for token discovery
type TokenDiscovery struct {
	ID        string   `json:"id"`
	Secret    string   `json:"secret"`
	Addresses []string `json:"addresses"`
}

// Networking contains elements describing cluster's networking configuration
type Networking struct {
	ServiceSubnet string `json:"serviceSubnet"`
	PodSubnet     string `json:"podSubnet"`
	DNSDomain     string `json:"dnsDomain"`
}

// Etcd contains elements describing Etcd configuration
type Etcd struct {
	Endpoints []string          `json:"endpoints"`
	CAFile    string            `json:"caFile"`
	CertFile  string            `json:"certFile"`
	KeyFile   string            `json:"keyFile"`
	DataDir   string            `json:"dataDir"`
	ExtraArgs map[string]string `json:"extraArgs,omitempty"`
	// Image specifies which container image to use for running etcd. If empty, automatically populated by kubeadm using the image repository and default etcd version
	Image      string          `json:"image"`
	SelfHosted *SelfHostedEtcd `json:"selfHosted,omitempty"`
}

// SelfHostedEtcd describes options required to configure self-hosted etcd
type SelfHostedEtcd struct {
	// CertificatesDir represents the directory where all etcd TLS assets are stored. By default this is
	// a dir names "etcd" in the main CertificatesDir value.
	CertificatesDir string `json:"certificatesDir"`
	// ClusterServiceName is the name of the service that load balances the etcd cluster
	ClusterServiceName string `json:"clusterServiceName"`
	// EtcdVersion is the version of etcd running in the cluster.
	EtcdVersion string `json:"etcdVersion"`
	// OperatorVersion is the version of the etcd-operator to use.
	OperatorVersion string `json:"operatorVersion"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeConfiguration contains elements describing a particular node
type NodeConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	CACertPath               string   `json:"caCertPath"`
	DiscoveryFile            string   `json:"discoveryFile"`
	DiscoveryToken           string   `json:"discoveryToken"`
	DiscoveryTokenAPIServers []string `json:"discoveryTokenAPIServers,omitempty"`
	NodeName                 string   `json:"nodeName"`
	TLSBootstrapToken        string   `json:"tlsBootstrapToken"`
	Token                    string   `json:"token"`

	// DiscoveryTokenCACertHashes specifies a set of public key pins to verify
	// when token-based discovery is used. The root CA found during discovery
	// must match one of these values. Specifying an empty set disables root CA
	// pinning, which can be unsafe. Each hash is specified as "<type>:<value>",
	// where the only currently supported type is "sha256". This is a hex-encoded
	// SHA-256 hash of the Subject Public Key Info (SPKI) object in DER-encoded
	// ASN.1. These hashes can be calculated using, for example, OpenSSL:
	// openssl x509 -pubkey -in ca.crt openssl rsa -pubin -outform der 2>&/dev/null | openssl dgst -sha256 -hex
	DiscoveryTokenCACertHashes []string `json:"discoveryTokenCACertHashes,omitempty"`

	// DiscoveryTokenUnsafeSkipCAVerification allows token-based discovery
	// without CA verification via DiscoveryTokenCACertHashes. This can weaken
	// the security of kubeadm since other nodes can impersonate the master.
	DiscoveryTokenUnsafeSkipCAVerification bool `json:"discoveryTokenUnsafeSkipCAVerification"`

	// FeatureGates enabled by the user
	FeatureGates map[string]bool `json:"featureGates,omitempty"`
}

// KubeletConfiguration contains elements describing initial remote configuration of kubelet
type KubeletConfiguration struct {
	BaseConfig *kubeletconfigv1alpha1.KubeletConfiguration `json:"baseConfig,omitempty"`
}

// HostPathMount contains elements describing volumes that are mounted from the
// host
type HostPathMount struct {
	Name      string `json:"name"`
	HostPath  string `json:"hostPath"`
	MountPath string `json:"mountPath"`
}

// KubeProxy contains elements describing the proxy configuration
type KubeProxy struct {
	Config *kubeproxyconfigv1alpha1.KubeProxyConfiguration `json:"config,omitempty"`
}
