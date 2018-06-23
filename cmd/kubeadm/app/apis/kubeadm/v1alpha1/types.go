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
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfigv1beta1 "k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig/v1beta1"
	kubeproxyconfigv1alpha1 "k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig/v1alpha1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MasterConfiguration contains a list of elements which make up master's
// configuration object.
type MasterConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// API holds configuration for the k8s apiserver.
	API API `json:"api"`
	// KubeProxy holds configuration for the k8s service proxy.
	KubeProxy KubeProxy `json:"kubeProxy"`
	// Etcd holds configuration for etcd.
	Etcd Etcd `json:"etcd"`
	// KubeletConfiguration holds configuration for the kubelet.
	KubeletConfiguration KubeletConfiguration `json:"kubeletConfiguration"`
	// Networking holds configuration for the networking topology of the cluster.
	Networking Networking `json:"networking"`
	// KubernetesVersion is the target version of the control plane.
	KubernetesVersion string `json:"kubernetesVersion"`
	// CloudProvider is the name of the cloud provider.
	CloudProvider string `json:"cloudProvider"`
	// NodeName is the name of the node that will host the k8s control plane.
	// Defaults to the hostname if not provided.
	NodeName string `json:"nodeName"`
	// AuthorizationModes is a set of authorization modes used inside the cluster.
	// If not specified, defaults to Node and RBAC, meaning both the node
	// authorizer and RBAC are enabled.
	AuthorizationModes []string `json:"authorizationModes,omitempty"`
	// NoTaintMaster will, if set, suppress the tainting of the
	// master node allowing workloads to be run on it (e.g. in
	// single node configurations).
	NoTaintMaster bool `json:"noTaintMaster,omitempty"`

	// Mark the controller and api server pods as privileged as some cloud
	// controllers like openstack need escalated privileges under some conditions
	// example - loading a config drive to fetch node information
	PrivilegedPods bool `json:"privilegedPods"`

	// Token is used for establishing bidirectional trust between nodes and masters.
	// Used for joining nodes in the cluster.
	Token string `json:"token"`
	// TokenTTL defines the ttl for Token. Defaults to 24h.
	TokenTTL *metav1.Duration `json:"tokenTTL,omitempty"`
	// TokenUsages describes the ways in which this token can be used.
	TokenUsages []string `json:"tokenUsages,omitempty"`
	// Extra groups that this token will authenticate as when used for authentication
	TokenGroups []string `json:"tokenGroups,omitempty"`

	// CRISocket is used to retrieve container runtime info.
	CRISocket string `json:"criSocket,omitempty"`

	// APIServerExtraArgs is a set of extra flags to pass to the API Server or override
	// default ones in form of <flagname>=<value>.
	// TODO: This is temporary and ideally we would like to switch all components to
	// use ComponentConfig + ConfigMaps.
	APIServerExtraArgs map[string]string `json:"apiServerExtraArgs,omitempty"`
	// ControllerManagerExtraArgs is a set of extra flags to pass to the Controller Manager
	// or override default ones in form of <flagname>=<value>
	// TODO: This is temporary and ideally we would like to switch all components to
	// use ComponentConfig + ConfigMaps.
	ControllerManagerExtraArgs map[string]string `json:"controllerManagerExtraArgs,omitempty"`
	// SchedulerExtraArgs is a set of extra flags to pass to the Scheduler or override
	// default ones in form of <flagname>=<value>
	// TODO: This is temporary and ideally we would like to switch all components to
	// use ComponentConfig + ConfigMaps.
	SchedulerExtraArgs map[string]string `json:"schedulerExtraArgs,omitempty"`

	// APIServerExtraVolumes is an extra set of host volumes mounted to the API server.
	APIServerExtraVolumes []HostPathMount `json:"apiServerExtraVolumes,omitempty"`
	// ControllerManagerExtraVolumes is an extra set of host volumes mounted to the
	// Controller Manager.
	ControllerManagerExtraVolumes []HostPathMount `json:"controllerManagerExtraVolumes,omitempty"`
	// SchedulerExtraVolumes is an extra set of host volumes mounted to the scheduler.
	SchedulerExtraVolumes []HostPathMount `json:"schedulerExtraVolumes,omitempty"`

	// APIServerCertSANs sets extra Subject Alternative Names for the API Server signing cert.
	APIServerCertSANs []string `json:"apiServerCertSANs,omitempty"`
	// CertificatesDir specifies where to store or look for all required certificates.
	CertificatesDir string `json:"certificatesDir"`

	// ImageRepository what container registry to pull control plane images from
	ImageRepository string `json:"imageRepository"`
	// ImagePullPolicy that control plane images. Can be Always, IfNotPresent or Never.
	ImagePullPolicy v1.PullPolicy `json:"imagePullPolicy,omitempty"`
	// UnifiedControlPlaneImage specifies if a specific container image should
	// be used for all control plane components.
	UnifiedControlPlaneImage string `json:"unifiedControlPlaneImage"`

	// AuditPolicyConfiguration defines the options for the api server audit system
	AuditPolicyConfiguration AuditPolicyConfiguration `json:"auditPolicy"`

	// FeatureGates enabled by the user.
	FeatureGates map[string]bool `json:"featureGates,omitempty"`

	// The cluster name
	ClusterName string `json:"clusterName,omitempty"`
}

// API struct contains elements of API server address.
type API struct {
	// AdvertiseAddress sets the IP address for the API server to advertise.
	AdvertiseAddress string `json:"advertiseAddress"`
	// ControlPlaneEndpoint sets a stable IP address or DNS name for the control plane; it
	// can be a valid IP address or a RFC-1123 DNS subdomain, both with optional TCP port.
	// In case the ControlPlaneEndpoint is not specified, the AdvertiseAddress + BindPort
	// are used; in case the ControlPlaneEndpoint is specified but without a TCP port,
	// the BindPort is used.
	// Possible usages are:
	// e.g. In an cluster with more than one control plane instances, this field should be
	// assigned the address of the external load balancer in front of the
	// control plane instances.
	// e.g.  in environments with enforced node recycling, the ControlPlaneEndpoint
	// could be used for assigning a stable DNS to the control plane.
	ControlPlaneEndpoint string `json:"controlPlaneEndpoint"`
	// BindPort sets the secure port for the API Server to bind to.
	// Defaults to 6443.
	BindPort int32 `json:"bindPort"`
}

// TokenDiscovery contains elements needed for token discovery.
type TokenDiscovery struct {
	// ID is the first part of a bootstrap token. Considered public information.
	// It is used when referring to a token without leaking the secret part.
	ID string `json:"id"`
	// Secret is the second part of a bootstrap token. Should only be shared
	// with trusted parties.
	Secret string `json:"secret"`
	// TODO: Seems unused. Remove?
	// Addresses []string `json:"addresses"`
}

// Networking contains elements describing cluster's networking configuration
type Networking struct {
	// ServiceSubnet is the subnet used by k8s services. Defaults to "10.96.0.0/12".
	ServiceSubnet string `json:"serviceSubnet"`
	// PodSubnet is the subnet used by pods.
	PodSubnet string `json:"podSubnet"`
	// DNSDomain is the dns domain used by k8s services. Defaults to "cluster.local".
	DNSDomain string `json:"dnsDomain"`
}

// Etcd contains elements describing Etcd configuration.
type Etcd struct {
	// Endpoints of etcd members. Useful for using external etcd.
	// If not provided, kubeadm will run etcd in a static pod.
	Endpoints []string `json:"endpoints"`
	// CAFile is an SSL Certificate Authority file used to secure etcd communication.
	CAFile string `json:"caFile"`
	// CertFile is an SSL certification file used to secure etcd communication.
	CertFile string `json:"certFile"`
	// KeyFile is an SSL key file used to secure etcd communication.
	KeyFile string `json:"keyFile"`
	// DataDir is the directory etcd will place its data.
	// Defaults to "/var/lib/etcd".
	DataDir string `json:"dataDir"`
	// ExtraArgs are extra arguments provided to the etcd binary
	// when run inside a static pod.
	ExtraArgs map[string]string `json:"extraArgs,omitempty"`
	// Image specifies which container image to use for running etcd.
	// If empty, automatically populated by kubeadm using the image
	// repository and default etcd version.
	Image string `json:"image"`
	// SelfHosted holds configuration for self-hosting etcd.
	SelfHosted *SelfHostedEtcd `json:"selfHosted,omitempty"`
	// ServerCertSANs sets extra Subject Alternative Names for the etcd server signing cert.
	ServerCertSANs []string `json:"serverCertSANs,omitempty"`
	// PeerCertSANs sets extra Subject Alternative Names for the etcd peer signing cert.
	PeerCertSANs []string `json:"peerCertSANs,omitempty"`
}

// SelfHostedEtcd describes options required to configure self-hosted etcd.
type SelfHostedEtcd struct {
	// CertificatesDir represents the directory where all etcd TLS assets are stored.
	// Defaults to "/etc/kubernetes/pki/etcd".
	CertificatesDir string `json:"certificatesDir"`
	// ClusterServiceName is the name of the service that load balances the etcd cluster.
	ClusterServiceName string `json:"clusterServiceName"`
	// EtcdVersion is the version of etcd running in the cluster.
	EtcdVersion string `json:"etcdVersion"`
	// OperatorVersion is the version of the etcd-operator to use.
	OperatorVersion string `json:"operatorVersion"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// NodeConfiguration contains elements describing a particular node.
// TODO: This struct should be replaced by dynamic kubelet configuration.
type NodeConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// CACertPath is the path to the SSL certificate authority used to
	// secure comunications between node and master.
	// Defaults to "/etc/kubernetes/pki/ca.crt".
	CACertPath string `json:"caCertPath"`
	// DiscoveryFile is a file or url to a kubeconfig file from which to
	// load cluster information.
	DiscoveryFile string `json:"discoveryFile"`
	// DiscoveryToken is a token used to validate cluster information
	// fetched from the master.
	DiscoveryToken string `json:"discoveryToken"`
	// DiscoveryTokenAPIServers is a set of IPs to API servers from which info
	// will be fetched. Currently we only pay attention to one API server but
	// hope to support >1 in the future.
	DiscoveryTokenAPIServers []string `json:"discoveryTokenAPIServers,omitempty"`
	// DiscoveryTimeout modifies the discovery timeout
	DiscoveryTimeout *metav1.Duration `json:"discoveryTimeout,omitempty"`
	// NodeName is the name of the node to join the cluster. Defaults
	// to the name of the host.
	NodeName string `json:"nodeName"`
	// TLSBootstrapToken is a token used for TLS bootstrapping.
	// Defaults to Token.
	TLSBootstrapToken string `json:"tlsBootstrapToken"`
	// Token is used for both discovery and TLS bootstrapping.
	Token string `json:"token"`
	// CRISocket is used to retrieve container runtime info.
	CRISocket string `json:"criSocket,omitempty"`
	// ClusterName is the name for the cluster in kubeconfig.
	ClusterName string `json:"clusterName,omitempty"`

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

	// FeatureGates enabled by the user.
	FeatureGates map[string]bool `json:"featureGates,omitempty"`
}

// KubeletConfiguration contains elements describing initial remote configuration of kubelet.
type KubeletConfiguration struct {
	BaseConfig *kubeletconfigv1beta1.KubeletConfiguration `json:"baseConfig,omitempty"`
}

// HostPathMount contains elements describing volumes that are mounted from the
// host.
type HostPathMount struct {
	// Name of the volume inside the pod template.
	Name string `json:"name"`
	// HostPath is the path in the host that will be mounted inside
	// the pod.
	HostPath string `json:"hostPath"`
	// MountPath is the path inside the pod where hostPath will be mounted.
	MountPath string `json:"mountPath"`
	// Writable controls write access to the volume
	Writable bool `json:"writable,omitempty"`
	// PathType is the type of the HostPath.
	PathType v1.HostPathType `json:"pathType,omitempty"`
}

// KubeProxy contains elements describing the proxy configuration.
type KubeProxy struct {
	Config *kubeproxyconfigv1alpha1.KubeProxyConfiguration `json:"config,omitempty"`
}

// AuditPolicyConfiguration holds the options for configuring the api server audit policy.
type AuditPolicyConfiguration struct {
	// Path is the local path to an audit policy.
	Path string `json:"path"`
	// LogDir is the local path to the directory where logs should be stored.
	LogDir string `json:"logDir"`
	// LogMaxAge is the number of days logs will be stored for. 0 indicates forever.
	LogMaxAge *int32 `json:"logMaxAge,omitempty"`
	//TODO(chuckha) add other options for audit policy.
}
