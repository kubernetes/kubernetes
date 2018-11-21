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
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InitConfiguration contains a list of elements that is specific "kubeadm init"-only runtime
// information.
type InitConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// ClusterConfiguration holds the cluster-wide information, and embeds that struct (which can be (un)marshalled separately as well)
	// When InitConfiguration is marshalled to bytes in the external version, this information IS NOT preserved (which can be seen from
	// the `json:"-"` tag. This is due to that when InitConfiguration is (un)marshalled, it turns into two YAML documents, one for the
	// InitConfiguration and ClusterConfiguration. Hence, the information must not be duplicated, and is therefore omitted here.
	ClusterConfiguration `json:"-"`

	// `kubeadm init`-only information. These fields are solely used the first time `kubeadm init` runs.
	// After that, the information in the fields ARE NOT uploaded to the `kubeadm-config` ConfigMap
	// that is used by `kubeadm upgrade` for instance. These fields must be omitempty.

	// BootstrapTokens is respected at `kubeadm init` time and describes a set of Bootstrap Tokens to create.
	// This information IS NOT uploaded to the kubeadm cluster configmap, partly because of its sensitive nature
	BootstrapTokens []BootstrapToken `json:"bootstrapTokens,omitempty"`

	// NodeRegistration holds fields that relate to registering the new master node to the cluster
	NodeRegistration NodeRegistrationOptions `json:"nodeRegistration,omitempty"`

	// APIEndpoint represents the endpoint of the instance of the API server to be deployed on this node.
	APIEndpoint APIEndpoint `json:"apiEndpoint,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterConfiguration contains cluster-wide configuration for a kubeadm cluster
type ClusterConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Etcd holds configuration for etcd.
	Etcd Etcd `json:"etcd"`

	// Networking holds configuration for the networking topology of the cluster.
	Networking Networking `json:"networking"`

	// KubernetesVersion is the target version of the control plane.
	KubernetesVersion string `json:"kubernetesVersion"`

	// ControlPlaneEndpoint sets a stable IP address or DNS name for the control plane; it
	// can be a valid IP address or a RFC-1123 DNS subdomain, both with optional TCP port.
	// In case the ControlPlaneEndpoint is not specified, the AdvertiseAddress + BindPort
	// are used; in case the ControlPlaneEndpoint is specified but without a TCP port,
	// the BindPort is used.
	// Possible usages are:
	// e.g. In a cluster with more than one control plane instances, this field should be
	// assigned the address of the external load balancer in front of the
	// control plane instances.
	// e.g.  in environments with enforced node recycling, the ControlPlaneEndpoint
	// could be used for assigning a stable DNS to the control plane.
	ControlPlaneEndpoint string `json:"controlPlaneEndpoint"`

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

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterStatus contains the cluster status. The ClusterStatus will be stored in the kubeadm-config
// ConfigMap in the cluster, and then updated by kubeadm when additional control plane instance joins or leaves the cluster.
type ClusterStatus struct {
	metav1.TypeMeta `json:",inline"`

	// APIEndpoints currently available in the cluster, one for each control plane/api server instance.
	// The key of the map is the IP of the host's default interface
	APIEndpoints map[string]APIEndpoint `json:"apiEndpoints"`
}

// APIEndpoint struct contains elements of API server instance deployed on a node.
type APIEndpoint struct {
	// AdvertiseAddress sets the IP address for the API server to advertise.
	AdvertiseAddress string `json:"advertiseAddress"`

	// BindPort sets the secure port for the API Server to bind to.
	// Defaults to 6443.
	BindPort int32 `json:"bindPort"`
}

// NodeRegistrationOptions holds fields that relate to registering a new master or node to the cluster, either via "kubeadm init" or "kubeadm join"
type NodeRegistrationOptions struct {

	// Name is the `.Metadata.Name` field of the Node API object that will be created in this `kubeadm init` or `kubeadm joiń` operation.
	// This field is also used in the CommonName field of the kubelet's client certificate to the API server.
	// Defaults to the hostname of the node if not provided.
	Name string `json:"name,omitempty"`

	// CRISocket is used to retrieve container runtime info. This information will be annotated to the Node API object, for later re-use
	CRISocket string `json:"criSocket,omitempty"`

	// Taints specifies the taints the Node API object should be registered with. If this field is unset, i.e. nil, in the `kubeadm init` process
	// it will be defaulted to []v1.Taint{'node-role.kubernetes.io/master=""'}. If you don't want to taint your master node, set this field to an
	// empty slice, i.e. `taints: {}` in the YAML file. This field is solely used for Node registration.
	Taints []v1.Taint `json:"taints,omitempty"`

	// KubeletExtraArgs passes through extra arguments to the kubelet. The arguments here are passed to the kubelet command line via the environment file
	// kubeadm writes at runtime for the kubelet to source. This overrides the generic base-level configuration in the kubelet-config-1.X ConfigMap
	// Flags have higher priority when parsing. These values are local and specific to the node kubeadm is executing on.
	KubeletExtraArgs map[string]string `json:"kubeletExtraArgs,omitempty"`
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

// BootstrapToken describes one bootstrap token, stored as a Secret in the cluster
type BootstrapToken struct {
	// Token is used for establishing bidirectional trust between nodes and masters.
	// Used for joining nodes in the cluster.
	Token *BootstrapTokenString `json:"token"`
	// Description sets a human-friendly message why this token exists and what it's used
	// for, so other administrators can know its purpose.
	Description string `json:"description,omitempty"`
	// TTL defines the time to live for this token. Defaults to 24h.
	// Expires and TTL are mutually exclusive.
	TTL *metav1.Duration `json:"ttl,omitempty"`
	// Expires specifies the timestamp when this token expires. Defaults to being set
	// dynamically at runtime based on the TTL. Expires and TTL are mutually exclusive.
	Expires *metav1.Time `json:"expires,omitempty"`
	// Usages describes the ways in which this token can be used. Can by default be used
	// for establishing bidirectional trust, but that can be changed here.
	Usages []string `json:"usages,omitempty"`
	// Groups specifies the extra groups that this token will authenticate as when/if
	// used for authentication
	Groups []string `json:"groups,omitempty"`
}

// Etcd contains elements describing Etcd configuration.
type Etcd struct {

	// Local provides configuration knobs for configuring the local etcd instance
	// Local and External are mutually exclusive
	Local *LocalEtcd `json:"local,omitempty"`

	// External describes how to connect to an external etcd cluster
	// Local and External are mutually exclusive
	External *ExternalEtcd `json:"external,omitempty"`
}

// LocalEtcd describes that kubeadm should run an etcd cluster locally
type LocalEtcd struct {

	// Image specifies which container image to use for running etcd.
	// If empty, automatically populated by kubeadm using the image
	// repository and default etcd version.
	Image string `json:"image"`

	// DataDir is the directory etcd will place its data.
	// Defaults to "/var/lib/etcd".
	DataDir string `json:"dataDir"`

	// ExtraArgs are extra arguments provided to the etcd binary
	// when run inside a static pod.
	ExtraArgs map[string]string `json:"extraArgs,omitempty"`

	// ServerCertSANs sets extra Subject Alternative Names for the etcd server signing cert.
	ServerCertSANs []string `json:"serverCertSANs,omitempty"`
	// PeerCertSANs sets extra Subject Alternative Names for the etcd peer signing cert.
	PeerCertSANs []string `json:"peerCertSANs,omitempty"`
}

// ExternalEtcd describes an external etcd cluster
type ExternalEtcd struct {
	// Endpoints of etcd members. Required for ExternalEtcd.
	Endpoints []string `json:"endpoints"`
	// CAFile is an SSL Certificate Authority file used to secure etcd communication.
	CAFile string `json:"caFile"`
	// CertFile is an SSL certification file used to secure etcd communication.
	CertFile string `json:"certFile"`
	// KeyFile is an SSL key file used to secure etcd communication.
	KeyFile string `json:"keyFile"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// JoinConfiguration contains elements describing a particular node.
// TODO: This struct should be replaced by dynamic kubelet configuration.
type JoinConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// NodeRegistration holds fields that relate to registering the new master node to the cluster
	NodeRegistration NodeRegistrationOptions `json:"nodeRegistration"`

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
	// TLSBootstrapToken is a token used for TLS bootstrapping.
	// Defaults to Token.
	TLSBootstrapToken string `json:"tlsBootstrapToken"`
	// Token is used for both discovery and TLS bootstrapping.
	Token string `json:"token"`

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

	// ControlPlane flag specifies that the joining node should host an additional
	// control plane instance.
	ControlPlane bool `json:"controlPlane,omitempty"`

	// APIEndpoint represents the endpoint of the instance of the API server eventually to be deployed on this node.
	APIEndpoint APIEndpoint `json:"apiEndpoint,omitempty"`

	// FeatureGates enabled by the user.
	FeatureGates map[string]bool `json:"featureGates,omitempty"`
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
