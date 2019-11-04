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

package v1beta3

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InitConfiguration contains a list of elements that is specific "kubeadm init"-only runtime
// information.
type InitConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Optional: Ignored in InitConfiguration
	// +optional
	ObjectMeta `json:"metadata,omitempty"`

	// `kubeadm init`-only information. These fields are solely used the first time `kubeadm init` runs.
	// After that, the information in the fields IS NOT uploaded to the `kubeadm-config` ConfigMap
	// that is used by `kubeadm upgrade` for instance. These fields must be omitempty.

	// BootstrapTokens is respected at `kubeadm init` time and describes a set of Bootstrap Tokens to create.
	// This information IS NOT uploaded to the kubeadm cluster configmap, partly because of its sensitive nature
	// Optional: If not specified, a token is generated automatically by kubeadm
	// +optional
	BootstrapTokens []BootstrapToken `json:"bootstrapTokens,omitempty"`

	// NodeRegistration holds fields that relate to registering the new control-plane node to the cluster
	NodeRegistration NodeRegistrationOptions `json:"nodeRegistration,omitempty"`

	// LocalAPIEndpoint represents the endpoint of the API server instance that's deployed on this control plane node
	// In HA setups, this differs from ClusterConfiguration.ControlPlaneEndpoint in the sense that ControlPlaneEndpoint
	// is the global endpoint for the cluster, which then loadbalances the requests to each individual API server. This
	// configuration object lets you customize what IP/DNS name and port the local API server advertises it's accessible
	// on. By default, kubeadm tries to auto-detect the IP of the default interface and use that, but in case that process
	// fails you may set the desired value here.
	LocalAPIEndpoint APIEndpoint `json:"localAPIEndpoint,omitempty"`

	// CertificateKey sets the key with which certificates and keys are encrypted prior to being uploaded in
	// a secret in the cluster during the uploadcerts init phase.
	// Optional: If not specified, a key will be generated upon certificate upload
	// +optional
	CertificateKey string `json:"certificateKey,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterConfiguration contains cluster-wide configuration for a kubeadm cluster
type ClusterConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Optional: Gets automatically defaulted if not initialized
	// +optional
	ObjectMeta `json:"metadata,omitempty"`

	// Etcd holds configuration for etcd.
	Etcd Etcd `json:"etcd,omitempty"`

	// Networking holds configuration for the networking topology of the cluster.
	Networking Networking `json:"networking,omitempty"`

	// KubernetesVersion is the target version of the control plane.
	// Optional: Defaulted to latest stable 1.x.y release. Internet connection is required for the defaulting.
	// +optional
	KubernetesVersion string `json:"kubernetesVersion,omitempty"`

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
	// Optional: If unspecified, AdvertiseAddress + BindPort are used (see above)
	// +optional
	ControlPlaneEndpoint string `json:"controlPlaneEndpoint,omitempty"`

	// APIServer contains extra settings for the API server control plane component
	APIServer APIServer `json:"apiServer,omitempty"`

	// ControllerManager contains extra settings for the controller manager control plane component
	ControllerManager ControlPlaneComponent `json:"controllerManager,omitempty"`

	// Scheduler contains extra settings for the scheduler control plane component
	Scheduler ControlPlaneComponent `json:"scheduler,omitempty"`

	// AddOns defines a list of addons deployed in this cluster
	// Optional: Defaulted to KubeProxy and CoreDNS, omitempty is intentionally skipped
	// +optional
	AddOns []AddOn `json:"addons"`

	// CertificatesDir specifies where to store or look for all required certificates.
	// Optional: Defaulted to "/etc/kubernetes/pki"
	// +optional
	CertificatesDir string `json:"certificatesDir,omitempty"`

	// ImageRepository sets the container registry to pull images from.
	// If empty, `k8s.gcr.io` will be used by default; in case of kubernetes version is a CI build (kubernetes version starts with `ci/` or `ci-cross/`)
	// `gcr.io/kubernetes-ci-images` will be used as a default for control plane components and for kube-proxy, while `k8s.gcr.io`
	// will be used for all the other images.
	// Optional: Defaulted to "k8s.gcr.io" or "gcr.io/kubernetes-ci-images" (see above)
	// +optional
	ImageRepository string `json:"imageRepository,omitempty"`

	// UseHyperKubeImage controls if hyperkube should be used for Kubernetes components instead of their respective separate images
	// Optional: Defaulted to false
	// +optional
	UseHyperKubeImage bool `json:"useHyperKubeImage,omitempty"`

	// FeatureGates enabled by the user.
	// Optional: No features will be enabled or disabled if unspecified. See kubeadm documentation for more information.
	// +optional
	FeatureGates map[string]bool `json:"featureGates,omitempty"`
}

// ControlPlaneComponent holds settings common to control plane component of the cluster
type ControlPlaneComponent struct {
	// ExtraArgs is an extra set of flags to pass to the control plane component.
	// TODO: This is temporary and ideally we would like to switch all components to
	// use ComponentConfig + ConfigMaps.
	// Optional: Not needed for normal operation
	// +optional
	ExtraArgs map[string]string `json:"extraArgs,omitempty"`

	// ExtraVolumes is an extra set of host volumes, mounted to the control plane component.
	// Optional: Not needed for normal operation
	// +optional
	ExtraVolumes []HostPathMount `json:"extraVolumes,omitempty"`
}

// APIServer holds settings necessary for API server deployments in the cluster
type APIServer struct {
	ControlPlaneComponent `json:",inline"`

	// CertSANs sets extra Subject Alternative Names for the API Server signing cert.
	// Optional: Not needed for normal operation
	// +optional
	CertSANs []string `json:"certSANs,omitempty"`

	// TimeoutForControlPlane controls the timeout that we use for API server to appear
	// Optional: Defaulted to 4 minutes
	// +optional
	TimeoutForControlPlane *metav1.Duration `json:"timeoutForControlPlane,omitempty"`
}

// ImageMeta allows to customize the image used for components that are not
// originated from the Kubernetes/Kubernetes release process
type ImageMeta struct {
	// ImageRepository sets the container registry to pull images from.
	// Optional: If not set, the ImageRepository defined in ClusterConfiguration will be used instead.
	// +optional
	ImageRepository string `json:"imageRepository,omitempty"`

	// ImageTag allows to specify a tag for the image.
	// Optional: In case this value is set, kubeadm does not change automatically the version of the above components during upgrades.
	// +optional
	ImageTag string `json:"imageTag,omitempty"`

	//TODO: evaluate if we need also a ImageName based on user feedbacks
}

// AddOn defines settings to be used when deploying a specific addon kind
type AddOn struct {
	// Kind defines the addon type
	Kind string `json:"kind"`

	// ImageMeta allows to customize the image used for the addon
	ImageMeta `json:",inline"`
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
	// Optional: If unspecified, kubeadm will select an IP address on its own.
	// +optional
	AdvertiseAddress string `json:"advertiseAddress,omitempty"`

	// BindPort sets the secure port for the API Server to bind to.
	// Optional: Defaults to 6443.
	// +optional
	BindPort int32 `json:"bindPort,omitempty"`
}

// NodeRegistrationOptions holds fields that relate to registering a new control-plane or node to the cluster, either via "kubeadm init" or "kubeadm join"
type NodeRegistrationOptions struct {

	// Name is the `.Metadata.Name` field of the Node API object that will be created in this `kubeadm init` or `kubeadm join` operation.
	// This field is also used in the CommonName field of the kubelet's client certificate to the API server.
	// Optional: Defaults to the hostname of the node if not provided.
	// +optional
	Name string `json:"name,omitempty"`

	// CRISocket is used to retrieve container runtime info. This information will be annotated to the Node API object, for later re-use
	// Optional: kubeadm will attempt to detect the CRI and its socket if none is provided here
	// +optional
	CRISocket string `json:"criSocket,omitempty"`

	// Taints specifies the taints the Node API object should be registered with. If this field is unset, i.e. nil, in the `kubeadm init` process
	// it will be defaulted to []v1.Taint{'node-role.kubernetes.io/master=""'}. If you don't want to taint your control-plane node, set this field to an
	// empty slice, i.e. `taints: []` in the YAML file. This field is solely used for Node registration.
	Taints []v1.Taint `json:"taints"`

	// KubeletExtraArgs passes through extra arguments to the kubelet. The arguments here are passed to the kubelet command line via the environment file
	// kubeadm writes at runtime for the kubelet to source. This overrides the generic base-level configuration in the kubelet-config-1.X ConfigMap
	// Flags have higher priority when parsing. These values are local and specific to the node kubeadm is executing on.
	// Optional: Not needed for normal operation
	// +optional
	KubeletExtraArgs map[string]string `json:"kubeletExtraArgs,omitempty"`

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored when the current node is registered.
	// Optional: Not needed for normal operation
	// +optional
	IgnorePreflightErrors []string `json:"ignorePreflightErrors,omitempty"`
}

// Networking contains elements describing cluster's networking configuration
type Networking struct {
	// ServiceSubnet is the subnet used by k8s services.
	// Optional: Defaults to "10.96.0.0/12".
	// +optional
	ServiceSubnet string `json:"serviceSubnet,omitempty"`
	// PodSubnet is the subnet used by pods.
	// Optional: No pod CIDR ranges are specified to either the Controller Manager or Kube-Proxy if this field is empty.
	// +optional
	PodSubnet string `json:"podSubnet,omitempty"`
	// DNSDomain is the dns domain used by k8s services.
	// Optional: Defaults to "cluster.local".
	// +optional
	DNSDomain string `json:"dnsDomain,omitempty"`
}

// BootstrapToken describes one bootstrap token, stored as a Secret in the cluster
type BootstrapToken struct {
	// Token is used for establishing bidirectional trust between nodes and control-planes.
	// Used for joining nodes in the cluster.
	Token *BootstrapTokenString `json:"token"`
	// Description sets a human-friendly message why this token exists and what it's used
	// for, so other administrators can know its purpose.
	// Optional: Not needed for normal operation
	// +optional
	Description string `json:"description,omitempty"`
	// TTL defines the time to live for this token.
	// Expires and TTL are mutually exclusive.
	// Optional: Defaults to 24h.
	// +optional
	TTL *metav1.Duration `json:"ttl,omitempty"`
	// Expires specifies the timestamp when this token expires.
	// Expires and TTL are mutually exclusive.
	// Optional: Defaults to being set dynamically at runtime based on the TTL.
	// +optional
	Expires *metav1.Time `json:"expires,omitempty"`
	// Usages describes the ways in which this token can be used. Can by default be used
	// for establishing bidirectional trust, but that can be changed here.
	// Optional: Defaults to `signing` and `authentication` usages
	// +optional
	Usages []string `json:"usages,omitempty"`
	// Groups specifies the extra groups that this token will authenticate as when/if
	// used for authentication
	// Optional: Defaults to the `system:bootstrappers:kubeadm:default-node-token` group
	// +optional
	Groups []string `json:"groups,omitempty"`
}

// Etcd contains elements describing Etcd configuration.
type Etcd struct {

	// Local provides configuration knobs for configuring the local etcd instance
	// Local and External are mutually exclusive
	// Optional: kubeadm will use a defaulted Local field if none is supplied
	// +optional
	Local *LocalEtcd `json:"local,omitempty"`

	// External describes how to connect to an external etcd cluster
	// Local and External are mutually exclusive
	// Optional: kubeadm will use local etcd if no External is supplied
	// +optional
	External *ExternalEtcd `json:"external,omitempty"`
}

// LocalEtcd describes that kubeadm should run an etcd cluster locally
type LocalEtcd struct {
	// ImageMeta allows to customize the container used for etcd
	ImageMeta `json:",inline"`

	// DataDir is the directory etcd will place its data.
	// Defaults to "/var/lib/etcd".
	DataDir string `json:"dataDir"`

	// ExtraArgs are extra arguments provided to the etcd binary
	// when run inside a static pod.
	// Optional: Not needed for normal operation
	// +optional
	ExtraArgs map[string]string `json:"extraArgs,omitempty"`

	// ServerCertSANs sets extra Subject Alternative Names for the etcd server signing cert.
	// Optional: Not needed for normal operation
	// +optional
	ServerCertSANs []string `json:"serverCertSANs,omitempty"`
	// PeerCertSANs sets extra Subject Alternative Names for the etcd peer signing cert.
	// Optional: Not needed for normal operation
	// +optional
	PeerCertSANs []string `json:"peerCertSANs,omitempty"`
}

// ExternalEtcd describes an external etcd cluster.
// Kubeadm has no knowledge of where certificate files live and they must be supplied.
type ExternalEtcd struct {
	// Endpoints of etcd members. Required for ExternalEtcd.
	Endpoints []string `json:"endpoints"`

	// CAFile is an SSL Certificate Authority file used to secure etcd communication.
	// Required if using a TLS connection.
	CAFile string `json:"caFile"`

	// CertFile is an SSL certification file used to secure etcd communication.
	// Required if using a TLS connection.
	CertFile string `json:"certFile"`

	// KeyFile is an SSL key file used to secure etcd communication.
	// Required if using a TLS connection.
	KeyFile string `json:"keyFile"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// JoinConfiguration contains elements describing a particular node.
type JoinConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Optional: Ingored in JoinConfiguration
	// +optional
	ObjectMeta `json:"metadata,omitempty"`

	// NodeRegistration holds fields that relate to registering the new control-plane node to the cluster
	NodeRegistration NodeRegistrationOptions `json:"nodeRegistration,omitempty"`

	// CACertPath is the path to the SSL certificate authority used to
	// secure comunications between node and control-plane.
	// Optional: Defaults to "/etc/kubernetes/pki/ca.crt".
	// +optional
	CACertPath string `json:"caCertPath,omitempty"`

	// Discovery specifies the options for the kubelet to use during the TLS Bootstrap process
	Discovery Discovery `json:"discovery"`

	// ControlPlane defines the additional control plane instance to be deployed on the joining node.
	// If nil, no additional control plane instance will be deployed.
	// Optional: If not supplied, the node won't be joined as a control plane one.
	// +optional
	ControlPlane *JoinControlPlane `json:"controlPlane,omitempty"`
}

// JoinControlPlane contains elements describing an additional control plane instance to be deployed on the joining node.
type JoinControlPlane struct {
	// LocalAPIEndpoint represents the endpoint of the API server instance to be deployed on this node.
	LocalAPIEndpoint APIEndpoint `json:"localAPIEndpoint,omitempty"`

	// CertificateKey is the key that is used for decryption of certificates after they are downloaded from the secret
	// upon joining a new control plane node. The corresponding encryption key is in the InitConfiguration.
	// Optional: Not needed if no cluster uploaded certificates are to be used
	// +optional
	CertificateKey string `json:"certificateKey,omitempty"`
}

// Discovery specifies the options for the kubelet to use during the TLS Bootstrap process
type Discovery struct {
	// BootstrapToken is used to set the options for bootstrap token based discovery
	// Optional: BootstrapToken and File are mutually exclusive
	// +optional
	BootstrapToken *BootstrapTokenDiscovery `json:"bootstrapToken,omitempty"`

	// File is used to specify a file or URL to a kubeconfig file from which to load cluster information
	// Optional: BootstrapToken and File are mutually exclusive
	// +optional
	File *FileDiscovery `json:"file,omitempty"`

	// TLSBootstrapToken is a token used for TLS bootstrapping.
	// Optional: If .BootstrapToken is set, this field is defaulted to .BootstrapToken.Token, but can be overridden.
	// Required: If .File is set, this field **must be set** in case the KubeConfigFile does not contain any other authentication information
	// +optional
	TLSBootstrapToken string `json:"tlsBootstrapToken,omitempty"`

	// Timeout modifies the discovery timeout
	// Optional: Defaulted to 5 minutes
	// +optional
	Timeout *metav1.Duration `json:"timeout,omitempty"`
}

// BootstrapTokenDiscovery is used to set the options for bootstrap token based discovery
type BootstrapTokenDiscovery struct {
	// Token is a token used to validate cluster information
	// fetched from the control-plane.
	Token string `json:"token"`

	// APIServerEndpoint is an IP or domain name to the API server from which info will be fetched.
	// Optional: The command line overwrites this field even if set
	// +optional
	APIServerEndpoint string `json:"apiServerEndpoint,omitempty"`

	// CACertHashes specifies a set of public key pins to verify
	// when token-based discovery is used. The root CA found during discovery
	// must match one of these values. Specifying an empty set disables root CA
	// pinning, which can be unsafe. Each hash is specified as "<type>:<value>",
	// where the only currently supported type is "sha256". This is a hex-encoded
	// SHA-256 hash of the Subject Public Key Info (SPKI) object in DER-encoded
	// ASN.1. These hashes can be calculated using, for example, OpenSSL:
	// openssl x509 -pubkey -in ca.crt openssl rsa -pubin -outform der 2>&/dev/null | openssl dgst -sha256 -hex
	// Optional: Not required if UnsafeSkipCAVerification=true
	// +optional
	CACertHashes []string `json:"caCertHashes,omitempty"`

	// UnsafeSkipCAVerification allows token-based discovery
	// without CA verification via CACertHashes. This can weaken
	// the security of kubeadm since other nodes can impersonate the control-plane.
	// Optional: Defaulted to false
	// +optional
	UnsafeSkipCAVerification bool `json:"unsafeSkipCAVerification,omitempty"`
}

// FileDiscovery is used to specify a file or URL to a kubeconfig file from which to load cluster information
type FileDiscovery struct {
	// KubeConfigPath is used to specify the actual file path or URL to the kubeconfig file from which to load cluster information
	KubeConfigPath string `json:"kubeConfigPath"`
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
	// ReadOnly controls write access to the volume
	// Optional: Defaulted to false
	// +optional
	ReadOnly bool `json:"readOnly,omitempty"`
	// PathType is the type of the HostPath.
	// Optional: Defaulted to ""
	// +optional
	PathType v1.HostPathType `json:"pathType,omitempty"`
}

// ObjectMeta is a cut down version on metav1.ObjectMeta with the bare minimum that is sensible in
// kubeadm terms and satisfies the requirements of Kustomize.
// The original metav1.ObjectMeta proved too large and some unneeded fields broke a bunch of tests (like the fuzzer).
type ObjectMeta struct {
	// Name is an unique name to identify the object with. In the terms of kubeadm, this is the cluster name.
	// Optional: Defaulted to "kubernetes" only when ObjectMeta is embedded in ClusterConfiguration.
	// +optional
	Name string `json:"name,omitempty"`
}
