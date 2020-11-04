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

package v1beta1

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED - This group version of InitConfiguration is deprecated by apis/kubeadm/v1beta2.InitConfiguration.
// InitConfiguration contains runtime information that are specific to "kubeadm init".
type InitConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// This field holds the cluster-wide information, and embeds that struct (which can be (un)marshalled separately as well)
	// When InitConfiguration is marshalled to bytes in the external version, this information IS NOT preserved (which can be seen from
	// the `json:"-"` tag. This is due to that when InitConfiguration is (un)marshalled, it turns into two YAML documents, one for the
	// InitConfiguration and ClusterConfiguration. Hence, the information must not be duplicated, and is therefore omitted here.
	ClusterConfiguration `json:"-"`

	// `bootstrapTokens` describes a set of Bootstrap Tokens to create during `kubeadm init`.
	// This information is NOT uploaded to the `kubeadm-config` ConfigMap, partly because of its sensitive nature
	BootstrapTokens []BootstrapToken `json:"bootstrapTokens,omitempty"`

	// `nodeRegistration` holds fields related to registering the new control-plane node to the cluster.
	NodeRegistration NodeRegistrationOptions `json:"nodeRegistration,omitempty"`

	// `localAPIEndpoint` represents the endpoint of the API server instance that's deployed on this control plane
	// instance.  In HA setups, this differs from `ClusterConfiguration.controlPlaneEndpoint` in the sense that
	// `controlPlaneEndpoint` is the global endpoint for the cluster, which loadbalances the requests to each individual
	// API server. This configuration object lets you customize what IP/DNS name and port on which the local API server
	// is accessible. By default, kubeadm tries to auto-detect the IP of the default interface and use that, but in
	// case that process fails you may set the desired value here.
	LocalAPIEndpoint APIEndpoint `json:"localAPIEndpoint,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED - This group version of ClusterConfiguration is deprecated by apis/kubeadm/v1beta2.ClusterConfiguration.
// ClusterConfiguration contains cluster-wide configuration for a kubeadm cluster.
type ClusterConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// `etcd` holds configuration for etcd.
	Etcd Etcd `json:"etcd"`

	// `networking` holds configuration for the networking topology of the cluster.
	Networking Networking `json:"networking"`

	// `kubernetesVersion` is the target version of the control plane.
	KubernetesVersion string `json:"kubernetesVersion"`

	// `controlPlaneEndpoint` sets a stable IP address or DNS name for the control plane.
	// It can be a valid IP address or a RFC-1123 DNS subdomain, both with optional TCP port.
	// If `controlPlaneEndpoint` is not specified, the `advertiseAddress` + `bindPort`
	// are used. If `controlPlaneEndpoint` is specified without a TCP port, the `bindPort` is used.
	// Possible usages are:
	//
	// - In a cluster with more than one control plane nodes, this field should be
	//   assigned the address of the external load balancer in front of the
	//   control plane nodes.
	// - In environments with enforced node recycling, the `controlPlaneEndpoint`
	//   could be used for assigning a stable DNS to the control plane.
	ControlPlaneEndpoint string `json:"controlPlaneEndpoint"`

	// `apiServer` contains extra settings for the API server.
	APIServer APIServer `json:"apiServer,omitempty"`

	// `controllerManager` contains extra settings for the controller manager.
	ControllerManager ControlPlaneComponent `json:"controllerManager,omitempty"`

	// `scheduler` contains extra settings for the scheduler.
	Scheduler ControlPlaneComponent `json:"scheduler,omitempty"`

	// `dns` defines the options for the DNS add-on installed in the cluster.
	DNS DNS `json:"dns"`

	// `certificatesDir` specifies where to store or look for all required certificates.
	CertificatesDir string `json:"certificatesDir"`

	// `imageRepository` specifies the container registry from which images are pulled.
	// If empty, `k8s.gcr.io` will be used. If kubernetes version is a CI build (starts with `ci/` or `ci-cross/`)
	// `gcr.io/kubernetes-ci-images` will be used for control plane components and
	// kube-proxy, while `k8s.gcr.io` will be used for all the other images.
	ImageRepository string `json:"imageRepository"`

	// `useHyperKubeImage` controls if hyperkube should be used for Kubernetes components
	// instead of their respective separate images.
	// DEPRECATED: As hyperkube is deprecated, this field is deprecated too.
	// It will be removed in future kubeadm config versions. Kubeadm may print multiple
	// warnings or ignore it when this is set to true.
	UseHyperKubeImage bool `json:"useHyperKubeImage,omitempty"`

	// `featureGates` is a map containing the feature gates to be enabled.
	FeatureGates map[string]bool `json:"featureGates,omitempty"`

	// `clusterName` contains the cluster name.
	ClusterName string `json:"clusterName,omitempty"`
}

// ControlPlaneComponent holds settings common to control plane component of the cluster
type ControlPlaneComponent struct {
	// `extraArgs` is an extra set of flags to pass to the control plane components.
	// TODO: This is temporary and ideally we would like to switch all components to
	// use ComponentConfig + ConfigMaps.
	ExtraArgs map[string]string `json:"extraArgs,omitempty"`

	// `extraVolumes` is an extra set of HostPath volumes to be mounted by the control plane component.
	ExtraVolumes []HostPathMount `json:"extraVolumes,omitempty"`
}

// APIServer holds settings necessary for API server instances in the cluster
type APIServer struct {
	ControlPlaneComponent `json:",inline"`

	// `certSANs` sets extra Subject Alternative Names (SANs) for the API Server signing cert.
	CertSANs []string `json:"certSANs,omitempty"`

	// `timeoutForControlPlane` controls the timeout that kubeadm waits for the API server to appear.
	TimeoutForControlPlane *metav1.Duration `json:"timeoutForControlPlane,omitempty"`
}

// DNSAddOnType defines string identifying DNS add-on types.
type DNSAddOnType string

const (
	// CoreDNS add-on type
	CoreDNS DNSAddOnType = "CoreDNS"

	// KubeDNS add-on type
	KubeDNS DNSAddOnType = "kube-dns"
)

// DNS defines the DNS add-on that should be used in the cluster
type DNS struct {
	// `type` defines the DNS add-on to be used. Can be one of "CoreDNS" or "kube-dns".
	Type DNSAddOnType `json:"type"`

	// `imageMeta` is used to customize the image used for the DNS add-on.
	ImageMeta `json:",inline"`
}

// ImageMeta allows to customize the image used for components that are not
// originated from the Kubernetes/Kubernetes release process
type ImageMeta struct {
	// `imageRepository` sets the container registry to pull images from.
	// If not set, the `imageRepository` defined in ClusterConfiguration will be used instead.
	ImageRepository string `json:"imageRepository,omitempty"`

	// `imageTag` allows for specifying a tag for the image.
	// When this value is set, kubeadm does not automatically change the version
	// of the above components during upgrades.
	ImageTag string `json:"imageTag,omitempty"`

	//TODO: evaluate if we need also a ImageName based on user feedbacks
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterStatus contains the cluster status. The ClusterStatus will be stored in the `kubeadm-config` ConfigMap in the
// cluster, and then updated by kubeadm when additional control plane nodes joins or leaves the cluster.
type ClusterStatus struct {
	metav1.TypeMeta `json:",inline"`

	// `apiEndpoints` contains a list of API endpoints currently available in the cluster,
	// one for each control-plane or API server instance. The key of the map is the IP
	// of the node's default interface.
	APIEndpoints map[string]APIEndpoint `json:"apiEndpoints"`
}

// APIEndpoint struct contains elements of API server instance deployed on a node.
type APIEndpoint struct {
	// `advertiseAddress` sets the IP address for the API server to advertise.
	AdvertiseAddress string `json:"advertiseAddress"`

	// `bindPort` sets the secure port for the API Server to bind to. Defaults to 6443.
	BindPort int32 `json:"bindPort"`
}

// NodeRegistrationOptions holds fields that relate to registering a new control-plane or node to the cluster, either via "kubeadm init" or "kubeadm join"
type NodeRegistrationOptions struct {
	// `name` is the `.metadata.name` field of the Node API object that will be created in this `kubeadm init` or
	// `kubeadm join` operation. This field is also used in the CommonName field of the kubelet's
	// client certificate to the API server. Defaults to the hostname of the node.
	Name string `json:"name,omitempty"`

	// `criSocket` is used to retrieve container runtime information. This information will be
	// annotated to the Node API object, for later re-use.
	CRISocket string `json:"criSocket,omitempty"`

	// `taints` specifies the taints the Node API object should be registered with. If this field is not set, i.e. nil,
	// it will be defaulted to `['node-role.kubernetes.io/master=""']` during `kubeadm init`.
	// If you don't want to taint your control-plane node, set this field to an empty list (`[]`).
	// This field is only used for node registration.
	Taints []v1.Taint `json:"taints,omitempty"`

	// `kubeletExtraArgs` contains extra arguments to pass to the kubelet. Kubeadm writes these arguments into an
	// environment file for the kubelet to source.
	// This overrides the generic base-level configuration in the `kubelet-config-1.x` ConfigMap
	// Command line flags have higher priority when parsing.
	// These values are local and specific to the node kubeadm is executing on.
	KubeletExtraArgs map[string]string `json:"kubeletExtraArgs,omitempty"`
}

// Networking contains elements describing cluster's networking configuration
type Networking struct {
	// `serviceSubnet` is the subnet used by Services. Defaults to "10.96.0.0/12".
	ServiceSubnet string `json:"serviceSubnet"`

	// `podSubnet` is the subnet used by Pods.
	PodSubnet string `json:"podSubnet"`

	// `dnsDomain` is the DNS domain used by Services. Defaults to "cluster.local".
	DNSDomain string `json:"dnsDomain"`
}

// BootstrapToken describes one bootstrap token, stored as a Secret in the cluster
type BootstrapToken struct {
	// `token` is used for establishing bidirectional trust between nodes and control-planes.
	// Used for joining nodes in the cluster.
	Token *BootstrapTokenString `json:"token"`

	// `description` contains a human-friendly message why this token exists and what it's used
	// for, so other administrators can know its purpose.
	Description string `json:"description,omitempty"`

	// `ttl`  defines the time to live (TTL) for this token. Defaults to `24h`.
	// The `expires` field and the `ttl` field are mutually exclusive.
	TTL *metav1.Duration `json:"ttl,omitempty"`

	// `expires` specifies the timestamp when this token expires. Defaults to being set
	// dynamically at runtime based on the `ttl`. The `expires` field and the `ttl` field are
	// mutually exclusive.
	Expires *metav1.Time `json:"expires,omitempty"`

	// `usages` describes the ways in which this token can be used. Can by default be used
	// for establishing bidirectional trust, but that can be changed here.
	Usages []string `json:"usages,omitempty"`

	// `groups` specifies the extra groups that this token will authenticate as when/if
	// used for authentication.
	Groups []string `json:"groups,omitempty"`
}

// Etcd contains elements describing Etcd configuration.
type Etcd struct {
	// `local` provides configurations for the local etcd instance.
	// The `local` field and the `external` field are mutually exclusive.
	Local *LocalEtcd `json:"local,omitempty"`

	// `external` describes how to connect to an external etcd service.
	// The `local` field and the `external` field are mutually exclusive.
	External *ExternalEtcd `json:"external,omitempty"`
}

// LocalEtcd describes that kubeadm should run an etcd cluster locally
type LocalEtcd struct {
	ImageMeta `json:",inline"`

	// `dataDir` is the directory for etcd to place its data. Defaults to "/var/lib/etcd".
	DataDir string `json:"dataDir"`

	// `extraArgs` are extra arguments provided to the etcd binary when run inside a static pod.
	ExtraArgs map[string]string `json:"extraArgs,omitempty"`

	// `serverCertSANs` sets extra Subject Alternative Names (SANs) for the etcd server signing cert.
	ServerCertSANs []string `json:"serverCertSANs,omitempty"`

	// `peerCertSANs` sets extra Subject Alternative Names (SANs) for the etcd peer signing cert.
	PeerCertSANs []string `json:"peerCertSANs,omitempty"`
}

// ExternalEtcd describes an external etcd cluster.
// Kubeadm has no knowledge of where certificate files live and they must be supplied.
type ExternalEtcd struct {
	// `endpoints` contains a list of etcd members. This field is required.
	Endpoints []string `json:"endpoints"`

	// `caFile` is an SSL Certificate Authority (CA) file used to secure etcd communication.
	// Required if using a TLS connection.
	CAFile string `json:"caFile"`

	// `certFile` is an SSL certification file used to secure etcd communication.
	// Required if using a TLS connection.
	CertFile string `json:"certFile"`

	// `keyFile` is an SSL key file used to secure etcd communication.
	// Required if using a TLS connection.
	KeyFile string `json:"keyFile"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DEPRECATED - This group version of JoinConfiguration is deprecated by apis/kubeadm/v1beta2.JoinConfiguration.
// JoinConfiguration contains elements describing a particular node.
type JoinConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// `nodeRegistration` holds fields related to registering a new control-plane node to the cluster
	NodeRegistration NodeRegistrationOptions `json:"nodeRegistration"`

	// `caCertPath` is the path to the SSL certificate authority (CA) used to
	// secure comunications between the node and the control-plane.
	// Defaults to "/etc/kubernetes/pki/ca.crt".
	CACertPath string `json:"caCertPath"`

	// `discovery` specifies the options for the kubelet to use during the TLS Bootstrap process.
	Discovery Discovery `json:"discovery"`

	// `controlPlane` defines the additional control plane instance to be deployed on the joining node.
	// If not specified, no additional control plane instance will be deployed.
	ControlPlane *JoinControlPlane `json:"controlPlane,omitempty"`
}

// JoinControlPlane contains elements describing an additional control plane instance to be deployed on the joining node.
type JoinControlPlane struct {
	// `localAPIEndpoint` represents the endpoint of the API server instance to be deployed on this node.
	LocalAPIEndpoint APIEndpoint `json:"localAPIEndpoint,omitempty"`
}

// Discovery specifies the options for the kubelet to use during the TLS Bootstrap process
type Discovery struct {
	// `bootstrapToken` is used to set the options for bootstrap token based discovery.
	// The `bootstrapToken` field and the `file` field are mutually exclusive.
	BootstrapToken *BootstrapTokenDiscovery `json:"bootstrapToken,omitempty"`

	// `file` is used to specify a file or URL to a kubeconfig file from which to load cluster information.
	// The `bootstrapToken` field and the `file` field are mutually exclusive.
	File *FileDiscovery `json:"file,omitempty"`

	// `tlsBootstrapToken` is a token used for TLS bootstrapping.
	// If `bootstrapToken` is set, this field is defaulted to `.bootstrapToken.token`, but can be overridden.
	// If `file` is set, this field **must be set** in case the KubeConfigFile does not contain any other
	// authentication information.
	TLSBootstrapToken string `json:"tlsBootstrapToken"`

	// `timeout` is used to customize timeout period for the discovery.
	Timeout *metav1.Duration `json:"timeout,omitempty"`
}

// BootstrapTokenDiscovery is used to set the options for bootstrap token based discovery.
type BootstrapTokenDiscovery struct {
	// `token` is a token used to validate cluster information fetched from the control-plane.
	Token string `json:"token"`

	// `apiServerEndpoint` is an IP or domain name for the API server from which info will be fetched.
	APIServerEndpoint string `json:"apiServerEndpoint,omitempty"`

	// `caCertHashes` specifies a set of public key pins to verify when token-based discovery is used.
	// The root CA found during discovery must match one of these values. Specifying an empty set disables
	// root CA pinning, which can be unsafe. Each hash is specified as `<type>:<value>`, where the only
	// type currently supported is "sha256". This is a hex-encoded SHA-256 hash of the Subject Public Key
	// Info (SPKI) object in DER-encoded ASN.1. These hashes can be calculated using, for example, OpenSSL.
	CACertHashes []string `json:"caCertHashes,omitempty"`

	// `unsafeSkipCAVerification` allows token-based discovery without CA verification via `caCertHashes`.
	// This can weaken the kubeadm security since other nodes can impersonate the control-plane.
	UnsafeSkipCAVerification bool `json:"unsafeSkipCAVerification"`
}

// FileDiscovery is used to specify a file or a URL to a kubeconfig file from which to load cluster information.
type FileDiscovery struct {
	// `kubeConfigPath` is used to specify the file path or a URL to the kubeconfig file from which
	// to load cluster information
	KubeConfigPath string `json:"kubeConfigPath"`
}

// HostPathMount contains elements describing volumes that are mounted from the host.
type HostPathMount struct {
	// `name` is the name of the volume inside the Pod template.
	Name string `json:"name"`

	// `hostPath` is the path on the host that will be mounted inside the Pod.
	HostPath string `json:"hostPath"`

	// `mountPath` is the path inside the Pod where `hostPath` will be mounted.
	MountPath string `json:"mountPath"`

	// `readOnly` indicates whether the volume is mounted in read-only mode.
	ReadOnly bool `json:"readOnly,omitempty"`

	// `pathType` is the type of the HostPath, for example, "DirectoryOrCreate", "File", etc.
	PathType v1.HostPathType `json:"pathType,omitempty"`
}
