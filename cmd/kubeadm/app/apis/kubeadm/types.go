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

package kubeadm

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InitConfiguration contains a list of fields that are specifically "kubeadm init"-only runtime
// information. The cluster-wide config is stored in ClusterConfiguration. The InitConfiguration
// object IS NOT uploaded to the kubeadm-config ConfigMap in the cluster, only the
// ClusterConfiguration is.
type InitConfiguration struct {
	metav1.TypeMeta

	// ClusterConfiguration holds the cluster-wide information, and embeds that struct (which can be (un)marshalled separately as well)
	// When InitConfiguration is marshalled to bytes in the external version, this information IS NOT preserved (which can be seen from
	// the `json:"-"` tag in the external variant of these API types.
	ClusterConfiguration `json:"-"`

	// BootstrapTokens is respected at "kubeadm init" time and describes a set of Bootstrap Tokens to create.
	BootstrapTokens []bootstraptokenv1.BootstrapToken

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	DryRun bool

	// NodeRegistration holds fields that relate to registering the new control-plane node to the cluster
	NodeRegistration NodeRegistrationOptions

	// LocalAPIEndpoint represents the endpoint of the API server instance that's deployed on this control plane node
	// In HA setups, this differs from ClusterConfiguration.ControlPlaneEndpoint in the sense that ControlPlaneEndpoint
	// is the global endpoint for the cluster, which then loadbalances the requests to each individual API server. This
	// configuration object lets you customize what IP/DNS name and port the local API server advertises it's accessible
	// on. By default, kubeadm tries to auto-detect the IP of the default interface and use that, but in case that process
	// fails you may set the desired value here.
	LocalAPIEndpoint APIEndpoint

	// CertificateKey sets the key with which certificates and keys are encrypted prior to being uploaded in
	// a secret in the cluster during the uploadcerts init phase.
	// The certificate key is a hex encoded string that is an AES key of size 32 bytes.
	CertificateKey string

	// SkipPhases is a list of phases to skip during command execution.
	// The list of phases can be obtained with the "kubeadm init --help" command.
	// The flag "--skip-phases" takes precedence over this field.
	SkipPhases []string

	// Patches contains options related to applying patches to components deployed by kubeadm during
	// "kubeadm init".
	Patches *Patches

	// Timeouts holds various timeouts that apply to kubeadm commands.
	Timeouts *Timeouts
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterConfiguration contains cluster-wide configuration for a kubeadm cluster
type ClusterConfiguration struct {
	metav1.TypeMeta

	// ComponentConfigs holds component configs known to kubeadm, should long-term only exist in the internal kubeadm API
	// +k8s:conversion-gen=false
	ComponentConfigs ComponentConfigMap

	// Etcd holds configuration for etcd.
	Etcd Etcd

	// Networking holds configuration for the networking topology of the cluster.
	Networking Networking

	// KubernetesVersion is the target version of the control plane.
	KubernetesVersion string

	// CIKubernetesVersion is the target CI version of the control plane.
	// Useful for running kubeadm with CI Kubernetes version.
	// +k8s:conversion-gen=false
	CIKubernetesVersion string

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
	ControlPlaneEndpoint string

	// APIServer contains extra settings for the API server control plane component
	APIServer APIServer

	// ControllerManager contains extra settings for the controller manager control plane component
	ControllerManager ControlPlaneComponent

	// Scheduler contains extra settings for the scheduler control plane component
	Scheduler ControlPlaneComponent

	// DNS defines the options for the DNS add-on installed in the cluster.
	DNS DNS

	// Proxy defines the options for the proxy add-on installed in the cluster.
	Proxy Proxy

	// CertificatesDir specifies where to store or look for all required certificates.
	CertificatesDir string

	// ImageRepository sets the container registry to pull images from.
	// If empty, `registry.k8s.io` will be used by default; in case of kubernetes version is a CI build (kubernetes version starts with `ci/`)
	// `gcr.io/k8s-staging-ci-images` will be used as a default for control plane components and for kube-proxy, while `registry.k8s.io`
	// will be used for all the other images.
	ImageRepository string

	// CIImageRepository is the container registry for core images generated by CI.
	// Useful for running kubeadm with images from CI builds.
	// +k8s:conversion-gen=false
	CIImageRepository string

	// FeatureGates enabled by the user.
	FeatureGates map[string]bool

	// The cluster name
	ClusterName string

	// EncryptionAlgorithm holds the type of asymmetric encryption algorithm used for keys and certificates.
	// Can be one of "RSA-2048" (default), "RSA-3072", "RSA-4096" or "ECDSA-P256".
	EncryptionAlgorithm EncryptionAlgorithmType

	// CertificateValidityPeriod specifies the validity period for a non-CA certificate generated by kubeadm.
	// Default value: 8760h (365 days * 24 hours = 1 year)
	CertificateValidityPeriod *metav1.Duration

	// CACertificateValidityPeriod specifies the validity period for a CA certificate generated by kubeadm.
	// Default value: 87600h (365 days * 24 hours * 10 = 10 years)
	CACertificateValidityPeriod *metav1.Duration
}

// ControlPlaneComponent holds settings common to control plane component of the cluster
type ControlPlaneComponent struct {
	// ExtraArgs is an extra set of flags to pass to the control plane component.
	// An argument name in this list is the flag name as it appears on the
	// command line except without leading dash(es). Extra arguments will override existing
	// default arguments. Duplicate extra arguments are allowed.
	ExtraArgs []Arg

	// ExtraVolumes is an extra set of host volumes, mounted to the control plane component.
	ExtraVolumes []HostPathMount

	// ExtraEnvs is an extra set of environment variables to pass to the control plane component.
	// Environment variables passed using ExtraEnvs will override any existing environment variables, or *_proxy environment variables that kubeadm adds by default.
	// +optional
	ExtraEnvs []EnvVar
}

// APIServer holds settings necessary for API server deployments in the cluster
type APIServer struct {
	ControlPlaneComponent

	// CertSANs sets extra Subject Alternative Names for the API Server signing cert.
	CertSANs []string

	// TimeoutForControlPlane controls the timeout that we use for API server to appear
	TimeoutForControlPlane *metav1.Duration
}

// DNS defines the DNS addon that should be used in the cluster
type DNS struct {
	// ImageMeta allows to customize the image used for the DNS addon
	ImageMeta `json:",inline"`

	// Disabled specifies whether to disable this addon in the cluster
	Disabled bool
}

// Proxy defines the proxy addon that should be used in the cluster
type Proxy struct {
	// Disabled specifies whether to disable this addon in the cluster
	Disabled bool
}

// ImageMeta allows to customize the image used for components that are not
// originated from the Kubernetes/Kubernetes release process
type ImageMeta struct {
	// ImageRepository sets the container registry to pull images from.
	// if not set, the ImageRepository defined in ClusterConfiguration will be used instead.
	ImageRepository string

	// ImageTag allows to specify a tag for the image.
	// In case this value is set, kubeadm does not change automatically the version of the above components during upgrades.
	ImageTag string

	//TODO: evaluate if we need also a ImageName based on user feedbacks
}

// APIEndpoint struct contains elements of API server instance deployed on a node.
type APIEndpoint struct {
	// AdvertiseAddress sets the IP address for the API server to advertise.
	AdvertiseAddress string

	// BindPort sets the secure port for the API Server to bind to.
	// Defaults to 6443.
	BindPort int32
}

// NodeRegistrationOptions holds fields that relate to registering a new control-plane or node to the cluster, either via "kubeadm init" or "kubeadm join"
type NodeRegistrationOptions struct {

	// Name is the `.Metadata.Name` field of the Node API object that will be created in this "kubeadm init" or "kubeadm join" operation.
	// This field is also used in the CommonName field of the kubelet's client certificate to the API server.
	// Defaults to the hostname of the node if not provided.
	Name string

	// CRISocket is used to retrieve container runtime info. This information will be annotated to the Node API object, for later re-use
	CRISocket string

	// Taints specifies the taints the Node API object should be registered with. If this field is unset, i.e. nil,
	// it will be defaulted with a control-plane taint for control-plane nodes. If you don't want to taint your control-plane
	// node, set this field to an empty slice, i.e. `taints: []` in the YAML file. This field is solely used for Node registration.
	Taints []v1.Taint

	// KubeletExtraArgs passes through extra arguments to the kubelet. The arguments here are passed to the kubelet command line via the environment file
	// kubeadm writes at runtime for the kubelet to source. This overrides the generic base-level configuration in the kubelet-config ConfigMap
	// Flags have higher priority when parsing. These values are local and specific to the node kubeadm is executing on.
	// An argument name in this list is the flag name as it appears on the command line except without leading dash(es).
	// Extra arguments will override existing default arguments. Duplicate extra arguments are allowed.
	KubeletExtraArgs []Arg

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored when the current node is registered, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	IgnorePreflightErrors []string

	// ImagePullPolicy specifies the policy for image pulling during kubeadm "init" and "join" operations.
	// The value of this field must be one of "Always", "IfNotPresent" or "Never".
	// If this field is unset kubeadm will default it to "IfNotPresent", or pull the required images if not present on the host.
	ImagePullPolicy v1.PullPolicy `json:"imagePullPolicy,omitempty"`

	// ImagePullSerial specifies if image pulling performed by kubeadm must be done serially or in parallel.
	ImagePullSerial *bool
}

// Networking contains elements describing cluster's networking configuration.
type Networking struct {
	// ServiceSubnet is the subnet used by k8s services. Defaults to "10.96.0.0/12".
	ServiceSubnet string
	// PodSubnet is the subnet used by pods.
	PodSubnet string
	// DNSDomain is the dns domain used by k8s services. Defaults to "cluster.local".
	DNSDomain string
}

// Etcd contains elements describing Etcd configuration.
type Etcd struct {

	// Local provides configuration knobs for configuring the local etcd instance
	// Local and External are mutually exclusive
	Local *LocalEtcd

	// External describes how to connect to an external etcd cluster
	// Local and External are mutually exclusive
	External *ExternalEtcd
}

// LocalEtcd describes that kubeadm should run an etcd cluster locally
type LocalEtcd struct {
	// ImageMeta allows to customize the container used for etcd
	ImageMeta `json:",inline"`

	// DataDir is the directory etcd will place its data.
	// Defaults to "/var/lib/etcd".
	DataDir string

	// ExtraArgs are extra arguments provided to the etcd binary
	// when run inside a static pod.
	// An argument name in this list is the flag name as it appears on the
	// command line except without leading dash(es). Extra arguments will override existing
	// default arguments. Duplicate extra arguments are allowed.
	ExtraArgs []Arg

	// ExtraEnvs is an extra set of environment variables to pass to the control plane component.
	// Environment variables passed using ExtraEnvs will override any existing environment variables, or *_proxy environment variables that kubeadm adds by default.
	// +optional
	ExtraEnvs []EnvVar

	// ServerCertSANs sets extra Subject Alternative Names for the etcd server signing cert.
	ServerCertSANs []string
	// PeerCertSANs sets extra Subject Alternative Names for the etcd peer signing cert.
	PeerCertSANs []string
}

// ExternalEtcd describes an external etcd cluster
type ExternalEtcd struct {

	// Endpoints of etcd members. Useful for using external etcd.
	// If not provided, kubeadm will run etcd in a static pod.
	Endpoints []string
	// CAFile is an SSL Certificate Authority file used to secure etcd communication.
	CAFile string
	// CertFile is an SSL certification file used to secure etcd communication.
	CertFile string
	// KeyFile is an SSL key file used to secure etcd communication.
	KeyFile string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// JoinConfiguration contains elements describing a particular node.
type JoinConfiguration struct {
	metav1.TypeMeta

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	DryRun bool

	// NodeRegistration holds fields that relate to registering the new control-plane node to the cluster
	NodeRegistration NodeRegistrationOptions

	// CACertPath is the path to the SSL certificate authority used to
	// secure comunications between node and control-plane.
	// Defaults to "/etc/kubernetes/pki/ca.crt".
	CACertPath string

	// Discovery specifies the options for the kubelet to use during the TLS Bootstrap process
	Discovery Discovery

	// ControlPlane defines the additional control plane instance to be deployed on the joining node.
	// If nil, no additional control plane instance will be deployed.
	ControlPlane *JoinControlPlane

	// SkipPhases is a list of phases to skip during command execution.
	// The list of phases can be obtained with the "kubeadm join --help" command.
	// The flag "--skip-phases" takes precedence over this field.
	SkipPhases []string

	// Patches contains options related to applying patches to components deployed by kubeadm during
	// "kubeadm join".
	Patches *Patches

	// Timeouts holds various timeouts that apply to kubeadm commands.
	Timeouts *Timeouts
}

// JoinControlPlane contains elements describing an additional control plane instance to be deployed on the joining node.
type JoinControlPlane struct {
	// LocalAPIEndpoint represents the endpoint of the API server instance to be deployed on this node.
	LocalAPIEndpoint APIEndpoint

	// CertificateKey is the key that is used for decryption of certificates after they are downloaded from the secret
	// upon joining a new control plane node. The corresponding encryption key is in the InitConfiguration.
	// The certificate key is a hex encoded string that is an AES key of size 32 bytes.
	CertificateKey string
}

// Discovery specifies the options for the kubelet to use during the TLS Bootstrap process
type Discovery struct {
	// BootstrapToken is used to set the options for bootstrap token based discovery
	// BootstrapToken and File are mutually exclusive
	BootstrapToken *BootstrapTokenDiscovery

	// File is used to specify a file or URL to a kubeconfig file from which to load cluster information
	// BootstrapToken and File are mutually exclusive
	File *FileDiscovery

	// TLSBootstrapToken is a token used for TLS bootstrapping.
	// If .BootstrapToken is set, this field is defaulted to .BootstrapToken.Token, but can be overridden.
	// If .File is set, this field **must be set** in case the KubeConfigFile does not contain any other authentication information
	TLSBootstrapToken string

	// Timeout modifies the discovery timeout
	Timeout *metav1.Duration
}

// BootstrapTokenDiscovery is used to set the options for bootstrap token based discovery
type BootstrapTokenDiscovery struct {
	// Token is a token used to validate cluster information
	// fetched from the control-plane.
	Token string

	// APIServerEndpoint is an IP or domain name to the API server from which info will be fetched.
	APIServerEndpoint string

	// CACertHashes specifies a set of public key pins to verify
	// when token-based discovery is used. The root CA found during discovery
	// must match one of these values. Specifying an empty set disables root CA
	// pinning, which can be unsafe. Each hash is specified as "<type>:<value>",
	// where the only currently supported type is "sha256". This is a hex-encoded
	// SHA-256 hash of the Subject Public Key Info (SPKI) object in DER-encoded
	// ASN.1. These hashes can be calculated using, for example, OpenSSL.
	CACertHashes []string

	// UnsafeSkipCAVerification allows token-based discovery
	// without CA verification via CACertHashes. This can weaken
	// the security of kubeadm since other nodes can impersonate the control-plane.
	UnsafeSkipCAVerification bool
}

// FileDiscovery is used to specify a file or URL to a kubeconfig file from which to load cluster information
type FileDiscovery struct {
	// KubeConfigPath is used to specify the actual file path or URL to the kubeconfig file from which to load cluster information
	KubeConfigPath string
}

// GetControlPlaneImageRepository returns name of image repository
// for control plane images (API,Controller Manager,Scheduler and Proxy)
// It will override location with CI registry name in case user requests special
// Kubernetes version from CI build area.
// (See: kubeadmconstants.DefaultCIImageRepository)
func (cfg *ClusterConfiguration) GetControlPlaneImageRepository() string {
	if cfg.CIImageRepository != "" {
		return cfg.CIImageRepository
	}
	return cfg.ImageRepository
}

// EncryptionAlgorithmType returns the type of encryption keys used in the cluster.
func (cfg *ClusterConfiguration) EncryptionAlgorithmType() EncryptionAlgorithmType {
	// If the feature gate is set to true, or false respect it.
	// If the feature gate is not set, use the EncryptionAlgorithm field (v1beta4).
	// TODO: remove this function when the feature gate is removed.
	if enabled, ok := cfg.FeatureGates[features.PublicKeysECDSA]; ok {
		if enabled {
			return EncryptionAlgorithmECDSAP256
		}
		return EncryptionAlgorithmRSA2048
	}
	return cfg.EncryptionAlgorithm
}

// HostPathMount contains elements describing volumes that are mounted from the
// host.
type HostPathMount struct {
	// Name of the volume inside the pod template.
	Name string
	// HostPath is the path in the host that will be mounted inside
	// the pod.
	HostPath string
	// MountPath is the path inside the pod where hostPath will be mounted.
	MountPath string
	// ReadOnly controls write access to the volume
	ReadOnly bool
	// PathType is the type of the HostPath.
	PathType v1.HostPathType
}

// Patches contains options related to applying patches to components deployed by kubeadm.
type Patches struct {
	// Directory is a path to a directory that contains files named "target[suffix][+patchtype].extension".
	// For example, "kube-apiserver0+merge.yaml" or just "etcd.json". "target" can be one of
	// "kube-apiserver", "kube-controller-manager", "kube-scheduler", "etcd", "kubeletconfiguration", "corednsdeployment".
	// "patchtype" can be one of "strategic" "merge" or "json" and they match the patch formats supported by kubectl.
	// The default "patchtype" is "strategic". "extension" must be either "json" or "yaml".
	// "suffix" is an optional string that can be used to determine which patches are applied
	// first alpha-numerically.
	Directory string
}

// DocumentMap is a convenient way to describe a map between a YAML document and its GVK type
// +k8s:deepcopy-gen=false
type DocumentMap map[schema.GroupVersionKind][]byte

// ComponentConfig holds a known component config
type ComponentConfig interface {
	// DeepCopy should create a new deep copy of the component config in place
	DeepCopy() ComponentConfig

	// Marshal is marshalling the config into a YAML document returned as a byte slice
	Marshal() ([]byte, error)

	// Unmarshal loads the config from a document map. No config in the document map is no error.
	Unmarshal(docmap DocumentMap) error

	// Default patches the component config with kubeadm preferred defaults
	Default(cfg *ClusterConfiguration, localAPIEndpoint *APIEndpoint, nodeRegOpts *NodeRegistrationOptions)

	// IsUserSupplied indicates if the component config was supplied or modified by a user or was kubeadm generated
	IsUserSupplied() bool

	// SetUserSupplied sets the state of the component config "user supplied" flag to, either true, or false.
	SetUserSupplied(userSupplied bool)

	// Mutate allows applying pre-defined modifications to the config before it's marshaled.
	Mutate() error

	// Set can be used to set the internal configuration in the ComponentConfig
	Set(interface{})

	// Get can be used to get the internal configuration in the ComponentConfig
	Get() interface{}
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ResetConfiguration contains a list of fields that are specifically "kubeadm reset"-only runtime information.
type ResetConfiguration struct {
	metav1.TypeMeta

	// CertificatesDir specifies the directory where the certificates are stored. If specified, it will be cleaned during the reset process.
	CertificatesDir string

	// CleanupTmpDir specifies whether the "/etc/kubernetes/tmp" directory should be cleaned during the reset process.
	CleanupTmpDir bool

	// CRISocket is used to retrieve container runtime info and used for the removal of the containers.
	// If CRISocket is not specified by flag or config file, kubeadm will try to detect one valid CRISocket instead.
	CRISocket string

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	DryRun bool

	// Force flag instructs kubeadm to reset the node without prompting for confirmation.
	Force bool

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored during the reset process, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	IgnorePreflightErrors []string

	// SkipPhases is a list of phases to skip during command execution.
	// The list of phases can be obtained with the "kubeadm reset phase --help" command.
	SkipPhases []string

	// UnmountFlags is a list of unmount2() syscall flags that kubeadm can use when unmounting
	// directories during "reset". A flag can be one of: MNT_FORCE, MNT_DETACH, MNT_EXPIRE, UMOUNT_NOFOLLOW.
	// By default this list is empty.
	UnmountFlags []string

	// Timeouts holds various timeouts that apply to kubeadm commands.
	Timeouts *Timeouts
}

// UpgradeApplyConfiguration contains a list of configurable options which are specific to the "kubeadm upgrade apply" command.
type UpgradeApplyConfiguration struct {
	// KubernetesVersion is the target version of the control plane.
	KubernetesVersion string

	// AllowExperimentalUpgrades instructs kubeadm to show unstable versions of Kubernetes as an upgrade
	// alternative and allows upgrading to an alpha/beta/release candidate version of Kubernetes.
	// Default: false
	AllowExperimentalUpgrades *bool

	// Enable AllowRCUpgrades will show release candidate versions of Kubernetes as an upgrade alternative and
	// allows upgrading to a release candidate version of Kubernetes.
	AllowRCUpgrades *bool

	// CertificateRenewal instructs kubeadm to execute certificate renewal during upgrades.
	CertificateRenewal *bool

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	DryRun *bool

	// EtcdUpgrade instructs kubeadm to execute etcd upgrade during upgrades.
	EtcdUpgrade *bool

	// ForceUpgrade flag instructs kubeadm to upgrade the cluster without prompting for confirmation.
	ForceUpgrade *bool

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored during the upgrade process, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	IgnorePreflightErrors []string

	// Patches contains options related to applying patches to components deployed by kubeadm during "kubeadm upgrade".
	Patches *Patches

	// PrintConfig specifies whether the configuration file that will be used in the upgrade should be printed or not.
	PrintConfig *bool

	// SkipPhases is a list of phases to skip during command execution.
	// NOTE: This field is currently ignored for "kubeadm upgrade apply", but in the future it will be supported.
	SkipPhases []string

	// ImagePullPolicy specifies the policy for image pulling during kubeadm "upgrade apply" operations.
	// The value of this field must be one of "Always", "IfNotPresent" or "Never".
	// If this field is unset kubeadm will default it to "IfNotPresent", or pull the required images if not present on the host.
	ImagePullPolicy v1.PullPolicy `json:"imagePullPolicy,omitempty"`

	// ImagePullSerial specifies if image pulling performed by kubeadm must be done serially or in parallel.
	ImagePullSerial *bool
}

// UpgradeDiffConfiguration contains a list of configurable options which are specific to the "kubeadm upgrade diff" command.
type UpgradeDiffConfiguration struct {
	// KubernetesVersion is the target version of the control plane.
	KubernetesVersion string

	// DiffContextLines is the number of lines of context in the diff.
	DiffContextLines int
}

// UpgradeNodeConfiguration contains a list of configurable options which are specific to the "kubeadm upgrade node" command.
type UpgradeNodeConfiguration struct {
	// CertificateRenewal instructs kubeadm to execute certificate renewal during upgrades.
	CertificateRenewal *bool

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	DryRun *bool

	// EtcdUpgrade instructs kubeadm to execute etcd upgrade during upgrades.
	EtcdUpgrade *bool

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored during the upgrade process, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	IgnorePreflightErrors []string

	// SkipPhases is a list of phases to skip during command execution.
	// The list of phases can be obtained with the "kubeadm upgrade node phase --help" command.
	SkipPhases []string

	// Patches contains options related to applying patches to components deployed by kubeadm during "kubeadm upgrade".
	Patches *Patches

	// ImagePullPolicy specifies the policy for image pulling during kubeadm "upgrade node" operations.
	// The value of this field must be one of "Always", "IfNotPresent" or "Never".
	// If this field is unset kubeadm will default it to "IfNotPresent", or pull the required images if not present on the host.
	ImagePullPolicy v1.PullPolicy `json:"imagePullPolicy,omitempty"`

	// ImagePullSerial specifies if image pulling performed by kubeadm must be done serially or in parallel.
	ImagePullSerial *bool
}

// UpgradePlanConfiguration contains a list of configurable options which are specific to the "kubeadm upgrade plan" command.
type UpgradePlanConfiguration struct {
	// KubernetesVersion is the target version of the control plane.
	// +optional
	KubernetesVersion string

	// AllowExperimentalUpgrades instructs kubeadm to show unstable versions of Kubernetes as an upgrade
	// alternative and allows upgrading to an alpha/beta/release candidate version of Kubernetes.
	// Default: false
	// +optional
	AllowExperimentalUpgrades *bool

	// Enable AllowRCUpgrades will show release candidate versions of Kubernetes as an upgrade alternative and
	// allows upgrading to a release candidate version of Kubernetes.
	AllowRCUpgrades *bool

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	DryRun *bool

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored during the upgrade process, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	IgnorePreflightErrors []string

	// PrintConfig specifies whether the configuration file that will be used in the upgrade should be printed or not.
	PrintConfig *bool
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UpgradeConfiguration contains a list of options that are specific to "kubeadm upgrade" subcommands.
type UpgradeConfiguration struct {
	metav1.TypeMeta

	// Apply holds a list of options that are specific to the "kubeadm upgrade apply" command.
	Apply UpgradeApplyConfiguration

	// Diff holds a list of options that are specific to the "kubeadm upgrade diff" command.
	Diff UpgradeDiffConfiguration

	// Node holds a list of options that are specific to the "kubeadm upgrade node" command.
	Node UpgradeNodeConfiguration

	// Plan holds a list of options that are specific to the "kubeadm upgrade plan" command.
	Plan UpgradePlanConfiguration

	// Timeouts holds various timeouts that apply to kubeadm commands.
	Timeouts *Timeouts
}

const (
	// UnmountFlagMNTForce represents the flag "MNT_FORCE"
	UnmountFlagMNTForce = "MNT_FORCE"
	// UnmountFlagMNTDetach represents the flag "MNT_DETACH"
	UnmountFlagMNTDetach = "MNT_DETACH"
	// UnmountFlagMNTExpire represents the flag "MNT_EXPIRE"
	UnmountFlagMNTExpire = "MNT_EXPIRE"
	// UnmountFlagUmountNoFollow represents the flag "UMOUNT_NOFOLLOW"
	UnmountFlagUmountNoFollow = "UMOUNT_NOFOLLOW"
)

// ComponentConfigMap is a map between a group name (as in GVK group) and a ComponentConfig
type ComponentConfigMap map[string]ComponentConfig

// Arg represents an argument with a name and a value.
type Arg struct {
	Name  string
	Value string
}

// EnvVar represents an environment variable present in a Container.
type EnvVar struct {
	v1.EnvVar
}

// EncryptionAlgorithmType can define an asymmetric encryption algorithm type.
type EncryptionAlgorithmType string

const (
	// EncryptionAlgorithmECDSAP256 defines the ECDSA encryption algorithm type with curve P256.
	EncryptionAlgorithmECDSAP256 EncryptionAlgorithmType = "ECDSA-P256"
	// EncryptionAlgorithmRSA2048 defines the RSA encryption algorithm type with key size 2048 bits.
	EncryptionAlgorithmRSA2048 EncryptionAlgorithmType = "RSA-2048"
	// EncryptionAlgorithmRSA3072 defines the RSA encryption algorithm type with key size 3072 bits.
	EncryptionAlgorithmRSA3072 EncryptionAlgorithmType = "RSA-3072"
	// EncryptionAlgorithmRSA4096 defines the RSA encryption algorithm type with key size 4096 bits.
	EncryptionAlgorithmRSA4096 EncryptionAlgorithmType = "RSA-4096"
)

// Timeouts holds various timeouts that apply to kubeadm commands.
type Timeouts struct {
	// ControlPlaneComponentHealthCheck is the amount of time to wait for a control plane
	// component, such as the API server, to be healthy during "kubeadm init" and "kubeadm join".
	ControlPlaneComponentHealthCheck *metav1.Duration

	// KubeletHealthCheck is the amount of time to wait for the kubelet to be healthy
	// during "kubeadm init" and "kubeadm join".
	KubeletHealthCheck *metav1.Duration

	// KubernetesAPICall is the amount of time to wait for the kubeadm client to complete a request to
	// the API server. This applies to all types of methods (GET, POST, etc).
	KubernetesAPICall *metav1.Duration

	// EtcdAPICall is the amount of time to wait for the kubeadm etcd client to complete a request to
	// the etcd cluster.
	EtcdAPICall *metav1.Duration

	// TLSBootstrap is the amount of time to wait for the kubelet to complete TLS bootstrap
	// for a joining node.
	TLSBootstrap *metav1.Duration

	// Discovery is the amount of time to wait for kubeadm to validate the API server identity
	// for a joining node.
	Discovery *metav1.Duration

	// UpgradeManifests is the timeout for upgradring static Pod manifests
	UpgradeManifests *metav1.Duration
}
