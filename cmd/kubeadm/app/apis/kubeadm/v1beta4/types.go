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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// InitConfiguration contains a list of elements that is specific "kubeadm init"-only runtime
// information.
type InitConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// `kubeadm init`-only information. These fields are solely used the first time `kubeadm init` runs.
	// After that, the information in the fields IS NOT uploaded to the `kubeadm-config` ConfigMap
	// that is used by `kubeadm upgrade` for instance. These fields must be omitempty.

	// BootstrapTokens is respected at `kubeadm init` time and describes a set of Bootstrap Tokens to create.
	// This information IS NOT uploaded to the kubeadm cluster configmap, partly because of its sensitive nature
	// +optional
	BootstrapTokens []bootstraptokenv1.BootstrapToken `json:"bootstrapTokens,omitempty"`

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	// +optional
	DryRun bool `json:"dryRun,omitempty"`

	// NodeRegistration holds fields that relate to registering the new control-plane node to the cluster
	// +optional
	NodeRegistration NodeRegistrationOptions `json:"nodeRegistration,omitempty"`

	// LocalAPIEndpoint represents the endpoint of the API server instance that's deployed on this control plane node
	// In HA setups, this differs from ClusterConfiguration.ControlPlaneEndpoint in the sense that ControlPlaneEndpoint
	// is the global endpoint for the cluster, which then loadbalances the requests to each individual API server. This
	// configuration object lets you customize what IP/DNS name and port the local API server advertises it's accessible
	// on. By default, kubeadm tries to auto-detect the IP of the default interface and use that, but in case that process
	// fails you may set the desired value here.
	// +optional
	LocalAPIEndpoint APIEndpoint `json:"localAPIEndpoint,omitempty"`

	// CertificateKey sets the key with which certificates and keys are encrypted prior to being uploaded in
	// a secret in the cluster during the uploadcerts init phase.
	// The certificate key is a hex encoded string that is an AES key of size 32 bytes.
	// +optional
	CertificateKey string `json:"certificateKey,omitempty"`

	// SkipPhases is a list of phases to skip during command execution.
	// The list of phases can be obtained with the "kubeadm init --help" command.
	// The flag "--skip-phases" takes precedence over this field.
	// +optional
	SkipPhases []string `json:"skipPhases,omitempty"`

	// Patches contains options related to applying patches to components deployed by kubeadm during
	// "kubeadm init".
	// +optional
	Patches *Patches `json:"patches,omitempty"`

	// Timeouts holds various timeouts that apply to kubeadm commands.
	// +optional
	Timeouts *Timeouts `json:"timeouts,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ClusterConfiguration contains cluster-wide configuration for a kubeadm cluster
type ClusterConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Etcd holds configuration for etcd.
	// +optional
	Etcd Etcd `json:"etcd,omitempty"`

	// Networking holds configuration for the networking topology of the cluster.
	// +optional
	Networking Networking `json:"networking,omitempty"`

	// KubernetesVersion is the target version of the control plane.
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
	// +optional
	ControlPlaneEndpoint string `json:"controlPlaneEndpoint,omitempty"`

	// APIServer contains extra settings for the API server control plane component
	// +optional
	APIServer APIServer `json:"apiServer,omitempty"`

	// ControllerManager contains extra settings for the controller manager control plane component
	// +optional
	ControllerManager ControlPlaneComponent `json:"controllerManager,omitempty"`

	// Scheduler contains extra settings for the scheduler control plane component
	// +optional
	Scheduler ControlPlaneComponent `json:"scheduler,omitempty"`

	// DNS defines the options for the DNS add-on installed in the cluster.
	// +optional
	DNS DNS `json:"dns,omitempty"`

	// Proxy defines the options for the proxy add-on installed in the cluster.
	Proxy Proxy `json:"proxy,omitempty"`

	// CertificatesDir specifies where to store or look for all required certificates.
	// +optional
	CertificatesDir string `json:"certificatesDir,omitempty"`

	// ImageRepository sets the container registry to pull images from.
	// If empty, `registry.k8s.io` will be used by default; in case of kubernetes version is a CI build (kubernetes version starts with `ci/`)
	// `gcr.io/k8s-staging-ci-images` will be used as a default for control plane components and for kube-proxy, while `registry.k8s.io`
	// will be used for all the other images.
	// +optional
	ImageRepository string `json:"imageRepository,omitempty"`

	// FeatureGates enabled by the user.
	// +optional
	FeatureGates map[string]bool `json:"featureGates,omitempty"`

	// The cluster name
	// +optional
	ClusterName string `json:"clusterName,omitempty"`

	// EncryptionAlgorithm holds the type of asymmetric encryption algorithm used for keys and certificates.
	// Can be one of "RSA-2048" (default), "RSA-3072", "RSA-4096" or "ECDSA-P256".
	// +optional
	EncryptionAlgorithm EncryptionAlgorithmType `json:"encryptionAlgorithm,omitempty"`

	// CertificateValidityPeriod specifies the validity period for a non-CA certificate generated by kubeadm.
	// Default value: 8760h (365 days * 24 hours = 1 year)
	// +optional
	CertificateValidityPeriod *metav1.Duration `json:"certificateValidityPeriod,omitempty"`

	// CACertificateValidityPeriod specifies the validity period for a CA certificate generated by kubeadm.
	// Default value: 87600h (365 days * 24 hours * 10 = 10 years)
	// +optional
	CACertificateValidityPeriod *metav1.Duration `json:"caCertificateValidityPeriod,omitempty"`
}

// ControlPlaneComponent holds settings common to control plane component of the cluster
type ControlPlaneComponent struct {
	// ExtraArgs is an extra set of flags to pass to the control plane component.
	// An argument name in this list is the flag name as it appears on the
	// command line except without leading dash(es). Extra arguments will override existing
	// default arguments. Duplicate extra arguments are allowed.
	// +optional
	ExtraArgs []Arg `json:"extraArgs,omitempty"`

	// ExtraVolumes is an extra set of host volumes, mounted to the control plane component.
	// +optional
	ExtraVolumes []HostPathMount `json:"extraVolumes,omitempty"`

	// ExtraEnvs is an extra set of environment variables to pass to the control plane component.
	// Environment variables passed using ExtraEnvs will override any existing environment variables, or *_proxy environment variables that kubeadm adds by default.
	// +optional
	ExtraEnvs []EnvVar `json:"extraEnvs,omitempty"`
}

// APIServer holds settings necessary for API server deployments in the cluster
type APIServer struct {
	ControlPlaneComponent `json:",inline"`

	// CertSANs sets extra Subject Alternative Names for the API Server signing cert.
	// +optional
	CertSANs []string `json:"certSANs,omitempty"`
}

// DNS defines the DNS addon that should be used in the cluster
type DNS struct {
	// ImageMeta allows to customize the image used for the DNS addon
	ImageMeta `json:",inline"`

	// Disabled specifies whether to disable this addon in the cluster
	// +optional
	Disabled bool `json:"disabled,omitempty"`
}

// Proxy defines the proxy addon that should be used in the cluster
type Proxy struct {
	// Disabled specifies whether to disable this addon in the cluster
	// +optional
	Disabled bool `json:"disabled,omitempty"`
}

// ImageMeta allows to customize the image used for components that are not
// originated from the Kubernetes/Kubernetes release process
type ImageMeta struct {
	// ImageRepository sets the container registry to pull images from.
	// if not set, the ImageRepository defined in ClusterConfiguration will be used instead.
	// +optional
	ImageRepository string `json:"imageRepository,omitempty"`

	// ImageTag allows to specify a tag for the image.
	// In case this value is set, kubeadm does not change automatically the version of the above components during upgrades.
	// +optional
	ImageTag string `json:"imageTag,omitempty"`

	//TODO: evaluate if we need also a ImageName based on user feedbacks
}

// APIEndpoint struct contains elements of API server instance deployed on a node.
type APIEndpoint struct {
	// AdvertiseAddress sets the IP address for the API server to advertise.
	// +optional
	AdvertiseAddress string `json:"advertiseAddress,omitempty"`

	// BindPort sets the secure port for the API Server to bind to.
	// Defaults to 6443.
	// +optional
	BindPort int32 `json:"bindPort,omitempty"`
}

// NodeRegistrationOptions holds fields that relate to registering a new control-plane or node to the cluster, either via "kubeadm init" or "kubeadm join"
type NodeRegistrationOptions struct {

	// Name is the `.Metadata.Name` field of the Node API object that will be created in this `kubeadm init` or `kubeadm join` operation.
	// This field is also used in the CommonName field of the kubelet's client certificate to the API server.
	// Defaults to the hostname of the node if not provided.
	// +optional
	Name string `json:"name,omitempty"`

	// CRISocket is used to retrieve container runtime info. This information will be annotated to the Node API object, for later re-use
	// +optional
	CRISocket string `json:"criSocket,omitempty"`

	// Taints specifies the taints the Node API object should be registered with. If this field is unset, i.e. nil,
	// it will be defaulted with a control-plane taint for control-plane nodes. If you don't want to taint your control-plane
	// node, set this field to an empty slice, i.e. `taints: []` in the YAML file. This field is solely used for Node registration.
	Taints []corev1.Taint `json:"taints"`

	// KubeletExtraArgs passes through extra arguments to the kubelet. The arguments here are passed to the kubelet command line via the environment file
	// kubeadm writes at runtime for the kubelet to source. This overrides the generic base-level configuration in the kubelet-config ConfigMap
	// Flags have higher priority when parsing. These values are local and specific to the node kubeadm is executing on.
	// An argument name in this list is the flag name as it appears on the command line except without leading dash(es).
	// Extra arguments will override existing default arguments. Duplicate extra arguments are allowed.
	// +optional
	KubeletExtraArgs []Arg `json:"kubeletExtraArgs,omitempty"`

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored when the current node is registered, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	// +optional
	IgnorePreflightErrors []string `json:"ignorePreflightErrors,omitempty"`

	// ImagePullPolicy specifies the policy for image pulling during kubeadm "init" and "join" operations.
	// The value of this field must be one of "Always", "IfNotPresent" or "Never".
	// If this field is unset kubeadm will default it to "IfNotPresent", or pull the required images if not present on the host.
	// +optional
	ImagePullPolicy corev1.PullPolicy `json:"imagePullPolicy,omitempty"`

	// ImagePullSerial specifies if image pulling performed by kubeadm must be done serially or in parallel.
	// Default: true
	// +optional
	ImagePullSerial *bool `json:"imagePullSerial,omitempty"`
}

// Networking contains elements describing cluster's networking configuration
type Networking struct {
	// ServiceSubnet is the subnet used by k8s services. Defaults to "10.96.0.0/12".
	// +optional
	ServiceSubnet string `json:"serviceSubnet,omitempty"`
	// PodSubnet is the subnet used by pods.
	// +optional
	PodSubnet string `json:"podSubnet,omitempty"`
	// DNSDomain is the dns domain used by k8s services. Defaults to "cluster.local".
	// +optional
	DNSDomain string `json:"dnsDomain,omitempty"`
}

// Etcd contains elements describing Etcd configuration.
type Etcd struct {

	// Local provides configuration knobs for configuring the local etcd instance
	// Local and External are mutually exclusive
	// +optional
	Local *LocalEtcd `json:"local,omitempty"`

	// External describes how to connect to an external etcd cluster
	// Local and External are mutually exclusive
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
	// An argument name in this list is the flag name as it appears on the
	// command line except without leading dash(es). Extra arguments will override existing
	// default arguments. Duplicate extra arguments are allowed.
	// +optional
	ExtraArgs []Arg `json:"extraArgs,omitempty"`

	// ExtraEnvs is an extra set of environment variables to pass to the control plane component.
	// Environment variables passed using ExtraEnvs will override any existing environment variables, or *_proxy environment variables that kubeadm adds by default.
	// +optional
	ExtraEnvs []EnvVar `json:"extraEnvs,omitempty"`

	// ServerCertSANs sets extra Subject Alternative Names for the etcd server signing cert.
	// +optional
	ServerCertSANs []string `json:"serverCertSANs,omitempty"`
	// PeerCertSANs sets extra Subject Alternative Names for the etcd peer signing cert.
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

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	// +optional
	DryRun bool `json:"dryRun,omitempty"`

	// NodeRegistration holds fields that relate to registering the new control-plane node to the cluster
	// +optional
	NodeRegistration NodeRegistrationOptions `json:"nodeRegistration,omitempty"`

	// CACertPath is the path to the SSL certificate authority used to
	// secure comunications between node and control-plane.
	// Defaults to "/etc/kubernetes/pki/ca.crt".
	// +optional
	CACertPath string `json:"caCertPath,omitempty"`

	// Discovery specifies the options for the kubelet to use during the TLS Bootstrap process
	Discovery Discovery `json:"discovery"`

	// ControlPlane defines the additional control plane instance to be deployed on the joining node.
	// If nil, no additional control plane instance will be deployed.
	// +optional
	ControlPlane *JoinControlPlane `json:"controlPlane,omitempty"`

	// SkipPhases is a list of phases to skip during command execution.
	// The list of phases can be obtained with the "kubeadm join --help" command.
	// The flag "--skip-phases" takes precedence over this field.
	// +optional
	SkipPhases []string `json:"skipPhases,omitempty"`

	// Patches contains options related to applying patches to components deployed by kubeadm during
	// "kubeadm join".
	// +optional
	Patches *Patches `json:"patches,omitempty"`

	// Timeouts holds various timeouts that apply to kubeadm commands.
	// +optional
	Timeouts *Timeouts `json:"timeouts,omitempty"`
}

// JoinControlPlane contains elements describing an additional control plane instance to be deployed on the joining node.
type JoinControlPlane struct {
	// LocalAPIEndpoint represents the endpoint of the API server instance to be deployed on this node.
	// +optional
	LocalAPIEndpoint APIEndpoint `json:"localAPIEndpoint,omitempty"`

	// CertificateKey is the key that is used for decryption of certificates after they are downloaded from the secret
	// upon joining a new control plane node. The corresponding encryption key is in the InitConfiguration.
	// The certificate key is a hex encoded string that is an AES key of size 32 bytes.
	// +optional
	CertificateKey string `json:"certificateKey,omitempty"`
}

// Discovery specifies the options for the kubelet to use during the TLS Bootstrap process
type Discovery struct {
	// BootstrapToken is used to set the options for bootstrap token based discovery
	// BootstrapToken and File are mutually exclusive
	// +optional
	BootstrapToken *BootstrapTokenDiscovery `json:"bootstrapToken,omitempty"`

	// File is used to specify a file or URL to a kubeconfig file from which to load cluster information
	// BootstrapToken and File are mutually exclusive
	// +optional
	File *FileDiscovery `json:"file,omitempty"`

	// TLSBootstrapToken is a token used for TLS bootstrapping.
	// If .BootstrapToken is set, this field is defaulted to .BootstrapToken.Token, but can be overridden.
	// If .File is set, this field **must be set** in case the KubeConfigFile does not contain any other authentication information
	// +optional
	TLSBootstrapToken string `json:"tlsBootstrapToken,omitempty" datapolicy:"token"`
}

// BootstrapTokenDiscovery is used to set the options for bootstrap token based discovery
type BootstrapTokenDiscovery struct {
	// Token is a token used to validate cluster information
	// fetched from the control-plane.
	Token string `json:"token" datapolicy:"token"`

	// APIServerEndpoint is an IP or domain name to the API server from which info will be fetched.
	// +optional
	APIServerEndpoint string `json:"apiServerEndpoint,omitempty"`

	// CACertHashes specifies a set of public key pins to verify
	// when token-based discovery is used. The root CA found during discovery
	// must match one of these values. Specifying an empty set disables root CA
	// pinning, which can be unsafe. Each hash is specified as "<type>:<value>",
	// where the only currently supported type is "sha256". This is a hex-encoded
	// SHA-256 hash of the Subject Public Key Info (SPKI) object in DER-encoded
	// ASN.1. These hashes can be calculated using, for example, OpenSSL.
	// +optional
	CACertHashes []string `json:"caCertHashes,omitempty" datapolicy:"security-key"`

	// UnsafeSkipCAVerification allows token-based discovery
	// without CA verification via CACertHashes. This can weaken
	// the security of kubeadm since other nodes can impersonate the control-plane.
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
	// +optional
	ReadOnly bool `json:"readOnly,omitempty"`
	// PathType is the type of the HostPath.
	// +optional
	PathType corev1.HostPathType `json:"pathType,omitempty"`
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
	// +optional
	Directory string `json:"directory,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ResetConfiguration contains a list of fields that are specifically "kubeadm reset"-only runtime information.
type ResetConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// CleanupTmpDir specifies whether the "/etc/kubernetes/tmp" directory should be cleaned during the reset process.
	// +optional
	CleanupTmpDir bool `json:"cleanupTmpDir,omitempty"`

	// CertificatesDir specifies the directory where the certificates are stored. If specified, it will be cleaned during the reset process.
	// +optional
	CertificatesDir string `json:"certificatesDir,omitempty"`

	// CRISocket is used to retrieve container runtime info and used for the removal of the containers.
	// If CRISocket is not specified by flag or config file, kubeadm will try to detect one valid CRISocket instead.
	// +optional
	CRISocket string `json:"criSocket,omitempty"`

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	// +optional
	DryRun bool `json:"dryRun,omitempty"`

	// Force flag instructs kubeadm to reset the node without prompting for confirmation.
	// +optional
	Force bool `json:"force,omitempty"`

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored during the reset process, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	// +optional
	IgnorePreflightErrors []string `json:"ignorePreflightErrors,omitempty"`

	// SkipPhases is a list of phases to skip during command execution.
	// The list of phases can be obtained with the "kubeadm reset phase --help" command.
	// +optional
	SkipPhases []string `json:"skipPhases,omitempty"`

	// UnmountFlags is a list of unmount2() syscall flags that kubeadm can use when unmounting
	// directories during "reset". A flag can be one of: MNT_FORCE, MNT_DETACH, MNT_EXPIRE, UMOUNT_NOFOLLOW.
	// By default this list is empty.
	// +optional
	UnmountFlags []string `json:"unmountFlags,omitempty"`

	// Timeouts holds various timeouts that apply to kubeadm commands.
	// +optional
	Timeouts *Timeouts `json:"timeouts,omitempty"`
}

// Arg represents an argument with a name and a value.
type Arg struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// EnvVar represents an environment variable present in a Container.
type EnvVar struct {
	corev1.EnvVar `json:",inline"`
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
	// Default: 4m
	// +optional
	ControlPlaneComponentHealthCheck *metav1.Duration `json:"controlPlaneComponentHealthCheck,omitempty"`

	// KubeletHealthCheck is the amount of time to wait for the kubelet to be healthy
	// during "kubeadm init" and "kubeadm join".
	// Default: 4m
	// +optional
	KubeletHealthCheck *metav1.Duration `json:"kubeletHealthCheck,omitempty"`

	// KubernetesAPICall is the amount of time to wait for the kubeadm client to complete a request to
	// the API server. This applies to all types of methods (GET, POST, etc).
	// Default: 1m
	// +optional
	KubernetesAPICall *metav1.Duration `json:"kubernetesAPICall,omitempty"`

	// EtcdAPICall is the amount of time to wait for the kubeadm etcd client to complete a request to
	// the etcd cluster.
	// Default: 2m
	// +optional
	EtcdAPICall *metav1.Duration `json:"etcdAPICall,omitempty"`

	// TLSBootstrap is the amount of time to wait for the kubelet to complete TLS bootstrap
	// for a joining node.
	// Default: 5m
	// +optional
	TLSBootstrap *metav1.Duration `json:"tlsBootstrap,omitempty"`

	// Discovery is the amount of time to wait for kubeadm to validate the API server identity
	// for a joining node.
	// Default: 5m
	// +optional
	Discovery *metav1.Duration `json:"discovery,omitempty"`

	// UpgradeManifests is the timeout for upgradring static Pod manifests
	// Default: 5m
	UpgradeManifests *metav1.Duration `json:"upgradeManifests,omitempty"`
}

// UpgradeApplyConfiguration contains a list of configurable options which are specific to the  "kubeadm upgrade apply" command.
type UpgradeApplyConfiguration struct {
	// KubernetesVersion is the target version of the control plane.
	// +optional
	KubernetesVersion string `json:"kubernetesVersion,omitempty"`

	// AllowExperimentalUpgrades instructs kubeadm to show unstable versions of Kubernetes as an upgrade
	// alternative and allows upgrading to an alpha/beta/release candidate version of Kubernetes.
	// Default: false
	// +optional
	AllowExperimentalUpgrades *bool `json:"allowExperimentalUpgrades,omitempty"`

	// Enable AllowRCUpgrades will show release candidate versions of Kubernetes as an upgrade alternative and
	// allows upgrading to a release candidate version of Kubernetes.
	// +optional
	AllowRCUpgrades *bool `json:"allowRCUpgrades,omitempty"`

	// CertificateRenewal instructs kubeadm to execute certificate renewal during upgrades.
	// Defaults to true.
	// +optional
	CertificateRenewal *bool `json:"certificateRenewal,omitempty"`

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	// +optional
	DryRun *bool `json:"dryRun,omitempty"`

	// EtcdUpgrade instructs kubeadm to execute etcd upgrade during upgrades.
	// Defaults to true.
	// +optional
	EtcdUpgrade *bool `json:"etcdUpgrade,omitempty"`

	// ForceUpgrade flag instructs kubeadm to upgrade the cluster without prompting for confirmation.
	// +optional
	ForceUpgrade *bool `json:"forceUpgrade,omitempty"`

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored during the upgrade process, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	// +optional
	IgnorePreflightErrors []string `json:"ignorePreflightErrors,omitempty"`

	// Patches contains options related to applying patches to components deployed by kubeadm during "kubeadm upgrade".
	// +optional
	Patches *Patches `json:"patches,omitempty"`

	// PrintConfig specifies whether the configuration file that will be used in the upgrade should be printed or not.
	// +optional
	PrintConfig *bool `json:"printConfig,omitempty"`

	// SkipPhases is a list of phases to skip during command execution.
	// NOTE: This field is currently ignored for "kubeadm upgrade apply", but in the future it will be supported.
	SkipPhases []string
}

// UpgradeDiffConfiguration contains a list of configurable options which are specific to the "kubeadm upgrade diff" command.
type UpgradeDiffConfiguration struct {
	// KubernetesVersion is the target version of the control plane.
	// +optional
	KubernetesVersion string `json:"kubernetesVersion,omitempty"`

	// DiffContextLines is the number of lines of context in the diff.
	// +optional
	DiffContextLines int `json:"contextLines,omitempty"`
}

// UpgradeNodeConfiguration contains a list of configurable options which are specific to the "kubeadm upgrade node" command.
type UpgradeNodeConfiguration struct {
	// CertificateRenewal instructs kubeadm to execute certificate renewal during upgrades.
	// Defaults to true.
	// +optional
	CertificateRenewal *bool `json:"certificateRenewal,omitempty"`

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	// +optional
	DryRun *bool `json:"dryRun,omitempty"`

	// EtcdUpgrade instructs kubeadm to execute etcd upgrade during upgrades.
	// Defaults to true.
	// +optional
	EtcdUpgrade *bool `json:"etcdUpgrade,omitempty"`

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored during the upgrade process, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	// +optional
	IgnorePreflightErrors []string `json:"ignorePreflightErrors,omitempty"`

	// SkipPhases is a list of phases to skip during command execution.
	// The list of phases can be obtained with the "kubeadm upgrade node phase --help" command.
	// +optional
	SkipPhases []string `json:"skipPhases,omitempty"`

	// Patches contains options related to applying patches to components deployed by kubeadm during "kubeadm upgrade".
	// +optional
	Patches *Patches `json:"patches,omitempty"`
}

// UpgradePlanConfiguration contains a list of configurable options which are specific to the "kubeadm upgrade plan" command.
type UpgradePlanConfiguration struct {
	// KubernetesVersion is the target version of the control plane.
	KubernetesVersion string `json:"kubernetesVersion,omitempty"`

	// AllowExperimentalUpgrades instructs kubeadm to show unstable versions of Kubernetes as an upgrade
	// alternative and allows upgrading to an alpha/beta/release candidate version of Kubernetes.
	// Default: false
	// +optional
	AllowExperimentalUpgrades *bool `json:"allowExperimentalUpgrades,omitempty"`

	// Enable AllowRCUpgrades will show release candidate versions of Kubernetes as an upgrade alternative and
	// allows upgrading to a release candidate version of Kubernetes.
	// +optional
	AllowRCUpgrades *bool `json:"allowRCUpgrades,omitempty"`

	// DryRun tells if the dry run mode is enabled, don't apply any change if it is and just output what would be done.
	// +optional
	DryRun *bool `json:"dryRun,omitempty"`

	// IgnorePreflightErrors provides a slice of pre-flight errors to be ignored during the upgrade process, e.g. 'IsPrivilegedUser,Swap'.
	// Value 'all' ignores errors from all checks.
	// +optional
	IgnorePreflightErrors []string `json:"ignorePreflightErrors,omitempty"`

	// PrintConfig specifies whether the configuration file that will be used in the upgrade should be printed or not.
	// +optional
	PrintConfig *bool `json:"printConfig,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UpgradeConfiguration contains a list of options that are specific to "kubeadm upgrade" subcommands.
type UpgradeConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Apply holds a list of options that are specific to the "kubeadm upgrade apply" command.
	// +optional
	Apply UpgradeApplyConfiguration `json:"apply,omitempty"`

	// Diff holds a list of options that are specific to the "kubeadm upgrade diff" command.
	// +optional
	Diff UpgradeDiffConfiguration `json:"diff,omitempty"`

	// Node holds a list of options that are specific to the "kubeadm upgrade node" command.
	// +optional
	Node UpgradeNodeConfiguration `json:"node,omitempty"`

	// Plan holds a list of options that are specific to the "kubeadm upgrade plan" command.
	// +optional
	Plan UpgradePlanConfiguration `json:"plan,omitempty"`

	// Timeouts holds various timeouts that apply to kubeadm commands.
	// +optional
	Timeouts *Timeouts `json:"timeouts,omitempty"`
}
