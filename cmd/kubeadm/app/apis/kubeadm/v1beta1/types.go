/*
Copyright 2017 The Kubernetes Authors.

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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// MasterConfiguration contains all necessary information for kubeadm to bootstrap a cluster
type MasterConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// APIServer specifies how the api server component is set up and configured
	APIServer APIServer `json:"apiServer"`
	// ControllerManager specifies how the controller manager component is set up and configured
	ControllerManager ControllerManager `json:"controllerManager"`
	// Scheduler specifies how the scheduler component is set up and configured
	Scheduler Scheduler `json:"scheduler"`
	// Etcd specifies how etcd is managed (by kubeadm or externally), and how the API server can connect to the etcd cluster
	// kubeadm defaults to deploying a local/single-node etcd cluster
	Etcd Etcd `json:"etcd"`
	// Networking defines a couple of cluster-wide properties related to the networking setup
	Networking Networking `json:"networking"`
	// BootstrapToken controls how/if bootstrap tokens should be enabled or how they should be created
	BootstrapToken BootstrapToken `json:"bootstrapToken"`
	// Paths specifies locations to various important directories and files during initialization time
	Paths MasterPaths `json:"paths"`

	// KubernetesVersion controls which version to use for the control plane, the kube-proxy and which other features will be enabled
	KubernetesVersion string `json:"kubernetesVersion"`
	// NodeName specifies the name the Node API object will have in a later stage. Defaulting to os.Hostname(), but can be useful to set in a custom environment
	NodeName string `json:"nodeName"`
	// FeatureGates specifies which kubeadm-specific features to enable/disable on demand
	FeatureGates map[string]bool `json:"featureGates"`
	// ImageRepository specifies what container registry to pull control plane images from
	ImageRepository string `json:"imageRepository"`
}

// APIServer specifies how the api server component is set up and configured
type APIServer struct {
	// Image specifies which container image to use for running the api server. If empty, automatically populated by kubeadm using the ImageRepository and KubernetesVersion
	Image string `json:"image"`
	// ExtraArgs lets the user customize and/or override the arguments that are given to the api server
	ExtraArgs map[string]string `json:"extraArgs"`
	// ExtraHostPathVolumes lets the user add new hostPath volumes and volumeMounts as needed
	ExtraHostPathVolumes []HostPathVolume `json:"extraHostPathVolumes"`

	// AdvertiseAddress sets the address for the API server to advertise.
	AdvertiseAddress string `json:"advertiseAddress"`
	// BindPort sets the secure port for the API Server to bind to
	BindPort int32 `json:"bindPort"`
	// AuthorizationModes specifies which authorization modes to use for the api server
	AuthorizationModes []string `json:"authorizationModes"`
	// ServingCertExtraSANs sets extra Subject Alternative Names for the API Server signing cert
	ServingCertExtraSANs []string `json:"servingCertExtraSANs"`
}

// ControllerManager specifies how the controller manager component is set up and configured
type ControllerManager struct {
	// Image specifies which container image to use for running the controller manager. If empty, automatically populated by kubeadm using the ImageRepository and KubernetesVersion
	Image string `json:"image"`
	// ExtraArgs lets the user customize and/or override the arguments that are given to the controller manager
	ExtraArgs map[string]string `json:"extraArgs"`
	// ExtraHostPathVolumes lets the user add new hostPath volumes and volumeMounts as needed
	ExtraHostPathVolumes []HostPathVolume `json:"extraHostPathVolumes"`
}

// Scheduler specifies how the scheduler component is set up and configured
type Scheduler struct {
	// Image specifies which container image to use for running the scheduler. If empty, automatically populated by kubeadm using the ImageRepository and KubernetesVersion
	Image string `json:"image"`
	// ExtraArgs lets the user customize and/or override the arguments that are given to the scheduler
	ExtraArgs map[string]string `json:"extraArgs"`
	// ExtraHostPathVolumes lets the user add new hostPath volumes and volumeMounts as needed
	ExtraHostPathVolumes []HostPathVolume `json:"extraHostPathVolumes"`
}

// HostPathVolume is a high-level abstraction for a v1.Volume and v1.VolumeMount attached to the Pod the component is running in
type HostPathVolume struct {
	// Name specifies the name of the Volume and VolumeMount
	Name string `json:"name"`
	// HostPath specifies the path on the host (can be a file or a directory) that should be mounted into the container
	HostPath string `json:"hostPath"`
	// ContainerPath specifies the path inside of the container (can be a file or a directory) that should contain the data mounted from the host
	ContainerPath string `json:"containerPath"`
}

// Networking defines a couple of cluster-wide properties related to the networking setup
type Networking struct {
	// ServiceSubnet specifies which subnet to use for virtual Service IPs
	ServiceSubnet string `json:"serviceSubnet"`
	// PodSubnet specifies which subnet to use for Pods in the cluster. Setting this is optional.
	// Setting this is enables the controller-manager's built-in subnet allocator that will set Node.Spec.PodCIDR,
	// which is required information for some third-party CNI network plugins
	PodSubnet string `json:"podSubnet"`
	// DNSDomain sets the DNS domain for the cluster; currently used when generating the API Server serving certificate
	DNSDomain string `json:"dnsDomain"`
}

// BootstrapToken controls how/if bootstrap tokens should be enabled or how they should be created
type BootstrapToken struct {
	// Node describes a Node Bootstrap Token that is automatically generated on cluster initialization
	// If this pointer is nil, this field defaults to being enabled, having an auto-generated token and having a time-limited ttl 
	Node *NodeBootstrapToken `json:"node"`
}

// NodeBootstrapToken controls how/if node bootstrap tokens should be enabled or how it should be created
type NodeBootstrapToken struct {
	// Enabled specifies whether a Node Bootstrap Token should automatically be generated and the cluster-info ConfigMap should be created
	Enabled bool `json:"enabled"`
	// Token specifies which Node Bootstrap Token in the "<6 chars>.<16 chars>" format to automatically add to the cluster
	Token string `json:"token"`
	// TokenTTL specifies how long the automatically generated Node Bootstrap Token should be valid for
	TokenTTL time.Duration `json:"tokenTTL"`
}

// Etcd specifies how etcd is managed (by kubeadm or externally), and how the API server can connect to the etcd cluster
// kubeadm defaults to deploying a local/single-node etcd cluster
type Etcd struct {
	// Local describes a kubeadm-managed etcd cluster running locally on the master node(s)
	Local *LocalEtcd `json:"local"`
	// External describes an etcd cluster running externally to the kubeadm-managed Kubernetes cluster
	External *ExternalEtcd `json:"external"`
}

// LocalEtcd specifies how kubeadm sets up its local, managed etcd cluster
type LocalEtcd struct {
	// Image specifies which container image to use for running etcd. If empty, automatically populated by kubeadm using the image repository and default etcd version
	Image string `json:"image"`
	// DataDir specifies which data directory for etcd to use and mount into the container via a hostPath mount
	DataDir string `json:"dataDir"`
	// ExtraArgs lets the user customize and/or override the arguments that are given to etcd
	ExtraArgs map[string]string `json:"extraArgs"`
}

// ExternalEtcd specifies how the API server can talk to the external etcd cluster
type ExternalEtcd struct {
	// Endpoints specifies to which endpoints the API server should connect to when talking to the external etcd cluster
	Endpoints []string `json:"endpoints"`
	// CACert specifies a path to the CA certificate to trust for the etcd serving certificate
	CACert string `json:"caCert"`
	// ClientCert specifies a path to the client certificate for the API server to use when talking to etcd
	ClientCert string `json:"certFile"`
	// ClientKey specifies a path to the client certificate's private key for the API server to use when talking to etcd
	ClientKey string `json:"keyFile"`
}

// MasterPaths specifies locations to various important directories and files during initialization time
type MasterPaths struct {
	// CertificatesDir specifies where to store or look for all required certificates
	CertificatesDir string `json:"certificatesDir"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type NodeConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	Discovery    NodeDiscovery `json:"discovery"`
	TLSBootstrap TLSBootstrap  `json:"tlsBootstrap"`

	NodeName       string `json:"nodeName"`
	ShortHandToken string `json:"shortHandToken"`
}

type NodeDiscovery struct {
	File  *DiscoveryFile  `json:"file"`
	Token *DiscoveryToken `json:"token"`
}

type DiscoveryFile struct {
	Path string `json:"path"`
	// TODO: Make it possible to inline the discovery file in this API?
	// FileBytes []byte `json:"fileBytes"`
}

type DiscoveryToken struct {
	Token      string   `json:"token"`
	APIServers []string `json:"apiServers"`
}

type TLSBootstrap struct {
	Token string `json:"token"`
}

type NodePaths struct {
	// CACertPath specifies where the CA cert of the cluster should be stored
	CACertPath string `json:"caCertPath"`
}
