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

package constants

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/util/version"
)

// KubernetesDir is the directory kubernetes owns for storing various configuration files
// This semi-constant MUST NOT be modified during runtime. It's a variable solely for use in unit testing.
var KubernetesDir = "/etc/kubernetes"

const (
	// ManifestsSubDirName defines directory name to store manifests
	ManifestsSubDirName = "manifests"
	// TempDirForKubeadm defines temporary directory for kubeadm
	TempDirForKubeadm = "/etc/kubernetes/tmp"

	// CACertAndKeyBaseName defines certificate authority base name
	CACertAndKeyBaseName = "ca"
	// CACertName defines certificate name
	CACertName = "ca.crt"
	// CAKeyName defines certificate name
	CAKeyName = "ca.key"

	// APIServerCertAndKeyBaseName defines API's server certificate and key base name
	APIServerCertAndKeyBaseName = "apiserver"
	// APIServerCertName defines API's server certificate name
	APIServerCertName = "apiserver.crt"
	// APIServerKeyName defines API's server key name
	APIServerKeyName = "apiserver.key"
	// APIServerCertCommonName defines API's server certificate common name (CN)
	APIServerCertCommonName = "kube-apiserver"

	// APIServerKubeletClientCertAndKeyBaseName defines kubelet client certificate and key base name
	APIServerKubeletClientCertAndKeyBaseName = "apiserver-kubelet-client"
	// APIServerKubeletClientCertName defines kubelet client certificate name
	APIServerKubeletClientCertName = "apiserver-kubelet-client.crt"
	// APIServerKubeletClientKeyName defines kubelet client key name
	APIServerKubeletClientKeyName = "apiserver-kubelet-client.key"
	// APIServerKubeletClientCertCommonName defines kubelet client certificate common name (CN)
	APIServerKubeletClientCertCommonName = "kube-apiserver-kubelet-client"

	// ServiceAccountKeyBaseName defines SA key base name
	ServiceAccountKeyBaseName = "sa"
	// ServiceAccountPublicKeyName defines SA public key base name
	ServiceAccountPublicKeyName = "sa.pub"
	// ServiceAccountPrivateKeyName defines SA private key base name
	ServiceAccountPrivateKeyName = "sa.key"

	// FrontProxyCACertAndKeyBaseName defines front proxy CA certificate and key base name
	FrontProxyCACertAndKeyBaseName = "front-proxy-ca"
	// FrontProxyCACertName defines front proxy CA certificate name
	FrontProxyCACertName = "front-proxy-ca.crt"
	// FrontProxyCAKeyName defaines front proxy CA key name
	FrontProxyCAKeyName = "front-proxy-ca.key"

	// FrontProxyClientCertAndKeyBaseName defines front proxy certificate and key base name
	FrontProxyClientCertAndKeyBaseName = "front-proxy-client"
	// FrontProxyClientCertName defines front proxy certificate name
	FrontProxyClientCertName = "front-proxy-client.crt"
	// FrontProxyClientKeyName defines front proxy key name
	FrontProxyClientKeyName = "front-proxy-client.key"
	// FrontProxyClientCertCommonName defines front proxy certificate common name
	FrontProxyClientCertCommonName = "front-proxy-client" //used as subject.commonname attribute (CN)

	// AdminKubeConfigFileName defines name for the KubeConfig aimed to be used by the superuser/admin of the cluster
	AdminKubeConfigFileName = "admin.conf"
	// KubeletBootstrapKubeConfigFileName defines the file name for the KubeConfig that the kubelet will use to do
	// the TLS bootstrap to get itself an unique credential
	KubeletBootstrapKubeConfigFileName = "bootstrap-kubelet.conf"

	// KubeletKubeConfigFileName defines the file name for the KubeConfig that the master kubelet will use for talking
	// to the API server
	KubeletKubeConfigFileName = "kubelet.conf"
	// ControllerManagerKubeConfigFileName defines the file name for the controller manager's KubeConfig file
	ControllerManagerKubeConfigFileName = "controller-manager.conf"
	// SchedulerKubeConfigFileName defines the file name for the scheduler's KubeConfig file
	SchedulerKubeConfigFileName = "scheduler.conf"

	// Some well-known users and groups in the core Kubernetes authorization system

	// ControllerManagerUser defines the well-known user the controller-manager should be authenticated as
	ControllerManagerUser = "system:kube-controller-manager"
	// SchedulerUser defines the well-known user the scheduler should be authenticated as
	SchedulerUser = "system:kube-scheduler"
	// MastersGroup defines the well-known group for the apiservers. This group is also superuser by default
	// (i.e. bound to the cluster-admin ClusterRole)
	MastersGroup = "system:masters"
	// NodesGroup defines the well-known group for all nodes.
	NodesGroup = "system:nodes"
	// NodesClusterRoleBinding defines the well-known ClusterRoleBinding which binds the too permissive system:node
	// ClusterRole to the system:nodes group. Since kubeadm is using the Node Authorizer, this ClusterRoleBinding's
	// system:nodes group subject is removed if present.
	NodesClusterRoleBinding = "system:node"

	// APICallRetryInterval defines how long kubeadm should wait before retrying a failed API operation
	APICallRetryInterval = 500 * time.Millisecond
	// DiscoveryRetryInterval specifies how long kubeadm should wait before retrying to connect to the master when doing discovery
	DiscoveryRetryInterval = 5 * time.Second
	// MarkMasterTimeout specifies how long kubeadm should wait for applying the label and taint on the master before timing out
	MarkMasterTimeout = 2 * time.Minute

	// MinimumAddressesInServiceSubnet defines minimum amount of nodes the Service subnet should allow.
	// We need at least ten, because the DNS service is always at the tenth cluster clusterIP
	MinimumAddressesInServiceSubnet = 10

	// DefaultTokenDuration specifies the default amount of time that a bootstrap token will be valid
	// Default behaviour is 24 hours
	DefaultTokenDuration = 24 * time.Hour

	// LabelNodeRoleMaster specifies that a node is a master
	// It's copied over to kubeadm until it's merged in core: https://github.com/kubernetes/kubernetes/pull/39112
	LabelNodeRoleMaster = "node-role.kubernetes.io/master"

	// LabelNodeRoleExcludeBalancer specifies that the node should be
	// exclude from load balancers created by a cloud provider.
	LabelNodeRoleExcludeBalancer = "node.role.kubernetes.io/exclude-balancer"

	// MasterConfigurationConfigMap specifies in what ConfigMap in the kube-system namespace the `kubeadm init` configuration should be stored
	MasterConfigurationConfigMap = "kubeadm-config"

	// MasterConfigurationConfigMapKey specifies in what ConfigMap key the master configuration should be stored
	MasterConfigurationConfigMapKey = "MasterConfiguration"

	// MinExternalEtcdVersion indicates minimum external etcd version which kubeadm supports
	MinExternalEtcdVersion = "3.0.14"

	// DefaultEtcdVersion indicates the default etcd version that kubeadm uses
	DefaultEtcdVersion = "3.0.17"

	// Etcd defines variable used internally when referring to etcd component
	Etcd = "etcd"
	// KubeAPIServer defines variable used internally when referring to kube-apiserver component
	KubeAPIServer = "kube-apiserver"
	// KubeControllerManager defines variable used internally when referring to kube-controller-manager component
	KubeControllerManager = "kube-controller-manager"
	// KubeScheduler defines variable used internally when referring to kube-scheduler component
	KubeScheduler = "kube-scheduler"
	// KubeProxy defines variable used internally when referring to kube-proxy component
	KubeProxy = "kube-proxy"

	// SelfHostingPrefix describes the prefix workloads that are self-hosted by kubeadm has
	SelfHostingPrefix = "self-hosted-"

	// KubeCertificatesVolumeName specifies the name for the Volume that is used for injecting certificates to control plane components (can be both a hostPath volume or a projected, all-in-one volume)
	KubeCertificatesVolumeName = "k8s-certs"

	// KubeConfigVolumeName specifies the name for the Volume that is used for injecting the kubeconfig to talk securely to the api server for a control plane component if applicable
	KubeConfigVolumeName = "kubeconfig"

	// V17NodeBootstrapTokenAuthGroup specifies which group a Node Bootstrap Token should be authenticated in, in v1.7
	V17NodeBootstrapTokenAuthGroup = "system:bootstrappers"

	// V18NodeBootstrapTokenAuthGroup specifies which group a Node Bootstrap Token should be authenticated in, in v1.8
	V18NodeBootstrapTokenAuthGroup = "system:bootstrappers:kubeadm:default-node-token"

	// DefaultCIImageRepository points to image registry where CI uploads images from ci-cross build job
	DefaultCIImageRepository = "gcr.io/kubernetes-ci-images"
)

var (

	// MasterTaint is the taint to apply on the PodSpec for being able to run that Pod on the master
	MasterTaint = v1.Taint{
		Key:    LabelNodeRoleMaster,
		Effect: v1.TaintEffectNoSchedule,
	}

	// MasterToleration is the toleration to apply on the PodSpec for being able to run that Pod on the master
	MasterToleration = v1.Toleration{
		Key:    LabelNodeRoleMaster,
		Effect: v1.TaintEffectNoSchedule,
	}

	// AuthorizationPolicyPath defines the supported location of authorization policy file
	AuthorizationPolicyPath = filepath.Join(KubernetesDir, "abac_policy.json")
	// AuthorizationWebhookConfigPath defines the supported location of webhook config file
	AuthorizationWebhookConfigPath = filepath.Join(KubernetesDir, "webhook_authz.conf")

	// DefaultTokenUsages specifies the default functions a token will get
	DefaultTokenUsages = []string{"signing", "authentication"}

	// MasterComponents defines the master component names
	MasterComponents = []string{KubeAPIServer, KubeControllerManager, KubeScheduler}

	// MinimumControlPlaneVersion specifies the minimum control plane version kubeadm can deploy
	MinimumControlPlaneVersion = version.MustParseSemantic("v1.7.0")

	// MinimumCSRAutoApprovalClusterRolesVersion defines whether kubeadm can rely on the built-in CSR approval ClusterRole or not (note, the binding is always created by kubeadm!)
	// TODO: Remove this when the v1.9 cycle starts and we bump the minimum supported version to v1.8.0
	MinimumCSRAutoApprovalClusterRolesVersion = version.MustParseSemantic("v1.8.0-alpha.3")

	// UseEnableBootstrapTokenAuthFlagVersion defines the first version where the API server supports the --enable-bootstrap-token-auth flag instead of the old and deprecated flag.
	// TODO: Remove this when the v1.9 cycle starts and we bump the minimum supported version to v1.8.0
	UseEnableBootstrapTokenAuthFlagVersion = version.MustParseSemantic("v1.8.0-beta.0")

	// MinimumKubeletVersion specifies the minimum version of kubelet which kubeadm supports
	MinimumKubeletVersion = version.MustParseSemantic("v1.8.0-beta.0")
)

// GetStaticPodDirectory returns the location on the disk where the Static Pod should be present
func GetStaticPodDirectory() string {
	return filepath.Join(KubernetesDir, ManifestsSubDirName)
}

// GetStaticPodFilepath returns the location on the disk where the Static Pod should be present
func GetStaticPodFilepath(componentName, manifestsDir string) string {
	return filepath.Join(manifestsDir, componentName+".yaml")
}

// GetAdminKubeConfigPath returns the location on the disk where admin kubeconfig is located by default
func GetAdminKubeConfigPath() string {
	return filepath.Join(KubernetesDir, AdminKubeConfigFileName)
}

// AddSelfHostedPrefix adds the self-hosted- prefix to the component name
func AddSelfHostedPrefix(componentName string) string {
	return fmt.Sprintf("%s%s", SelfHostingPrefix, componentName)
}

// CreateTempDirForKubeadm is a function that creates a temporary directory under /etc/kubernetes/tmp (not using /tmp as that would potentially be dangerous)
func CreateTempDirForKubeadm(dirName string) (string, error) {
	// creates target folder if not already exists
	if err := os.MkdirAll(TempDirForKubeadm, 0700); err != nil {
		return "", fmt.Errorf("failed to create directory %q: %v", TempDirForKubeadm, err)
	}

	tempDir, err := ioutil.TempDir(TempDirForKubeadm, dirName)
	if err != nil {
		return "", fmt.Errorf("couldn't create a temporary directory: %v", err)
	}
	return tempDir, nil
}

// GetNodeBootstrapTokenAuthGroup gets the bootstrap token auth group conditionally based on version
func GetNodeBootstrapTokenAuthGroup(k8sVersion *version.Version) string {
	if k8sVersion.AtLeast(UseEnableBootstrapTokenAuthFlagVersion) {
		return V18NodeBootstrapTokenAuthGroup
	}
	return V17NodeBootstrapTokenAuthGroup
}
