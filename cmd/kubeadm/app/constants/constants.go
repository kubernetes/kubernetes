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
	"path/filepath"
	"time"

	"k8s.io/client-go/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util/version"
)

const (
	// KubernetesDir is the directory kubernetes owns for storing various configuration files
	KubernetesDir = "/etc/kubernetes"

	ManifestsSubDirName = "manifests"

	CACertAndKeyBaseName = "ca"
	CACertName           = "ca.crt"
	CAKeyName            = "ca.key"

	APIServerCertAndKeyBaseName = "apiserver"
	APIServerCertName           = "apiserver.crt"
	APIServerKeyName            = "apiserver.key"

	APIServerKubeletClientCertAndKeyBaseName = "apiserver-kubelet-client"
	APIServerKubeletClientCertName           = "apiserver-kubelet-client.crt"
	APIServerKubeletClientKeyName            = "apiserver-kubelet-client.key"

	ServiceAccountKeyBaseName    = "sa"
	ServiceAccountPublicKeyName  = "sa.pub"
	ServiceAccountPrivateKeyName = "sa.key"

	FrontProxyCACertAndKeyBaseName = "front-proxy-ca"
	FrontProxyCACertName           = "front-proxy-ca.crt"
	FrontProxyCAKeyName            = "front-proxy-ca.key"

	FrontProxyClientCertAndKeyBaseName = "front-proxy-client"
	FrontProxyClientCertName           = "front-proxy-client.crt"
	FrontProxyClientKeyName            = "front-proxy-client.key"

	AdminKubeConfigFileName             = "admin.conf"
	KubeletKubeConfigFileName           = "kubelet.conf"
	ControllerManagerKubeConfigFileName = "controller-manager.conf"
	SchedulerKubeConfigFileName         = "scheduler.conf"

	// Some well-known users and groups in the core Kubernetes authorization system

	ControllerManagerUser   = "system:kube-controller-manager"
	SchedulerUser           = "system:kube-scheduler"
	MastersGroup            = "system:masters"
	NodesGroup              = "system:nodes"
	NodesClusterRoleBinding = "system:node"

	// Constants for what we name our ServiceAccounts with limited access to the cluster in case of RBAC
	KubeDNSServiceAccountName   = "kube-dns"
	KubeProxyServiceAccountName = "kube-proxy"

	// APICallRetryInterval defines how long kubeadm should wait before retrying a failed API operation
	APICallRetryInterval = 500 * time.Millisecond
	// DiscoveryRetryInterval specifies how long kubeadm should wait before retrying to connect to the master when doing discovery
	DiscoveryRetryInterval = 5 * time.Second

	// Minimum amount of nodes the Service subnet should allow.
	// We need at least ten, because the DNS service is always at the tenth cluster clusterIP
	MinimumAddressesInServiceSubnet = 10

	// DefaultTokenDuration specifies the default amount of time that a bootstrap token will be valid
	// Default behaviour is "never expire" == 0
	DefaultTokenDuration = 0

	// LabelNodeRoleMaster specifies that a node is a master
	// It's copied over to kubeadm until it's merged in core: https://github.com/kubernetes/kubernetes/pull/39112
	LabelNodeRoleMaster = "node-role.kubernetes.io/master"

	// MinExternalEtcdVersion indicates minimum external etcd version which kubeadm supports
	MinExternalEtcdVersion = "3.0.14"
)

var (

	// MasterToleration is the toleration to apply on the PodSpec for being able to run that Pod on the master
	MasterToleration = v1.Toleration{
		Key:    LabelNodeRoleMaster,
		Effect: v1.TaintEffectNoSchedule,
	}

	AuthorizationPolicyPath        = filepath.Join(KubernetesDir, "abac_policy.json")
	AuthorizationWebhookConfigPath = filepath.Join(KubernetesDir, "webhook_authz.conf")

	// DefaultTokenUsages specifies the default functions a token will get
	DefaultTokenUsages = []string{"signing", "authentication"}

	// MinimumControlPlaneVersion specifies the minimum control plane version kubeadm can deploy
	MinimumControlPlaneVersion = version.MustParseSemantic("v1.6.0")

	// MinimumCSRSARApproverVersion specifies the minimum kubernetes version that can be used for enabling the new-in-v1.7 CSR approver based on a SubjectAccessReview
	MinimumCSRSARApproverVersion = version.MustParseSemantic("v1.7.0-beta.0")

	// MinimumAPIAggregationVersion specifies the minimum kubernetes version that can be used enabling the API aggregation in the apiserver and the front proxy flags
	MinimumAPIAggregationVersion = version.MustParseSemantic("v1.7.0-alpha.1")

	// MinimumNodeAuthorizerVersion specifies the minimum kubernetes version that can be used for enabling the node authorizer
	MinimumNodeAuthorizerVersion = version.MustParseSemantic("v1.7.0-beta.1")
)
