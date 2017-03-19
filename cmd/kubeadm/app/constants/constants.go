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
	"path"
	"time"

	"k8s.io/client-go/pkg/api/v1"
)

const (
	// KubernetesDir is the directory kubernetes owns for storing various configuration files
	KubernetesDir = "/etc/kubernetes"

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

	// Important: a "v"-prefix shouldn't exist here; semver doesn't allow that
	MinimumControlPlaneVersion = "1.6.0-beta.3"

	// Some well-known users and groups in the core Kubernetes authorization system

	ControllerManagerUser = "system:kube-controller-manager"
	SchedulerUser         = "system:kube-scheduler"
	MastersGroup          = "system:masters"
	NodesGroup            = "system:nodes"

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

	// DefaultAdmissionControl specifies the default admission control options that will be used
	DefaultAdmissionControl = "NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds"
)

var (

	// MasterToleration is the toleration to apply on the PodSpec for being able to run that Pod on the master
	MasterToleration = v1.Toleration{
		Key:    LabelNodeRoleMaster,
		Effect: v1.TaintEffectNoSchedule,
	}

	AuthorizationPolicyPath        = path.Join(KubernetesDir, "abac_policy.json")
	AuthorizationWebhookConfigPath = path.Join(KubernetesDir, "webhook_authz.conf")

	// DefaultTokenUsages specifies the default functions a token will get
	DefaultTokenUsages = []string{"signing", "authentication"}
)
