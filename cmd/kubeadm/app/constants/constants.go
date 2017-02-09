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

import "time"

const (
	CACertAndKeyBaseName = "ca"
	CACertName           = "ca.crt"
	CAKeyName            = "ca.key"

	APIServerCertAndKeyBaseName = "apiserver"
	APIServerCertName           = "apiserver.crt"
	APIServerKeyName            = "apiserver.key"

	APIServerKubeletClientCertAndKeyBaseName = "apiserver-kubelet-client"
	APIServerKubeletClientCertName           = "apiserver-kubelet-client.crt"
	APIServerKubeletClientKeyName            = "apiserver-kubelet-client.key"

	// TODO: These constants should actually come from pkg/kubeapiserver/authorizer, but we can't vendor that package in now
	// because of all the other sub-packages that would get vendored. To fix this, a pkg/kubeapiserver/authorizer/modes package
	// or similar should exist that only has these constants; then we can vendor it.
	AuthzModeAlwaysAllow = "AlwaysAllow"
	AuthzModeABAC        = "ABAC"
	AuthzModeRBAC        = "RBAC"
	AuthzModeWebhook     = "Webhook"

	// Important: a "v"-prefix shouldn't exist here; semver doesn't allow that
	MinimumControlPlaneVersion = "1.6.0-alpha.1"

	// Constants for what we name our ServiceAccounts with limited access to the cluster in case of RBAC
	KubeDNSServiceAccountName   = "kube-dns"
	KubeProxyServiceAccountName = "kube-proxy"

	// APICallRetryInterval defines how long kubeadm should wait before retrying a failed API operation
	APICallRetryInterval = 500 * time.Millisecond

	// Minimum amount of nodes the Service subnet should allow.
	// We need at least ten, because the DNS service is always at the tenth cluster clusterIP
	MinimumAddressesInServiceSubnet = 10
)
