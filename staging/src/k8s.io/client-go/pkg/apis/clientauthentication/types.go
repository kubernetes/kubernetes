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

package clientauthentication

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ExecCredential is used by exec-based plugins to communicate credentials to
// HTTP transports.
type ExecCredential struct {
	metav1.TypeMeta

	// Spec holds information passed to the plugin by the transport. This contains
	// request and runtime specific information, such as if the session is interactive.
	Spec ExecCredentialSpec

	// Status is filled in by the plugin and holds the credentials that the transport
	// should use to contact the API.
	// +optional
	Status *ExecCredentialStatus
}

// ExecCredentialSpec holds request and runtime specific information provided by
// the transport.
type ExecCredentialSpec struct {
	// Interactive is true when the transport detects the command is being called from an
	// interactive prompt, i.e., when stdin has been passed to this exec plugin.
	// +optional
	Interactive bool

	// Cluster contains information to allow an exec plugin to communicate with the
	// kubernetes cluster being authenticated to. Note that Cluster is non-nil only
	// when provideClusterInfo is set to true in the exec provider config (i.e.,
	// ExecConfig.ProvideClusterInfo).
	// +optional
	Cluster *Cluster
}

// ExecCredentialStatus holds credentials for the transport to use.
type ExecCredentialStatus struct {
	// ExpirationTimestamp indicates a time when the provided credentials expire.
	// +optional
	ExpirationTimestamp *metav1.Time
	// Token is a bearer token used by the client for request authentication.
	// +optional
	Token string `datapolicy:"token"`
	// PEM-encoded client TLS certificate.
	// +optional
	ClientCertificateData string
	// PEM-encoded client TLS private key.
	// +optional
	ClientKeyData string `datapolicy:"secret-key"`
}

// Cluster contains information to allow an exec plugin to communicate
// with the kubernetes cluster being authenticated to.
//
// To ensure that this struct contains everything someone would need to communicate
// with a kubernetes cluster (just like they would via a kubeconfig), the fields
// should shadow "k8s.io/client-go/tools/clientcmd/api/v1".Cluster, with the exception
// of CertificateAuthority, since CA data will always be passed to the plugin as bytes.
type Cluster struct {
	// Server is the address of the kubernetes cluster (https://hostname:port).
	Server string
	// TLSServerName is passed to the server for SNI and is used in the client to
	// check server certificates against. If ServerName is empty, the hostname
	// used to contact the server is used.
	// +optional
	TLSServerName string
	// InsecureSkipTLSVerify skips the validity check for the server's certificate.
	// This will make your HTTPS connections insecure.
	// +optional
	InsecureSkipTLSVerify bool
	// CAData contains PEM-encoded certificate authority certificates.
	// If empty, system roots should be used.
	// +listType=atomic
	// +optional
	CertificateAuthorityData []byte
	// ProxyURL is the URL to the proxy to be used for all requests to this
	// cluster.
	// +optional
	ProxyURL string
	// DisableCompression allows client to opt-out of response compression for all requests to the server. This is useful
	// to speed up requests (specifically lists) when client-server network bandwidth is ample, by saving time on
	// compression (server-side) and decompression (client-side): https://github.com/kubernetes/kubernetes/issues/112296.
	// +optional
	DisableCompression bool
	// Config holds additional config data that is specific to the exec
	// plugin with regards to the cluster being authenticated to.
	//
	// This data is sourced from the clientcmd Cluster object's
	// extensions[client.authentication.k8s.io/exec] field:
	//
	// clusters:
	// - name: my-cluster
	//   cluster:
	//     ...
	//     extensions:
	//     - name: client.authentication.k8s.io/exec  # reserved extension name for per cluster exec config
	//       extension:
	//         audience: 06e3fbd18de8  # arbitrary config
	//
	// In some environments, the user config may be exactly the same across many clusters
	// (i.e. call this exec plugin) minus some details that are specific to each cluster
	// such as the audience.  This field allows the per cluster config to be directly
	// specified with the cluster info.  Using this field to store secret data is not
	// recommended as one of the prime benefits of exec plugins is that no secrets need
	// to be stored directly in the kubeconfig.
	// +optional
	Config runtime.Object
}
