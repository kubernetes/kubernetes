/*
Copyright 2014 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
)

// Where possible, json tags match the cli argument names.
// Top level config objects and all values required for proper functioning are not "omitempty".  Any truly optional piece of config is allowed to be omitted.

// Config holds the information needed to build connect to remote kubernetes clusters as a given user
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
type Config struct {
	// Legacy field from pkg/api/types.go TypeMeta.
	// TODO(jlowdermilk): remove this after eliminating downstream dependencies.
	// +k8s:conversion-gen=false
	// +optional
	Kind string `json:"kind,omitempty"`
	// Legacy field from pkg/api/types.go TypeMeta.
	// TODO(jlowdermilk): remove this after eliminating downstream dependencies.
	// +k8s:conversion-gen=false
	// +optional
	APIVersion string `json:"apiVersion,omitempty"`
	// `preferences` holds general information to be use for cli interactions
	Preferences Preferences `json:"preferences"`
	// `clusters` is a map of referencable names to cluster configs.
	Clusters []NamedCluster `json:"clusters"`
	// `users` is a map of referencable names to user configs
	AuthInfos []NamedAuthInfo `json:"users"`
	// `contexts` is a map of referencable names to context configs.
	Contexts []NamedContext `json:"contexts"`
	// `current-context` is the name of the context that you would like to use by default
	CurrentContext string `json:"current-context"`
	// `extensions` holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields
	// +optional
	Extensions []NamedExtension `json:"extensions,omitempty"`
}

type Preferences struct {
	// `colors` specifies whether the console output is generated with syntax highlights. Currently unused.
	// +optional
	Colors bool `json:"colors,omitempty"`
	// `extensions` holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields.
	// +optional
	Extensions []NamedExtension `json:"extensions,omitempty"`
}

// Cluster contains information about how to communicate with a kubernetes cluster
type Cluster struct {
	// `server` is the address of the kubernetes cluster (https://hostname:port).
	Server string `json:"server"`
	// `tls-server-name` is used to check server certificate. If `tls-server-name` is empty, the hostname used to contact the server is used.
	// +optional
	TLSServerName string `json:"tls-server-name,omitempty"`
	// `insecure-skip-tls-verify` skips the validity check for the server's certificate. This will make your HTTPS connections insecure.
	// +optional
	InsecureSkipTLSVerify bool `json:"insecure-skip-tls-verify,omitempty"`
	// `certificate-authority` is the path to a certificate file for the certificate authority.
	// +optional
	CertificateAuthority string `json:"certificate-authority,omitempty"`
	// `certificate-authority-data` contains PEM-encoded certificate authority certificates. Overrides `certificate-authority`.
	// +optional
	CertificateAuthorityData []byte `json:"certificate-authority-data,omitempty"`
	// `proxy-url` is the URL to the proxy to be used for all requests made by this
	// client. URLs with "http", "https", and "socks5" schemes are supported.  If
	// this configuration is not provided or the empty string, the client
	// attempts to construct a proxy configuration from `http_proxy` and
	// `https_proxy` environment variables. If these environment variables are not
	// set, the client does not attempt to proxy requests.
	//
	// "socks5" proxying does not currently support spdy streaming endpoints (exec,
	// attach, port forward).
	// +optional
	ProxyURL string `json:"proxy-url,omitempty"`
	// `disable-compression` allows client to opt-out of response compression for all requests to the server. This is useful
	// to speed up requests (specifically lists) when client-server network bandwidth is ample, by saving time on
	// compression (server-side) and decompression (client-side): https://github.com/kubernetes/kubernetes/issues/112296.
	// +optional
	DisableCompression bool `json:"disable-compression,omitempty"`
	// `extensions` holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields.
	// +optional
	Extensions []NamedExtension `json:"extensions,omitempty"`
}

// AuthInfo contains information that describes identity information.  This is use to tell the kubernetes cluster who you are.
type AuthInfo struct {
	// `client-certificate` is the path to a client cert file for TLS.
	// +optional
	ClientCertificate string `json:"client-certificate,omitempty"`
	// `client-certificate-data` contains PEM-encoded data from a client cert file for TLS. Overrides `client-certificate`.
	// +optional
	ClientCertificateData []byte `json:"client-certificate-data,omitempty"`
	// `client-key` is the path to a client key file for TLS.
	// +optional
	ClientKey string `json:"client-key,omitempty"`
	// `client-key-data` contains PEM-encoded data from a client key file for TLS. Overrides `client-key`.
	// +optional
	ClientKeyData []byte `json:"client-key-data,omitempty" datapolicy:"security-key"`
	// `token` is the bearer token for authentication to the kubernetes cluster.
	// +optional
	Token string `json:"token,omitempty" datapolicy:"token"`
	// `tokenFile` is a pointer to a file that contains a bearer token (as described above).
	// If both `token` and `tokenFile` are present, `token` takes precedence.
	// +optional
	TokenFile string `json:"tokenFile,omitempty"`
	// `as` is the username to act-as.
	// +optional
	Impersonate string `json:"as,omitempty"`
	// `as-uid` is the uid to impersonate.
	// +optional
	ImpersonateUID string `json:"as-uid,omitempty"`
	// `as-groups` is the groups to impersonate.
	// +optional
	ImpersonateGroups []string `json:"as-groups,omitempty"`
	// `as-user-extra` contains additional information for impersonated user.
	// +optional
	ImpersonateUserExtra map[string][]string `json:"as-user-extra,omitempty"`
	// `username` is the username for basic authentication to the kubernetes cluster.
	// +optional
	Username string `json:"username,omitempty"`
	// `password` is the password for basic authentication to the kubernetes cluster.
	// +optional
	Password string `json:"password,omitempty" datapolicy:"password"`
	// `auth-provider` specifies a custom authentication plugin for the kubernetes cluster.
	// +optional
	AuthProvider *AuthProviderConfig `json:"auth-provider,omitempty"`
	// `exec` specifies a custom exec-based authentication plugin for the kubernetes cluster.
	// +optional
	Exec *ExecConfig `json:"exec,omitempty"`
	// `extensions` holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields
	// +optional
	Extensions []NamedExtension `json:"extensions,omitempty"`
}

// Context is a tuple of references to a cluster (how do I communicate with a kubernetes cluster), a user (how do I identify myself), and a namespace (what subset of resources do I want to work with)
type Context struct {
	// `cluster` is the name of the cluster for this context.
	Cluster string `json:"cluster"`
	// `user` is the name of the authInfo for this context.
	AuthInfo string `json:"user"`
	// `namespace` is the default namespace to use on unspecified requests.
	// +optional
	Namespace string `json:"namespace,omitempty"`
	// `extensions` holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields.
	// +optional
	Extensions []NamedExtension `json:"extensions,omitempty"`
}

// NamedCluster relates nicknames to cluster information
type NamedCluster struct {
	// `name` is the nickname for this cluster.
	Name string `json:"name"`
	// `cluster` holds the cluster information.
	Cluster Cluster `json:"cluster"`
}

// NamedContext relates nicknames to context information
type NamedContext struct {
	// `name` is the nickname for this context.
	Name string `json:"name"`
	// `context` holds the context information.
	Context Context `json:"context"`
}

// NamedAuthInfo relates nicknames to auth information
type NamedAuthInfo struct {
	// `name` is the nickname for this AuthInfo.
	Name string `json:"name"`
	// `user` holds the auth information.
	AuthInfo AuthInfo `json:"user"`
}

// NamedExtension relates nicknames to extension information
type NamedExtension struct {
	// `name` is the nickname for this Extension.
	Name string `json:"name"`
	// `extension` holds the extension information.
	Extension runtime.RawExtension `json:"extension"`
}

// AuthProviderConfig holds the configuration for a specified auth provider.
type AuthProviderConfig struct {
	// `name` is the name of the authentication provider.
	Name string `json:"name"`
	// `config` contains the configuration for the specified auth provider.
	Config map[string]string `json:"config"`
}

// ExecConfig specifies a command to provide client credentials. The command is exec'd
// and outputs structured stdout holding credentials.
//
// See the client.authentication.k8s.io API group for specifications of the exact input
// and output format
type ExecConfig struct {
	// `command` is the command to execute.
	Command string `json:"command"`
	// `args` is the arguments to pass to the command when executing it.
	// +optional
	Args []string `json:"args"`
	// `env` defines additional environment variables to expose to the process. These
	// are unioned with the host's environment, as well as variables client-go uses
	// to pass argument to the plugin.
	// +optional
	Env []ExecEnvVar `json:"env"`

	// `apiVersion` is the preferred input version of the ExecInfo. The returned ExecCredentials MUST use
	// the same encoding version as the input.
	APIVersion string `json:"apiVersion,omitempty"`

	// `installHint` is shown to the user when the executable doesn't seem to be
	// present. For example, `brew install foo-cli` might be a good install hint for
	// `foo-cli` on Mac OS systems.
	InstallHint string `json:"installHint,omitempty"`

	// `rovideClusterInfo` determines whether or not to provide cluster information,
	// which could potentially contain very large CA data, to this exec plugin as a
	// part of the KUBERNETES_EXEC_INFO environment variable. By default, it is set
	// to false. Package k8s.io/client-go/tools/auth/exec provides helper methods for
	// reading this environment variable.
	ProvideClusterInfo bool `json:"provideClusterInfo"`

	// `nteractiveMode` determines this plugin's relationship with standard input. Valid
	// values are:
	// - "Never" - this exec plugin never uses standard input;
	// - "IfAvailable" - this exec plugin wants to use standard input if it is available;
	// - "Always" - this exec plugin requires standard input to function.
	// See ExecInteractiveMode values for more details.
	//
	// If `apiVersion is `client.authentication.k8s.io/v1alpha1` or
	// `client.authentication.k8s.io/v1beta1`, then this field is optional and defaults
	// to "IfAvailable" when unset. Otherwise, this field is required.
	//+optional
	InteractiveMode ExecInteractiveMode `json:"interactiveMode,omitempty"`
}

// ExecEnvVar is used for setting environment variables when executing an exec-based
// credential plugin.
type ExecEnvVar struct {
	// `name` is the name of the environment variable.
	Name string `json:"name"`
	// `value` contains the value of the environment variable.
	Value string `json:"value"`
}

// ExecInteractiveMode is a string that describes an exec plugin's relationship with standard input.
type ExecInteractiveMode string

const (
	// "Never" declares that this exec plugin never needs to use standard
	// input, and therefore the exec plugin will be run regardless of whether standard input is
	// available for user input.
	NeverExecInteractiveMode ExecInteractiveMode = "Never"
	// "IfAvailable" declares that this exec plugin would like to use standard input
	// if it is available, but can still operate if standard input is not available. Therefore, the
	// exec plugin will be run regardless of whether stdin is available for user input. If standard
	// input is available for user input, then it will be provided to this exec plugin.
	IfAvailableExecInteractiveMode ExecInteractiveMode = "IfAvailable"
	// "Always" declares that this exec plugin requires standard input in order to
	// run, and therefore the exec plugin will only be run if standard input is available for user
	// input. If standard input is not available for user input, then the exec plugin will not be run
	// and an error will be returned by the exec plugin runner.
	AlwaysExecInteractiveMode ExecInteractiveMode = "Always"
)
