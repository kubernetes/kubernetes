/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package api

import (
	"k8s.io/kubernetes/pkg/runtime"
)

// Where possible, json tags match the cli argument names.
// Top level config objects and all values required for proper functioning are not "omitempty".  Any truly optional piece of config is allowed to be omitted.

// Config holds the information needed to build connect to remote kubernetes clusters as a given user
// IMPORTANT if you add fields to this struct, please update IsConfigEmpty()
type Config struct {
	// Legacy field from pkg/api/types.go TypeMeta.
	// TODO(jlowdermilk): remove this after eliminating downstream dependencies.
	Kind string `json:"kind,omitempty"`
	// DEPRECATED: APIVersion is the preferred api version for communicating with the kubernetes cluster (v1, v2, etc).
	// Because a cluster can run multiple API groups and potentially multiple versions of each, it no longer makes sense to specify
	// a single value for the cluster version.
	// This field isn't really needed anyway, so we are deprecating it without replacement.
	// It will be ignored if it is present.
	APIVersion string `json:"apiVersion,omitempty"`
	// Preferences holds general information to be use for cli interactions
	Preferences Preferences `json:"preferences"`
	// Clusters is a map of referencable names to cluster configs
	Clusters map[string]*Cluster `json:"clusters"`
	// AuthInfos is a map of referencable names to user configs
	AuthInfos map[string]*AuthInfo `json:"users"`
	// Contexts is a map of referencable names to context configs
	Contexts map[string]*Context `json:"contexts"`
	// CurrentContext is the name of the context that you would like to use by default
	CurrentContext string `json:"current-context"`
	// Extensions holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields
	Extensions map[string]runtime.Object `json:"extensions,omitempty"`
}

// IMPORTANT if you add fields to this struct, please update IsConfigEmpty()
type Preferences struct {
	Colors bool `json:"colors,omitempty"`
	// Extensions holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields
	Extensions map[string]runtime.Object `json:"extensions,omitempty"`
}

// Cluster contains information about how to communicate with a kubernetes cluster
type Cluster struct {
	// LocationOfOrigin indicates where this object came from.  It is used for round tripping config post-merge, but never serialized.
	LocationOfOrigin string
	// Server is the address of the kubernetes cluster (https://hostname:port).
	Server string `json:"server"`
	// APIVersion is the preferred api version for communicating with the kubernetes cluster (v1, v2, etc).
	APIVersion string `json:"api-version,omitempty"`
	// InsecureSkipTLSVerify skips the validity check for the server's certificate. This will make your HTTPS connections insecure.
	InsecureSkipTLSVerify bool `json:"insecure-skip-tls-verify,omitempty"`
	// CertificateAuthority is the path to a cert file for the certificate authority.
	CertificateAuthority string `json:"certificate-authority,omitempty"`
	// CertificateAuthorityData contains PEM-encoded certificate authority certificates. Overrides CertificateAuthority
	CertificateAuthorityData []byte `json:"certificate-authority-data,omitempty"`
	// Extensions holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields
	Extensions map[string]runtime.Object `json:"extensions,omitempty"`
}

// AuthInfo contains information that describes identity information.  This is use to tell the kubernetes cluster who you are.
type AuthInfo struct {
	// LocationOfOrigin indicates where this object came from.  It is used for round tripping config post-merge, but never serialized.
	LocationOfOrigin string
	// ClientCertificate is the path to a client cert file for TLS.
	ClientCertificate string `json:"client-certificate,omitempty"`
	// ClientCertificateData contains PEM-encoded data from a client cert file for TLS. Overrides ClientCertificate
	ClientCertificateData []byte `json:"client-certificate-data,omitempty"`
	// ClientKey is the path to a client key file for TLS.
	ClientKey string `json:"client-key,omitempty"`
	// ClientKeyData contains PEM-encoded data from a client key file for TLS. Overrides ClientKey
	ClientKeyData []byte `json:"client-key-data,omitempty"`
	// Token is the bearer token for authentication to the kubernetes cluster.
	Token string `json:"token,omitempty"`
	// Impersonate is the username to act-as.
	Impersonate string `json:"act-as,omitempty"`
	// Username is the username for basic authentication to the kubernetes cluster.
	Username string `json:"username,omitempty"`
	// Password is the password for basic authentication to the kubernetes cluster.
	Password string `json:"password,omitempty"`
	// AuthProvider specifies a custom authentication plugin for the kubernetes cluster.
	AuthProvider *AuthProviderConfig `json:"auth-provider,omitempty"`
	// Extensions holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields
	Extensions map[string]runtime.Object `json:"extensions,omitempty"`
}

// Context is a tuple of references to a cluster (how do I communicate with a kubernetes cluster), a user (how do I identify myself), and a namespace (what subset of resources do I want to work with)
type Context struct {
	// LocationOfOrigin indicates where this object came from.  It is used for round tripping config post-merge, but never serialized.
	LocationOfOrigin string
	// Cluster is the name of the cluster for this context
	Cluster string `json:"cluster"`
	// AuthInfo is the name of the authInfo for this context
	AuthInfo string `json:"user"`
	// Namespace is the default namespace to use on unspecified requests
	Namespace string `json:"namespace,omitempty"`
	// Extensions holds additional information. This is useful for extenders so that reads and writes don't clobber unknown fields
	Extensions map[string]runtime.Object `json:"extensions,omitempty"`
}

// AuthProviderConfig holds the configuration for a specified auth provider.
type AuthProviderConfig struct {
	Name   string            `json:"name"`
	Config map[string]string `json:"config,omitempty"`
}

// NewConfig is a convenience function that returns a new Config object with non-nil maps
func NewConfig() *Config {
	return &Config{
		Preferences: *NewPreferences(),
		Clusters:    make(map[string]*Cluster),
		AuthInfos:   make(map[string]*AuthInfo),
		Contexts:    make(map[string]*Context),
		Extensions:  make(map[string]runtime.Object),
	}
}

// NewConfig is a convenience function that returns a new Config object with non-nil maps
func NewContext() *Context {
	return &Context{Extensions: make(map[string]runtime.Object)}
}

// NewConfig is a convenience function that returns a new Config object with non-nil maps
func NewCluster() *Cluster {
	return &Cluster{Extensions: make(map[string]runtime.Object)}
}

// NewConfig is a convenience function that returns a new Config object with non-nil maps
func NewAuthInfo() *AuthInfo {
	return &AuthInfo{Extensions: make(map[string]runtime.Object)}
}

// NewConfig is a convenience function that returns a new Config object with non-nil maps
func NewPreferences() *Preferences {
	return &Preferences{Extensions: make(map[string]runtime.Object)}
}
