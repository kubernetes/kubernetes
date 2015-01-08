/*
Copyright 2014 Google Inc. All rights reserved.

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

package clientcmd

import ()

// Where possible, yaml tags match the cli argument names.
// Top level config objects and all values required for proper functioning are not "omitempty".  Any truly optional piece of config is allowed to be omitted.

// Config holds the information needed to build connect to remote kubernetes clusters as a given user
type Config struct {
	// Preferences holds general information to be use for cli interactions
	Preferences Preferences `yaml:"preferences"`
	// Clusters is a map of referencable names to cluster configs
	Clusters map[string]Cluster `yaml:"clusters"`
	// AuthInfos is a map of referencable names to user configs
	AuthInfos map[string]AuthInfo `yaml:"users"`
	// Contexts is a map of referencable names to context configs
	Contexts map[string]Context `yaml:"contexts"`
	// CurrentContext is the name of the context that you would like to use by default
	CurrentContext string `yaml:"current-context"`
}

type Preferences struct {
	Colors bool `yaml:"colors,omitempty"`
}

// Cluster contains information about how to communicate with a kubernetes cluster
type Cluster struct {
	// Server is the address of the kubernetes cluster (https://hostname:port).
	Server string `yaml:"server"`
	// APIVersion is the preferred api version for communicating with the kubernetes cluster (v1beta1, v1beta2, v1beta3, etc).
	APIVersion string `yaml:"api-version,omitempty"`
	// InsecureSkipTLSVerify skips the validity check for the server's certificate. This will make your HTTPS connections insecure.
	InsecureSkipTLSVerify bool `yaml:"insecure-skip-tls-verify,omitempty"`
	// CertificateAuthority is the path to a cert file for the certificate authority.
	CertificateAuthority string `yaml:"certificate-authority,omitempty"`
}

// AuthInfo contains information that describes identity information.  This is use to tell the kubernetes cluster who you are.
type AuthInfo struct {
	// AuthPath is the path to a kubernetes auth file (~/.kubernetes_auth).  If you provide an AuthPath, the other options specified are ignored
	AuthPath string `yaml:"auth-path,omitempty"`
	// ClientCertificate is the path to a client cert file for TLS.
	ClientCertificate string `yaml:"client-certificate,omitempty"`
	// ClientKey is the path to a client key file for TLS.
	ClientKey string `yaml:"client-key,omitempty"`
	// Token is the bearer token for authentication to the kubernetes cluster.
	Token string `yaml:"token,omitempty"`
}

// Context is a tuple of references to a cluster (how do I communicate with a kubernetes cluster), a user (how do I identify myself), and a namespace (what subset of resources do I want to work with)
type Context struct {
	// Cluster is the name of the cluster for this context
	Cluster string `yaml:"cluster"`
	// AuthInfo is the name of the authInfo for this context
	AuthInfo string `yaml:"user"`
	// Namespace is the default namespace to use on unspecified requests
	Namespace string `yaml:"namespace,omitempty"`
}

// NewConfig is a convenience function that returns a new Config object with non-nil maps
func NewConfig() *Config {
	return &Config{
		Clusters:  make(map[string]Cluster),
		AuthInfos: make(map[string]AuthInfo),
		Contexts:  make(map[string]Context),
	}
}
