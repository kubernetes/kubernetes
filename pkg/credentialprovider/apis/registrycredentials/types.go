/*
Copyright 2020 The Kubernetes Authors.

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

package registrycredentials

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

type RegistryCredentialConfig struct {
	metav1.TypeMeta `json:",inline"`

	Providers []RegistryCredentialProvider `json:"providers"`
}

// RegistryCredentialProvider is used by the kubelet container runtime to match the
// image property string (from the container spec) with exec-based credential provider
// plugins that provide container registry credentials.
type RegistryCredentialProvider struct {

	// ImageMatchers is a list of strings used to match against the image property
	// (sometimes called "registry path") to determine which images to provide
	// credentials for.  If one of the strings matches the image property, then the
	// RegistryCredentialProvider will be used by kubelet to provide credentials
	// for the image pull.

	// The image property of a container supports the same syntax as the docker
	// command does, including private registries and tags. A registry path is
	// similar to a URL, but does not contain a protocol specifier (https://).
	//
	// Each ImageMatcher string is a pattern which can optionally contain
	// a port and a path, similar to the image spec.  Globs can be used in the
	// hostname (but not the port or the path).
	//
	// Globs are supported as subdomains (*.k8s.io) or (k8s.*.io), and
	// top-level-domains (k8s.*).  Matching partial subdomains is also supported
	// (app*.k8s.io).  Each glob can only match a single subdomain segment, so
	// *.io does not match *.k8s.io.
	//
	// The image property matches when it has the same number of parts as the
	// ImageMatcher string, and each part matches.  Additionally the path of
	// ImageMatcher must be a prefix of the target URL. If the ImageMatcher
	// contains a port, then the port must match as well.
	ImageMatchers []string `json:"imageMatchers"`

	// Exec specifies a custom exec-based plugin.  This type is defined in
	Exec ExecConfig `json:"exec"`
}

// ExecConfig specifies a command to provide client credentials. The command is exec'd
// and outputs structured stdout holding credentials.
//
// See the client.authentiction.k8s.io API group for specifications of the exact input
// and output format
type ExecConfig struct {
	// Command to execute.
	Command string `json:"command"`
	// Arguments to pass to the command when executing it.
	// +optional
	Args []string `json:"args,omitempty"`
	// Env defines additional environment variables to expose to the process.
	// +optional
	Env []ExecEnvVar `json:"env,omitempty"`
}

// ExecEnvVar is used for setting environment variables when executing an exec-based
// credential plugin.
type ExecEnvVar struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RegistryCredentialPluginRequest is passed to the plugin via stdin, and includes the image that will be pulled by kubelet.
type RegistryCredentialPluginRequest struct {
	metav1.TypeMeta `json:",inline"`
	// Image is used when passed to registry credential providers as part of an
	// image pull
	Image string `json:"image"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RegistryCredentialPluginResponse holds credentials for the kubelet runtime
// to use for image pulls.  It is returned from the plugin as stdout.
type RegistryCredentialPluginResponse struct {
	metav1.TypeMeta `json:",inline"`

	// +optional
	Username *string `json:"username,omitempty"`
	// +optional
	Password *string `json:"password,omitempty"`
}
