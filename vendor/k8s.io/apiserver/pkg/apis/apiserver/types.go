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

package apiserver

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AdmissionConfiguration provides versioned configuration for admission controllers.
type AdmissionConfiguration struct {
	metav1.TypeMeta

	// Plugins allows specifying a configuration per admission control plugin.
	// +optional
	Plugins []AdmissionPluginConfiguration
}

// AdmissionPluginConfiguration provides the configuration for a single plug-in.
type AdmissionPluginConfiguration struct {
	// Name is the name of the admission controller.
	// It must match the registered admission plugin name.
	Name string

	// Path is the path to a configuration file that contains the plugin's
	// configuration
	// +optional
	Path string

	// Configuration is an embedded configuration object to be used as the plugin's
	// configuration. If present, it will be used instead of the path to the configuration file.
	// +optional
	Configuration *runtime.Unknown
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EgressSelectorConfiguration provides versioned configuration for egress selector clients.
type EgressSelectorConfiguration struct {
	metav1.TypeMeta

	// EgressSelections contains a list of egress selection client configurations
	EgressSelections []EgressSelection
}

// EgressSelection provides the configuration for a single egress selection client.
type EgressSelection struct {
	// Name is the name of the egress selection.
	// Currently supported values are "Master", "Etcd" and "Cluster"
	Name string

	// Connection is the exact information used to configure the egress selection
	Connection Connection
}

// Connection provides the configuration for a single egress selection client.
type Connection struct {
	// Type is the type of connection used to connect from client to konnectivity server.
	// Currently supported values are "http-connect" and "direct".
	Type string

	// httpConnect is the config needed to use http-connect to the konnectivity server.
	// +optional
	HTTPConnect *HTTPConnectConfig
}

type HTTPConnectConfig struct {
	// URL is the location of the konnectivity server to connect to.
	// As an example it might be "https://127.0.0.1:8131"
	URL string

	// CABundle is the file location of the CA to be used to determine trust with the konnectivity server.
	// +optional
	CABundle string

	// ClientKey is the file location of the client key to be used in mtls handshakes with the konnectivity server.
	// +optional
	ClientKey string

	// ClientCert is the file location of the client certificate to be used in mtls handshakes with the konnectivity server.
	// +optional
	ClientCert string
}
