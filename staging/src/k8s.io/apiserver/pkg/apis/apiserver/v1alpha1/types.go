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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// AdmissionConfiguration provides versioned configuration for admission controllers.
type AdmissionConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// Plugins allows specifying a configuration per admission control plugin.
	// +optional
	Plugins []AdmissionPluginConfiguration `json:"plugins"`
}

// AdmissionPluginConfiguration provides the configuration for a single plug-in.
type AdmissionPluginConfiguration struct {
	// Name is the name of the admission controller.
	// It must match the registered admission plugin name.
	Name string `json:"name"`

	// Path is the path to a configuration file that contains the plugin's
	// configuration
	// +optional
	Path string `json:"path"`

	// Configuration is an embedded configuration object to be used as the plugin's
	// configuration. If present, it will be used instead of the path to the configuration file.
	// +optional
	Configuration *runtime.Unknown `json:"configuration"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ConnectivityServiceConfiguration provides versioned configuration for network proxy clients.
type ConnectivityServiceConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	// ConnectionServices contains a list of network proxy client configurations
	ConnectionServices []ConnectionService `json:"connectionServices"`
}

// ConnectionService provides the configuration for a single network proxy client.
type ConnectionService struct {
	// Name is the name of the connectivity service.
	// Currently supported values are "Master" and "Cluster"
	Name string `json:"name"`

	// Connection is the exact information used to configure the network proxy
	Connection Connection `json:"connection"`
}

// ConnectivityServiceConfiguration provides the configuration for a single network proxy client.
type Connection struct {
	// Type is the type of connection used to connect from client to network/proxy-server.
	// Currently supported values are "http-connect" and "direct".
	Type string `json:"type"`

	// URL is the location of the proxy server to connect to.
	// As an example it might be "https://127.0.0.1:8131"
	// +optional
	URL string `json:"url,omitempty"`

	// CABundle is the file location of the CA to be used to determine trust with the network-proxy.
	// +optional
	CABundle string `json:"caBundle,omitempty"`

	// ClientKeyFile is the file location of the client key to be used in mtls handshakes with the network-proxy.
	// +optional
	ClientKeyFile string `json:"clientKeyFile,omitempty"`

	// ClientCertFile is the file location of the client certificate to be used in mtls handshakes with the network-proxy.
	// +optional
	ClientCertFile string `json:"clientCertFile,omitempty"`
}
