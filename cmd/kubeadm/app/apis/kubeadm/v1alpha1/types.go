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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type MasterConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	API               API        `json:"api"`
	Discovery         Discovery  `json:"discovery"`
	Etcd              Etcd       `json:"etcd"`
	Networking        Networking `json:"networking"`
	KubernetesVersion string     `json:"kubernetesVersion"`
	CloudProvider     string     `json:"cloudProvider"`
	AuthorizationMode string     `json:"authorizationMode"`

	Phases Phases `json:"phases"`

	// The directory where certificates are stored
	CertificatesDir string `json:"certificatesDir"`
}

type PhaseMetadata struct {
	Annotations map[string]string `json:"annotations"`
}

type Phases struct {
	Certificates CertificatesPhase `json:"certificates"`
}

type CertificatesPhase struct {
	Metadata PhaseMetadata `json:"metadata"`

	// In the future, we may provide more options for generating certs
	// For instance, we may want to provide integrations with something like Vault in the future for storing the certs
	SelfSign *SelfSignCertificates `json:"selfSign"`
}

type SelfSignCertificates struct {
	// This phase needs to know these values as well:
	// ServiceSubnet, DNSDomain, CertificatesDir, AdvertiseAddresses

	// All IP addresses and DNS names these certs should be signed for
	// Defaults to the default networking interface's IP address and the hostname of the master node
	AltNames []string `json:"altNames"`
}

type API struct {
	AdvertiseAddresses []string `json:"advertiseAddresses"`
	Port               int32    `json:"port"`
}

type Discovery struct {
	HTTPS *HTTPSDiscovery `json:"https"`
	File  *FileDiscovery  `json:"file"`
	Token *TokenDiscovery `json:"token"`
}

type HTTPSDiscovery struct {
	URL string `json:"url"`
}

type FileDiscovery struct {
	Path string `json:"path"`
}

type TokenDiscovery struct {
	ID        string   `json:"id"`
	Secret    string   `json:"secret"`
	Addresses []string `json:"addresses"`
}

type Networking struct {
	ServiceSubnet string `json:"serviceSubnet"`
	PodSubnet     string `json:"podSubnet"`
	DNSDomain     string `json:"dnsDomain"`
}

type Etcd struct {
	Endpoints []string `json:"endpoints"`
	CAFile    string   `json:"caFile"`
	CertFile  string   `json:"certFile"`
	KeyFile   string   `json:"keyFile"`
}

type NodeConfiguration struct {
	metav1.TypeMeta `json:",inline"`

	Discovery Discovery `json:"discovery"`
}

// ClusterInfo TODO add description
type ClusterInfo struct {
	metav1.TypeMeta `json:",inline"`
	// TODO(phase1+) this may become simply `api.Config`
	CertificateAuthorities []string `json:"certificateAuthorities"`
	Endpoints              []string `json:"endpoints"`
}
