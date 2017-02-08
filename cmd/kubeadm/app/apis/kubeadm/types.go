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

package kubeadm

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type EnvParams struct {
	KubernetesDir    string
	HostPKIPath      string
	HostEtcdPath     string
	HyperkubeImage   string
	RepositoryPrefix string
	DiscoveryImage   string
	EtcdImage        string
}

type MasterConfiguration struct {
	metav1.TypeMeta

	API               API
	Discovery         Discovery
	Etcd              Etcd
	Networking        Networking
	KubernetesVersion string
	CloudProvider     string
	AuthorizationMode string

	Phases Phases

	// The directory where certificates are stored
	CertificatesDir string
}

type PhaseMetadata struct {
	Annotations map[string]string
}

type Phases struct {
	Certificates CertificatesPhase
}

type CertificatesPhase struct {
	Metadata PhaseMetadata

	// In the future, we may provide more options for generating certs
	// For instance, we may want to provide integrations with something like Vault in the future for storing the certs
	SelfSign *SelfSignCertificates
}

type SelfSignCertificates struct {
	// This phase needs to know these values as well:
	// ServiceSubnet, DNSDomain, CertificatesDir, AdvertiseAddresses

	// All IP addresses and DNS names these certs should be signed for
	// Defaults to the default networking interface's IP address and the hostname of the master node
	AltNames []string
}

type API struct {
	AdvertiseAddresses []string
	Port               int32
}

type Discovery struct {
	HTTPS *HTTPSDiscovery
	File  *FileDiscovery
	Token *TokenDiscovery
}

type HTTPSDiscovery struct {
	URL string
}

type FileDiscovery struct {
	Path string
}

type TokenDiscovery struct {
	ID        string
	Secret    string
	Addresses []string
}

type Networking struct {
	ServiceSubnet string
	PodSubnet     string
	DNSDomain     string
}

type Etcd struct {
	Endpoints []string
	CAFile    string
	CertFile  string
	KeyFile   string
}

type NodeConfiguration struct {
	metav1.TypeMeta

	Discovery Discovery
}

// ClusterInfo TODO add description
type ClusterInfo struct {
	metav1.TypeMeta
	// TODO(phase1+) this may become simply `api.Config`
	CertificateAuthorities []string
	Endpoints              []string
}
