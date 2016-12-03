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
	metav1 "k8s.io/client-go/pkg/apis/meta/v1"
)

type EnvParams struct {
	KubernetesDir     string
	HostPKIPath       string
	HostEtcdPath      string
	HyperkubeImage    string
	RepositoryPrefix  string
	DiscoveryImage    string
	EtcdImage         string
	ComponentLoglevel string
}

type MasterConfiguration struct {
	metav1.TypeMeta

	Secrets           Secrets
	API               API
	Discovery         Discovery
	Etcd              Etcd
	Networking        Networking
	KubernetesVersion string
	CloudProvider     string
}

type API struct {
	AdvertiseAddresses []string
	ExternalDNSNames   []string
	BindPort           int32
}

type Discovery struct {
	BindPort int32
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

type Secrets struct {
	GivenToken  string // dot-separated `<TokenID>.<Token>` set by the user
	TokenID     string // optional on master side, will be generated if not specified
	Token       []byte // optional on master side, will be generated if not specified
	BearerToken string // set based on Token
}

type NodeConfiguration struct {
	metav1.TypeMeta

	MasterAddresses []string
	Secrets         Secrets
	APIPort         int32
	DiscoveryPort   int32
}

// ClusterInfo TODO add description
type ClusterInfo struct {
	metav1.TypeMeta
	// TODO(phase1+) this may become simply `api.Config`
	CertificateAuthorities []string `json:"certificateAuthorities"`
	Endpoints              []string `json:"endpoints"`
}
