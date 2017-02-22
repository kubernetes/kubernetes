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
	// SelfHosted enables an alpha deployment type where the apiserver, scheduler, and
	// controller manager are managed by Kubernetes itself. This option is likely to
	// become the default in the future.
	SelfHosted bool

	APIServerExtraArgs         map[string]string
	ControllerManagerExtraArgs map[string]string
	SchedulerExtraArgs         map[string]string
}

type API struct {
	AdvertiseAddresses []string
	ExternalDNSNames   []string
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

	CACertPath     string
	DiscoveryFile  string
	DiscoveryToken string
	// Currently we only pay attention to one api server but hope to support >1 in the future
	DiscoveryTokenAPIServers []string
	TLSBootstrapToken        string
	Token                    string
}

// ClusterInfo TODO add description
type ClusterInfo struct {
	metav1.TypeMeta
	// TODO(phase1+) this may become simply `api.Config`
	CertificateAuthorities []string
	Endpoints              []string
}
