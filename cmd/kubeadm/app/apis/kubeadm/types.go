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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type EnvParams struct {
	KubernetesDir    string
	HyperkubeImage   string
	RepositoryPrefix string
	EtcdImage        string
}

type MasterConfiguration struct {
	metav1.TypeMeta

	API                API
	Etcd               Etcd
	Networking         Networking
	KubernetesVersion  string
	CloudProvider      string
	AuthorizationModes []string

	Token    string
	TokenTTL time.Duration

	// SelfHosted enables an alpha deployment type where the apiserver, scheduler, and
	// controller manager are managed by Kubernetes itself. This option is likely to
	// become the default in the future.
	SelfHosted bool

	APIServerExtraArgs         map[string]string
	ControllerManagerExtraArgs map[string]string
	SchedulerExtraArgs         map[string]string

	// APIServerCertSANs sets extra Subject Alternative Names for the API Server signing cert
	APIServerCertSANs []string
	// CertificatesDir specifies where to store or look for all required certificates
	CertificatesDir string
}

type API struct {
	// AdvertiseAddress sets the address for the API server to advertise.
	AdvertiseAddress string
	// BindPort sets the secure port for the API Server to bind to
	BindPort int32
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
	DataDir   string
	ExtraArgs map[string]string
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
