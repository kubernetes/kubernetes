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

import "k8s.io/kubernetes/pkg/api/unversioned"

type MasterConfiguration struct {
	unversioned.TypeMeta

	Secrets           Secrets    `json:"secrets"`
	API               API        `json:"api"`
	Networking        Networking `json:"networking"`
	KubernetesVersion string     `json:"kubernetesVersion"`
	CloudProvider     string     `json:"cloudProvider"`
}

type API struct {
	AdvertiseAddresses []string `json:"advertiseAddresses"`
	ExternalDNSNames   []string `json:"externalDNSNames"`
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

type Secrets struct {
	GivenToken  string `json:"givenToken"`  // dot-separated `<TokenID>.<Token>` set by the user
	TokenID     string `json:"tokenID"`     // optional on master side, will be generated if not specified
	Token       []byte `json:"token"`       // optional on master side, will be generated if not specified
	BearerToken string `json:"bearerToken"` // set based on Token
}

type NodeConfiguration struct {
	unversioned.TypeMeta

	MasterAddresses []string `json:"masterAddresses"`
}

// ClusterInfo TODO add description
type ClusterInfo struct {
	unversioned.TypeMeta
	// TODO(phase1+) this may become simply `api.Config`
	CertificateAuthorities []string `json:"certificateAuthorities"`
	Endpoints              []string `json:"endpoints"`
}
