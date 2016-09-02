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

package kubeadmapi

type BootstrapParams struct {
	// TODO this is mostly out of date and bloated now, let's revisit this soon
	Discovery *OutOfBandDiscovery
	EnvParams map[string]string
}

type OutOfBandDiscovery struct {
	// 'join-node' side
	ApiServerURLs string // comma separated
	CaCertFile    string
	GivenToken    string // dot-separated `<TokenID>.<Token>` set by the user
	TokenID       string // optional on master side, will be generated if not specified
	Token         []byte // optional on master side, will be generated if not specified
	BearerToken   string // set based on Token
	// 'init-master' side
	ApiServerDNSName string // optional, used in master bootstrap
	ListenIP         string // optional IP for master to listen on, rather than autodetect
}

type ClusterInfo struct {
	// TODO Kind, apiVersion
	// TODO clusterId, fetchedTime, expiredTime
	CertificateAuthorities []string `json:"certificateAuthorities"`
	Endpoints              []string `json:"endpoints"`
}
