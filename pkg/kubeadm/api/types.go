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

package api

import (
	"net"
)

type BootstrapParams struct {
	// TODO(phase1+) this is mostly out of date and bloated now, let's revisit this soon
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
	// TODO(phase1+) this is not really the address anything is going to listen on, may we should
	// call it `--advertise-ip-address` or something like that, or simply --use-address`,
	// or actually we should probably probably provide user to a way of saying that they
	// prefer public network or private with an option to pass the interface name.
	// we should also discuss if we could just pick private interface by default, which
	// is very often `eth1` and also make API server bind to that with 0.0.0.0 being
	// optional. DNS name is another one we don't really have, of course, and we should
	// have some flexibility there. All of this should reflect API severver flags etc.
	// Also kubelets care about hostnames and IPs, which we should present in a sane way
	// through one or two flags.
	ListenIP          net.IP // optional IP for master to listen on, rather than autodetect
	UseHyperkubeImage bool
}

type ClusterInfo struct {
	// TODO(pahse1?) this may become simply `api.Config`
	CertificateAuthorities []string `json:"certificateAuthorities"`
	Endpoints              []string `json:"endpoints"`
}
