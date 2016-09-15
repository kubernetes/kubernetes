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

type KubeadmConfig struct {
	InitFlags
	JoinFlags
	ManualFlags
	Secrets struct {
		GivenToken  string // dot-separated `<TokenID>.<Token>` set by the user
		TokenID     string // optional on master side, will be generated if not specified
		Token       []byte // optional on master side, will be generated if not specified
		BearerToken string // set based on Token
	}
	EnvParams map[string]string // TODO(phase2) this is likely to be come componentconfig
}

// TODO(phase2) should we add validatin funcs on these structs?

type InitFlags struct {
	API struct {
		AdvertiseAddrs  []net.IP
		ExternalDNSName []string
	}
	Services struct {
		CIDR      net.IPNet
		DNSDomain string
	}
	CloudProvider string
	Schedulable bool
}

const (
	DefaultServiceDNSDomain   = "cluster.local"
	DefaultServicesCIDRString = "100.64.0.0/12"
)

var (
	DefaultServicesCIDR  *net.IPNet
	ListOfCloudProviders = []string{
		"aws",
		"azure",
		"cloudstack",
		"gce",
		"mesos",
		"openstack",
		"ovirt",
		"rackspace",
		"vsphere",
	}
	SupportedCloudProviders map[string]bool
)

func init() {
	_, DefaultServicesCIDR, _ = net.ParseCIDR(DefaultServicesCIDRString)
	SupportedCloudProviders = make(map[string]bool, len(ListOfCloudProviders))
	for _, v := range ListOfCloudProviders {
		SupportedCloudProviders[v] = true
	}
}

type JoinFlags struct {
	MasterAddrs []net.IP
}

// TODO(phase1?) we haven't decided whether manual sub commands should get merged into  main commands...
type ManualFlags struct {
	ApiServerURLs string // comma separated
	CaCertFile    string
	BearerToken   string // set based on Token
	ListenIP      net.IP // optional IP for master to listen on, rather than autodetect
}

type ClusterInfo struct {
	// TODO(pahse1?) this may become simply `api.Config`
	CertificateAuthorities []string `json:"certificateAuthorities"`
	Endpoints              []string `json:"endpoints"`
}
