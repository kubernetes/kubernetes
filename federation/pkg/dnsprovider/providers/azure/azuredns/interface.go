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

package azuredns

import (
	"github.com/Azure/azure-sdk-for-go/arm/dns"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/golang/glog"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	azurestub "k8s.io/kubernetes/federation/pkg/dnsprovider/providers/azure/azuredns/stubs"
)

// API is an interface abstracting the Azure DNS clients from azure-sdk-for-go behind a single interface
// The mock implementation is below
// The interface is defined in stubs/azurednsapi.go

// Compile time check for interface adherence
var _ dnsprovider.Interface = Interface{}

// Interface is the abstraction layer to allow for mocking
type Interface struct {
	service azurestub.API
}

// Zones initializes a new Zones interface, which is the root
// of the DNS hierarchy
func (c Interface) Zones() (dnsprovider.Zones, bool) {
	return Zones{&c}, true
}

// New initializes a new API interface from the --dns-provider-config
// The --dns-provider-config option is required.
// In the future, we could try inferring defaults.
func New(config Config) *Interface {
	api := &DNSAPI{}

	glog.V(4).Infof("azuredns: Created Azure DNS DNSAPI for subscription: %s", config.Global.SubscriptionID)
	api.conf = config

	api.zc = dns.NewZonesClient(config.Global.SubscriptionID)
	spt, err := NewServicePrincipalTokenFromCredentials(config, azure.PublicCloud.ResourceManagerEndpoint)
	if err != nil {
		glog.Fatalf("azuredns: Error authenticating to Azure DNS: %v", err)
		return nil
	}

	api.zc.Authorizer = autorest.NewBearerAuthorizer(spt)

	api.rc = dns.NewRecordSetsClient(config.Global.SubscriptionID)
	spt, err = NewServicePrincipalTokenFromCredentials(config, azure.PublicCloud.ResourceManagerEndpoint)
	if err != nil {
		glog.Fatalf("azuredns: Error authenticating to Azure DNS: %v", err)
		return nil
	}

	api.rc.Authorizer = autorest.NewBearerAuthorizer(spt)
	return &Interface{service: api}
}
