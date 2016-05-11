/*
Copyright 2014 The Kubernetes Authors All rights reserved.
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

package clouddns

import (
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns/internal/interfaces"
)

var _ dnsprovider.Interface = Interface{}

type Interface struct {
	project_ string
	service  interfaces.Service
}

/*
// NewInterface initializes the underlying Google Cloud DNS library
// and wraps it in a provider-independent interface, which it returns.
func NewInterface(project string, tokenSource oauth2.TokenSource) dnsprovider.Interface {
	client := oauth2.NewClient(oauth2.NoContext, tokenSource)
	if client != nil {
		return client, nil
	} else {
		return nil, fmt.Errorf("Failed to get OAuth client.  No details provided.")
	}
	service, err := dns.New(oauthClient)
	if err != nil {
		glog.Errorf("Failed to get Cloud DNS client: %v", err)
	}
	glog.Infof("Successfully got DNS service: %v\n", service)
	return newInterfaceWithStub(project, internal.NewService(service))
}
*/

// newInterfaceWithStub facilitates stubbing out the underlying Google Cloud DNS
// library for testing purposes.  It returns an provider-independent interface.
func newInterfaceWithStub(project string, service interfaces.Service) *Interface {
	return &Interface{project, service}
}

/*
func getOauthClient() (*http.Client, error) {
	tokenSource, err := google.DefaultTokenSource(
		oauth2.NoContext,
		compute.CloudPlatformScope,
		compute.ComputeScope)
	if err != nil {
		return nil, err
	}
	client := oauth2.NewClient(oauth2.NoContext, tokenSource)
	if client != nil {
		return client, nil
	} else {
		return nil, fmt.Errorf("Failed to get OAuth client.  No details provided.")
	}
}
*/

// Zones returns the provider's Zones interface, or false if not supported.
func (i Interface) Zones() (zones dnsprovider.Zones, supported bool) {
	return Zones{i.service.ManagedZones(), &i}, true
}

func (i Interface) project() string {
	return i.project_
}
