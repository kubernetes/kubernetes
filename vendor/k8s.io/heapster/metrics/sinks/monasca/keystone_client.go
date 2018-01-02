// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package monasca

import (
	"net/url"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/identity/v3/endpoints"
	"github.com/rackspace/gophercloud/openstack/identity/v3/services"
	"github.com/rackspace/gophercloud/openstack/identity/v3/tokens"
	"github.com/rackspace/gophercloud/pagination"
)

// KeystoneClient defines the interface of any client that can talk with Keystone.
type KeystoneClient interface {
	MonascaURL() (*url.URL, error)
	GetToken() (string, error)
}

// KeystoneClientImpl can authenticate with keystone and provide
// tokens, required for accessing the Monasca APIs.
type KeystoneClientImpl struct {
	client     *gophercloud.ServiceClient
	opts       gophercloud.AuthOptions
	token      *tokens.Token
	monascaURL *url.URL
}

// MonascaURL Discovers the monasca service API endpoint and returns it.
func (ksClient *KeystoneClientImpl) MonascaURL() (*url.URL, error) {
	if ksClient.monascaURL == nil {
		monascaURL, err := ksClient.serviceEndpoint("monitoring", gophercloud.AvailabilityPublic)
		if err != nil {
			return nil, err
		}
		ksClient.monascaURL = monascaURL
	}
	return ksClient.monascaURL, nil
}

// discovers a single service endpoint from a given service type
func (ksClient *KeystoneClientImpl) serviceEndpoint(serviceType string, availability gophercloud.Availability) (*url.URL, error) {
	serviceID, err := ksClient.serviceID(serviceType)
	if err != nil {
		return nil, err
	}
	return ksClient.endpointURL(serviceID, availability)
}

// finds the URL for a given service ID and availability
func (ksClient *KeystoneClientImpl) endpointURL(serviceID string, availability gophercloud.Availability) (*url.URL, error) {
	opts := endpoints.ListOpts{Availability: availability, ServiceID: serviceID, PerPage: 1, Page: 1}
	pager := endpoints.List(ksClient.client, opts)
	endpointURL := (*url.URL)(nil)
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		endpointList, err := endpoints.ExtractEndpoints(page)
		if err != nil {
			return false, err
		}
		for _, e := range endpointList {
			URL, err := url.Parse(e.URL)
			if err != nil {
				return false, err
			}
			endpointURL = URL
		}
		return false, nil
	})
	if err != nil {
		return nil, err
	}
	return endpointURL, nil
}

// returns the first found service ID from a given service type
func (ksClient *KeystoneClientImpl) serviceID(serviceType string) (string, error) {
	opts := services.ListOpts{ServiceType: serviceType, PerPage: 1, Page: 1}
	pager := services.List(ksClient.client, opts)
	serviceID := ""
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		serviceList, err := services.ExtractServices(page)
		if err != nil {
			return false, err
		}
		for _, s := range serviceList {
			serviceID = s.ID
		}
		return false, nil
	})
	if err != nil {
		return "", err
	}
	return serviceID, nil
}

// GetToken returns a valid X-Auth-Token.
func (ksClient *KeystoneClientImpl) GetToken() (string, error) {
	// generate if needed
	if ksClient.token == nil {
		return ksClient.newToken()
	}
	// validate
	valid, err := tokens.Validate(ksClient.client, ksClient.token.ID)
	if err != nil || !valid {
		return ksClient.newToken()
	}
	return ksClient.token.ID, nil
}

// generates a brand new Keystone token
func (ksClient *KeystoneClientImpl) newToken() (string, error) {
	token, err := tokens.Create(ksClient.client, ksClient.opts, nil).Extract()
	if err != nil {
		return "", err
	}
	ksClient.token = token
	return token.ID, nil
}

// NewKeystoneClient initilizes a keystone client with the provided configuration.
func NewKeystoneClient(config Config) (KeystoneClient, error) {
	opts := config.AuthOptions
	provider, err := openstack.AuthenticatedClient(opts)
	if err != nil {
		return nil, err
	}
	client := openstack.NewIdentityV3(provider)
	// build a closure for ksClient reauthentication
	client.ReauthFunc = func() error {
		return openstack.AuthenticateV3(client.ProviderClient, opts)
	}
	return &KeystoneClientImpl{client: client, opts: opts, token: nil, monascaURL: nil}, nil
}
