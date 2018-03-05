/*
Copyright 2018 The Kubernetes Authors.

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

package openstack

import (
	"fmt"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	tokens2 "github.com/gophercloud/gophercloud/openstack/identity/v2/tokens"
	tokens3 "github.com/gophercloud/gophercloud/openstack/identity/v3/tokens"
	"github.com/gophercloud/gophercloud/openstack/utils"
)

const expiresAtDelta = 10 * time.Second

type openstackToken struct {
	ID        string
	ExpiresAt time.Time
}

func (o *openstackToken) Valid() bool {
	return o != nil && o.ID != "" && !o.expired()
}

func (o *openstackToken) expired() bool {
	if o.ExpiresAt.IsZero() {
		return false
	}
	return o.ExpiresAt.Add(-expiresAtDelta).Before(time.Now())
}

func tokenFromAuth(options gophercloud.AuthOptions) (*openstackToken, error) {
	client, err := openstack.NewClient(options.IdentityEndpoint)
	if err != nil {
		return nil, fmt.Errorf("error creating openstack client: %s", err)
	}

	versions := []*utils.Version{
		{ID: "v2.0", Priority: 20, Suffix: "/v2.0/"},
		{ID: "v3", Priority: 30, Suffix: "/v3/"},
	}

	chosen, endpoint, err := utils.ChooseVersion(client, versions)
	if err != nil {
		return nil, fmt.Errorf("error choosing openstack identity API version: %s", err)
	}

	var token *openstackToken
	switch chosen.ID {
	case "v2.0":
		token, err = v2auth(client, endpoint, options, gophercloud.EndpointOpts{})
	case "v3":
		token, err = v3auth(client, endpoint, &options, gophercloud.EndpointOpts{})
	default:
		return nil, fmt.Errorf("unrecognized openstack identity API version: %s", chosen.ID)
	}
	if err != nil {
		return nil, fmt.Errorf("error authenticating with openstack: %s", err)
	}
	return &openstackToken{token.ID, token.ExpiresAt}, nil
}

func v2auth(client *gophercloud.ProviderClient, endpoint string, options gophercloud.AuthOptions, eo gophercloud.EndpointOpts) (*openstackToken, error) {
	v2Client, err := openstack.NewIdentityV2(client, eo)
	if err != nil {
		return nil, err
	}

	if endpoint != "" {
		v2Client.Endpoint = endpoint
	}

	result := tokens2.Create(v2Client, options)
	v2token, err := result.ExtractToken()
	if err != nil {
		return nil, fmt.Errorf("error parsing openstack v2 token: %s", err)
	}
	return &openstackToken{v2token.ID, v2token.ExpiresAt}, nil
}

func v3auth(client *gophercloud.ProviderClient, endpoint string, options tokens3.AuthOptionsBuilder, eo gophercloud.EndpointOpts) (*openstackToken, error) {
	v3Client, err := openstack.NewIdentityV3(client, eo)
	if err != nil {
		return nil, err
	}

	if endpoint != "" {
		v3Client.Endpoint = endpoint
	}

	result := tokens3.Create(v3Client, options)
	v3token, err := result.ExtractToken()
	if err != nil {
		return nil, fmt.Errorf("error parsing openstack v3 token: %s", err)
	}
	return &openstackToken{v3token.ID, v3token.ExpiresAt}, nil
}
