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
	"testing"

	"github.com/rackspace/gophercloud/openstack"
	"github.com/stretchr/testify/assert"
)

func TestMonascaURLDiscovery(t *testing.T) {
	// setup
	ksClient, err := NewKeystoneClient(testConfig)
	assert.NoError(t, err)

	// do
	monURL, err := ksClient.MonascaURL()

	// assert
	assert.NoError(t, err)
	assert.Equal(t, monURL.String(), monascaAPIStub.URL)
}

func TestGetTokenWhenMissing(t *testing.T) {
	// setup
	ksClient, err := NewKeystoneClient(testConfig)
	assert.NoError(t, err)

	// do
	token, err := ksClient.GetToken()

	// assert
	assert.NoError(t, err)
	assert.Equal(t, token, testToken)
}

func TestGetTokenWhenInvalid(t *testing.T) {
	// setup
	opts := testConfig.AuthOptions
	provider, err := openstack.AuthenticatedClient(opts)
	assert.NoError(t, err)
	client := openstack.NewIdentityV3(provider)
	ksClient := &KeystoneClientImpl{client: client, opts: opts, token: invalidToken, monascaURL: nil}

	// do
	token, err := ksClient.GetToken()

	// assert
	assert.NoError(t, err)
	assert.Equal(t, token, testToken)
}

func TestGetTokenWhenValid(t *testing.T) {
	// setup
	opts := testConfig.AuthOptions
	provider, err := openstack.AuthenticatedClient(opts)
	assert.NoError(t, err)
	client := openstack.NewIdentityV3(provider)
	ksClient := &KeystoneClientImpl{client: client, opts: opts, token: validToken, monascaURL: nil}

	// do
	token, err := ksClient.GetToken()

	// assert
	assert.NoError(t, err)
	assert.Equal(t, token, testToken)
}

func TestKeystoneClientReauthenticate(t *testing.T) {
	// setup
	opts := testConfig.AuthOptions
	provider, err := openstack.AuthenticatedClient(opts)
	assert.NoError(t, err)
	client := openstack.NewIdentityV3(provider)
	client.TokenID = "someinvalidtoken"
	client.ReauthFunc = func() error { return openstack.AuthenticateV3(client.ProviderClient, opts) }
	ksClient := &KeystoneClientImpl{client: client, opts: opts, token: validToken, monascaURL: nil}

	// do
	token, err := ksClient.GetToken()

	// assert
	assert.NoError(t, err)
	assert.Equal(t, token, testToken)
}
