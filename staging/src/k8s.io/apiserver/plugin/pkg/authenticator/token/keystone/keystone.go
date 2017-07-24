/*
Copyright 2017 The Kubernetes Authors.

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

package keystone

import (
	"crypto/tls"
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/golang/glog"
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"
	"github.com/gophercloud/gophercloud/openstack/utils"

	"fmt"
	netutil "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/authentication/user"
	certutil "k8s.io/client-go/util/cert"
)

// KeystoneAuthenticator contacts openstack keystone to validate user's token passed in the request.
// The keystone endpoint is passed during apiserver startup
type KeystoneAuthenticator struct {
	authURL string
	client  *gophercloud.ServiceClient
}

// AuthenticatePassword checks the token via Keystone call
func (keystoneAuthenticator *KeystoneAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {

	// We can use the Keystone GET /v3/auth/tokens API to validate the token
	// and get information about the user as well
	// http://git.openstack.org/cgit/openstack/keystone/tree/api-ref/source/v3/authenticate-v3.inc#n437
	// https://developer.openstack.org/api-ref/identity/v3/?expanded=validate-and-show-information-for-token-detail
	request_opts := gophercloud.RequestOpts{
		MoreHeaders: map[string]string{
			"X-Auth-Token":    token,
			"X-Subject-Token": token,
		},
	}
	url := keystoneAuthenticator.client.ServiceURL("auth", "tokens")
	response, err := keystoneAuthenticator.client.Request("GET", url, &request_opts)
	if err != nil {
		glog.V(4).Info("Failed: bad response from API call: %v", err)
		return nil, false, errors.New("Failed to authenticate")
	}

	defer response.Body.Close()
	bodyBytes, err := ioutil.ReadAll(response.Body)
	if err != nil {
		glog.V(4).Infof("Cannot get HTTP response body from keystone token validate: %v", err)
		return nil, false, errors.New("Failed to authenticate")
	}

	// Read the response and parse the name and id for the user
	var data map[string]interface{}
	err = json.Unmarshal(bodyBytes, &data)
	if err != nil {
		glog.V(4).Infof("Cannot unmarshal response: %v", err)
		return nil, false, errors.New("Failed to authenticate")
	}

	token_info := data["token"].(map[string]interface{})
	project_info := token_info["project"].(map[string]interface{})
	user_info := &user.DefaultInfo{
		Name: project_info["name"].(string),
		UID:  project_info["id"].(string),
	}

	return user_info, true, nil
}

// Construct a Keystone v3 client, bail out if we cannot find the v3 API endpoint
func keystoneV3Client(options gophercloud.AuthOptions, transport http.RoundTripper) (*gophercloud.ProviderClient, error) {
	client, err := openstack.NewClient(options.IdentityEndpoint)
	if err != nil {
		return nil, err
	}

	if transport != nil {
		client.HTTPClient.Transport = transport
	}

	versions := []*utils.Version{
		{ID: "v3.0", Priority: 30, Suffix: "/v3/"},
	}
	chosen, _, err := utils.ChooseVersion(client, versions)
	if err != nil {
		return nil, fmt.Errorf("Unable to find identity API v3 version : %v", err)
	}

	switch chosen.ID {
	case "v3.0":
		return client, nil
	default:
		// The switch statement must be out of date from the versions list.
		return nil, fmt.Errorf("Unsupported identity API version: %s", chosen.ID)
	}
}

// NewKeystoneAuthenticator returns a password authenticator that validates credentials using openstack keystone
func NewKeystoneAuthenticator(authURL string, caFile string) (*KeystoneAuthenticator, error) {
	if !strings.HasPrefix(authURL, "https") {
		return nil, errors.New("Auth URL should be secure and start with https")
	}
	var transport http.RoundTripper
	if authURL == "" {
		return nil, errors.New("Auth URL is empty")
	}
	if caFile != "" {
		roots, err := certutil.NewPool(caFile)
		if err != nil {
			return nil, err
		}
		config := &tls.Config{}
		config.RootCAs = roots
		transport = netutil.SetOldTransportDefaults(&http.Transport{TLSClientConfig: config})
	}
	opts := gophercloud.AuthOptions{IdentityEndpoint: authURL}
	provider, err := keystoneV3Client(opts, transport)
	if err != nil {
		return nil, err
	}

	// We should use the V3 API
	client, err := openstack.NewIdentityV3(provider, gophercloud.EndpointOpts{})
	if err != nil {
		glog.V(4).Info("Failed: Unable to use keystone v3 identity service: %v", err)
		return nil, errors.New("Failed to authenticate")
	}
	if err != nil {
		glog.V(4).Info("Failed: Starting openstack authenticate client: %v", err)
		return nil, errors.New("Failed to authenticate")
	}

	// Make sure we look under /v3 for resources
	client.IdentityBase = client.IdentityEndpoint
	client.Endpoint = client.IdentityEndpoint


	return &KeystoneAuthenticator{authURL: authURL, client: client}, nil
}
