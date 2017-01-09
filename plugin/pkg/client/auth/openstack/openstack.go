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

package openstack

import (
	"fmt"
	"net/http"

	"github.com/golang/glog"
	"github.com/rackspace/gophercloud/openstack"

	"k8s.io/kubernetes/pkg/client/restclient"
)

func init() {
	if err := restclient.RegisterAuthProviderPlugin("openstack", newOpenstackAuthProvider); err != nil {
		glog.Fatalf("Failed to register openstack auth plugin: %s", err)
	}
}

// openstackAuthProvider is an authprovider in case the environment is OpenStack.
// Generally, the components in OpenStack world use tokens to authenticate with
// each other, also the openstack client uses some envrioment variables to determine
// the user identity, we can achive the same experience, so the user only needs
// to source their openstack rc, and then we will generate the token here, and send
// to apiserver, it will also handles the token expiration automatically.
type openstackAuthProvider struct{}

// tokenRoundTripper implements the RoundTripper interface: adding the bearer token
// and refreshing token when it sees 401 the first time.
type tokenRoundTripper struct {
	http.RoundTripper
	token    string
	getToken func() (string, error)
	failed   bool
}

// refreshToken token by authenticate with openstack again
func refreshToken() (string, error) {
	options, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return "", fmt.Errorf("failed to read openstack env vars: %s", err)
	}
	client, err := openstack.AuthenticatedClient(options)
	if err != nil {
		return "", fmt.Errorf("authentication failed: %s", err)
	}
	return client.TokenID, nil
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header, len(r.Header))
	for k, s := range r.Header {
		r2.Header[k] = append([]string(nil), s...)
	}
	return r2
}

// RoundTrip adds the bearer token into the request, and rotate it automatically
func (t *tokenRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Set("Authorization", "Bearer "+t.token)
	resp, err := t.RoundTripper.RoundTrip(req)
	if err != nil || resp == nil {
		return resp, err
	}
	if resp.StatusCode != http.StatusUnauthorized || t.failed {
		t.failed = false
		return resp, err
	}
	t.failed = true
	// will need to refresh the token, and redo the request
	if t.token, err = t.getToken(); err != nil {
		return resp, err
	}
	newReq := cloneRequest(req)
	newReq.Header.Set("Authorization", "Bearer "+t.token)
	return t.RoundTripper.RoundTrip(newReq)
}

func newOpenstackAuthProvider(clusterAddress string, c3Config map[string]string, persister restclient.AuthProviderConfigPersister) (restclient.AuthProvider, error) {
	return &openstackAuthProvider{}, nil
}

func (c *openstackAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &tokenRoundTripper{RoundTripper: rt, getToken: refreshToken}
}

func (c *openstackAuthProvider) Login() error { return nil }
