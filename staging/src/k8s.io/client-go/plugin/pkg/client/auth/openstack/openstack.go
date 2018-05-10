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
	"net/http"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack"

	"k8s.io/apimachinery/pkg/util/net"
	restclient "k8s.io/client-go/rest"
)

func init() {
	if err := restclient.RegisterAuthProviderPlugin("openstack", newOpenstackAuthProvider); err != nil {
		glog.Fatalf("Failed to register openstack auth plugin: %s", err)
	}
}

// openstackAuthProvider is an authprovider for openstack. this provider reads
// the environment variables to determine the client identity, and generates a
// token which will be inserted into the request header later.
type openstackAuthProvider struct {
	tokenGetter TokenGetter
	persister   restclient.AuthProviderConfigPersister
}

// newOpenstackAuthProvider creates an auth provider which works with openstack
// environment.
func newOpenstackAuthProvider(clusterAddress string, oapConfig map[string]string, persister restclient.AuthProviderConfigPersister) (restclient.AuthProvider, error) {
	options, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, fmt.Errorf("error reading openstack env vars: %s", err)
	}
	tg := &tokenGetter{options}
	ctg := newCachedGetter(tg, persister, oapConfig)
	return &openstackAuthProvider{
		tokenGetter: ctg,
		persister:   persister,
	}, nil
}

// Login is not used
func (oap *openstackAuthProvider) Login() error { return nil }

// TokenGetter returns a bearer token that can be inserted into request.
type TokenGetter interface {
	Token() (*openstackToken, error)
}

type tokenGetter struct {
	options gophercloud.AuthOptions
}

// Token gets a token from the OpenStack Identity service
func (tg *tokenGetter) Token() (*openstackToken, error) {
	token, err := tokenFromAuth(tg.options)
	if err != nil {
		return nil, fmt.Errorf("error getting openstack token: %s", err)
	}
	return token, nil
}

type cachedGetter struct {
	lk        sync.Mutex
	tg        TokenGetter
	persister restclient.AuthProviderConfigPersister
	cache     map[string]string
}

func newCachedGetter(tg TokenGetter, persister restclient.AuthProviderConfigPersister, cache map[string]string) *cachedGetter {
	if cache == nil {
		cache = make(map[string]string)
	}
	return &cachedGetter{
		tg:        tg,
		persister: persister,
		cache:     cache,
	}
}

// Token returns the current available token, create a new one if expired.
func (g *cachedGetter) Token() (*openstackToken, error) {
	if token, err := g.cachedToken(); err == nil {
		if token.Valid() {
			return token, nil
		}
	}
	token, err := g.tg.Token()
	if err != nil {
		return nil, err
	}
	cache := g.update(token)
	if g.persister != nil {
		if err := g.persister.Persist(cache); err != nil {
			glog.V(4).Infof("Failed to persist token: %v", err)
		}
	}
	return token, nil
}

func (g *cachedGetter) cachedToken() (*openstackToken, error) {
	g.lk.Lock()
	defer g.lk.Unlock()
	id, ok := g.cache["token-id"]
	if !ok {
		return nil, fmt.Errorf("the token ID is not in the cache")
	}
	expiresAt, ok := g.cache["expires-at"]
	if !ok {
		return nil, fmt.Errorf("the token expires-at time is not in the cache")
	}
	expiresAtTime, err := time.Parse(time.RFC3339Nano, expiresAt)
	if err != nil {
		return nil, err
	}
	return &openstackToken{
		ID:        id,
		ExpiresAt: expiresAtTime,
	}, nil
}

func (g *cachedGetter) update(token *openstackToken) map[string]string {
	g.lk.Lock()
	defer g.lk.Unlock()
	ret := map[string]string{}
	for k, v := range g.cache {
		ret[k] = v
	}
	ret["token-id"] = token.ID
	ret["expires-at"] = token.ExpiresAt.Format(time.RFC3339Nano)
	return ret
}

// tokenRoundTripper implements the RoundTripper interface: adding the bearer token
// into the request header.
type tokenRoundTripper struct {
	http.RoundTripper

	tg        TokenGetter
	persister restclient.AuthProviderConfigPersister
}

var _ net.RoundTripperWrapper = &tokenRoundTripper{}

// RoundTrip adds the bearer token into the request.
func (t *tokenRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	// if the authorization header already present, use it.
	if req.Header.Get("Authorization") != "" {
		return t.RoundTripper.RoundTrip(req)
	}

	token, err := t.tg.Token()
	if err == nil {
		req.Header.Set("Authorization", "Bearer "+token.ID)
	} else {
		glog.V(4).Infof("Failed to get token: %s", err)
	}

	res, err := t.RoundTripper.RoundTrip(req)

	if res.StatusCode == 401 {
		glog.V(4).Infof("The credentials that were supplied are invalid for the target cluster")
		emptyCache := make(map[string]string)
		t.persister.Persist(emptyCache)
	}

	return res, nil
}

func (t *tokenRoundTripper) WrappedRoundTripper() http.RoundTripper { return t.RoundTripper }

func (oap *openstackAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &tokenRoundTripper{
		RoundTripper: rt,
		tg:           oap.tokenGetter,
		persister:    oap.persister,
	}
}
