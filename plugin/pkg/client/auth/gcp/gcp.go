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

package gcp

import (
	"net/http"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"

	clientreporestclient "k8s.io/client-go/1.4/rest"
	"k8s.io/kubernetes/pkg/client/restclient"
)

func init() {
	if err := restclient.RegisterAuthProviderPlugin("gcp", newGCPAuthProvider); err != nil {
		glog.Fatalf("Failed to register gcp auth plugin: %v", err)
	}
	if err := clientreporestclient.RegisterAuthProviderPlugin("gcp", newGCPAuthProviderForClientRepo); err != nil {
		glog.Fatalf("Failed to register gcp auth plugin: %v", err)
	}
}

type gcpAuthProvider struct {
	tokenSource oauth2.TokenSource
	persister   restclient.AuthProviderConfigPersister
}

func newGCPAuthProvider(_ string, gcpConfig map[string]string, persister restclient.AuthProviderConfigPersister) (restclient.AuthProvider, error) {
	ts, err := newCachedTokenSource(gcpConfig["access-token"], gcpConfig["expiry"], persister)
	if err != nil {
		return nil, err
	}
	return &gcpAuthProvider{ts, persister}, nil
}

// newGCPAuthProviderForClientRepo is the same as newGCPAuthProvider, but is programmed against client-go's interface
func newGCPAuthProviderForClientRepo(_ string, gcpConfig map[string]string, persister clientreporestclient.AuthProviderConfigPersister) (clientreporestclient.AuthProvider, error) {
	ts, err := newCachedTokenSource(gcpConfig["access-token"], gcpConfig["expiry"], persister)
	if err != nil {
		return nil, err
	}
	return &gcpAuthProvider{ts, persister}, nil
}

func (g *gcpAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &oauth2.Transport{
		Source: g.tokenSource,
		Base:   rt,
	}
}

func (g *gcpAuthProvider) Login() error { return nil }

type cachedTokenSource struct {
	source      oauth2.TokenSource
	accessToken string
	expiry      time.Time
	persister   restclient.AuthProviderConfigPersister
}

func newCachedTokenSource(accessToken, expiry string, persister restclient.AuthProviderConfigPersister) (*cachedTokenSource, error) {
	var expiryTime time.Time
	if parsedTime, err := time.Parse(time.RFC3339Nano, expiry); err == nil {
		expiryTime = parsedTime
	}
	ts, err := google.DefaultTokenSource(context.Background(), "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, err
	}
	return &cachedTokenSource{
		source:      ts,
		accessToken: accessToken,
		expiry:      expiryTime,
		persister:   persister,
	}, nil
}

func (t *cachedTokenSource) Token() (*oauth2.Token, error) {
	tok := &oauth2.Token{
		AccessToken: t.accessToken,
		TokenType:   "Bearer",
		Expiry:      t.expiry,
	}
	if tok.Valid() && !tok.Expiry.IsZero() {
		return tok, nil
	}
	tok, err := t.source.Token()
	if err != nil {
		return nil, err
	}
	if t.persister != nil {
		cached := map[string]string{
			"access-token": tok.AccessToken,
			"expiry":       tok.Expiry.Format(time.RFC3339Nano),
		}
		if err := t.persister.Persist(cached); err != nil {
			glog.V(4).Infof("Failed to persist token: %v", err)
		}
	}
	return tok, nil
}
