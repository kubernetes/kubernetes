/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// oidc implements the authenticator.Token interface using the OpenID Connect protocol.
package oidc

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oidc"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/util/crypto"
	"k8s.io/kubernetes/pkg/util/net"
)

const (
	DefaultRetries = 5
	DefaultBackoff = time.Second * 3
)

type OIDCOptions struct {
	IssuerURL     string
	ClientID      string
	CAFile        string
	UsernameClaim string
	GroupsClaim   string

	// 0 disables retry
	MaxRetries   int
	RetryBackoff time.Duration
}

type OIDCAuthenticator struct {
	clientConfig     oidc.ClientConfig
	client           *oidc.Client
	usernameClaim    string
	groupsClaim      string
	stopSyncProvider chan struct{}
	maxRetries       int
	retryBackoff     time.Duration
}

// New creates a new OpenID Connect client with the given issuerURL and clientID.
// NOTE(yifan): For now we assume the server provides the "jwks_uri" so we don't
// need to manager the key sets by ourselves.
func New(opts OIDCOptions) (*OIDCAuthenticator, error) {
	var cfg oidc.ProviderConfig
	var err error
	var roots *x509.CertPool

	url, err := url.Parse(opts.IssuerURL)
	if err != nil {
		return nil, err
	}

	if url.Scheme != "https" {
		return nil, fmt.Errorf("'oidc-issuer-url' (%q) has invalid scheme (%q), require 'https'", opts.IssuerURL, url.Scheme)
	}

	if opts.CAFile != "" {
		roots, err = crypto.CertPoolFromFile(opts.CAFile)
		if err != nil {
			glog.Errorf("Failed to read the CA file: %v", err)
		}
	}
	if roots == nil {
		glog.Info("No x509 certificates provided, will use host's root CA set")
	}

	// Copied from http.DefaultTransport.
	tr := net.SetTransportDefaults(&http.Transport{
		// According to golang's doc, if RootCAs is nil,
		// TLS uses the host's root CA set.
		TLSClientConfig: &tls.Config{RootCAs: roots},
	})

	hc := &http.Client{}
	hc.Transport = tr

	maxRetries := opts.MaxRetries
	if maxRetries < 0 {
		maxRetries = DefaultRetries
	}
	retryBackoff := opts.RetryBackoff
	if retryBackoff < 0 {
		retryBackoff = DefaultBackoff
	}

	for i := 0; i <= maxRetries; i++ {
		if i == maxRetries {
			return nil, fmt.Errorf("failed to fetch provider config after %v retries", maxRetries)
		}

		cfg, err = oidc.FetchProviderConfig(hc, opts.IssuerURL)
		if err == nil {
			break
		}
		glog.Errorf("Failed to fetch provider config, trying again in %v: %v", retryBackoff, err)
		time.Sleep(retryBackoff)
	}

	glog.Infof("Fetched provider config from %s: %#v", opts.IssuerURL, cfg)

	ccfg := oidc.ClientConfig{
		HTTPClient:     hc,
		Credentials:    oidc.ClientCredentials{ID: opts.ClientID},
		ProviderConfig: cfg,
	}

	client, err := oidc.NewClient(ccfg)
	if err != nil {
		return nil, err
	}

	// SyncProviderConfig will start a goroutine to periodically synchronize the provider config.
	// The synchronization interval is set by the expiration length of the config, and has a mininum
	// and maximum threshold.
	stop := client.SyncProviderConfig(opts.IssuerURL)

	return &OIDCAuthenticator{
		ccfg,
		client,
		opts.UsernameClaim,
		opts.GroupsClaim,
		stop,
		maxRetries,
		retryBackoff,
	}, nil
}

// AuthenticateToken decodes and verifies a JWT using the OIDC client, if the verification succeeds,
// then it will extract the user info from the JWT claims.
func (a *OIDCAuthenticator) AuthenticateToken(value string) (user.Info, bool, error) {
	jwt, err := jose.ParseJWT(value)
	if err != nil {
		return nil, false, err
	}

	if err := a.client.VerifyJWT(jwt); err != nil {
		return nil, false, err
	}

	claims, err := jwt.Claims()
	if err != nil {
		return nil, false, err
	}

	claim, ok, err := claims.StringClaim(a.usernameClaim)
	if err != nil {
		return nil, false, err
	}
	if !ok {
		return nil, false, fmt.Errorf("cannot find %q in JWT claims", a.usernameClaim)
	}

	var username string
	switch a.usernameClaim {
	case "email":
		// TODO(yifan): Check 'email_verified' to make sure the email is valid.
		username = claim
	default:
		// For all other cases, use issuerURL + claim as the user name.
		username = fmt.Sprintf("%s#%s", a.clientConfig.ProviderConfig.Issuer, claim)
	}

	// TODO(yifan): Add UID, also populate the issuer to upper layer.
	info := &user.DefaultInfo{Name: username}

	if a.groupsClaim != "" {
		groups, found, err := claims.StringsClaim(a.groupsClaim)
		if err != nil {
			// Custom claim is present, but isn't an array of strings.
			return nil, false, fmt.Errorf("custom group claim contains invalid object: %v", err)
		}
		if found {
			info.Groups = groups
		}
	}
	return info, true, nil
}

// Close closes the OIDC authenticator, this will close the provider sync goroutine.
func (a *OIDCAuthenticator) Close() {
	// This assumes the s.stopSyncProvider is an unbuffered channel.
	// So instead of closing the channel, we send am empty struct here.
	// This guarantees that when this function returns, there is no flying requests,
	// because a send to an unbuffered channel happens after the receive from the channel.
	a.stopSyncProvider <- struct{}{}
}
