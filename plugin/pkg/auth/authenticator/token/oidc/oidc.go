/*
Copyright 2015 The Kubernetes Authors.

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

/*
oidc implements the authenticator.Token interface using the OpenID Connect protocol.

	config := oidc.OIDCOptions{
		IssuerURL:     "https://accounts.google.com",
		ClientID:      os.Getenv("GOOGLE_CLIENT_ID"),
		UsernameClaim: "email",
	}
	tokenAuthenticator, err := oidc.New(config)
*/
package oidc

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oidc"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/auth/user"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/runtime"
)

type OIDCOptions struct {
	// IssuerURL is the URL the provider signs ID Tokens as. This will be the "iss"
	// field of all tokens produced by the provider and is used for configuration
	// discovery.
	//
	// The URL is usually the provider's URL without a path, for example
	// "https://accounts.google.com" or "https://login.salesforce.com".
	//
	// The provider must implement configuration discovery.
	// See: https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderConfig
	IssuerURL string

	// ClientID the JWT must be issued for, the "sub" field. This plugin only trusts a single
	// client to ensure the plugin can be used with public providers.
	//
	// The plugin supports the "authorized party" OpenID Connect claim, which allows
	// specialized providers to issue tokens to a client for a different client.
	// See: https://openid.net/specs/openid-connect-core-1_0.html#IDToken
	ClientID string

	// Path to a PEM encoded root certificate of the provider.
	CAFile string

	// UsernameClaim is the JWT field to use as the user's username.
	UsernameClaim string

	// GroupsClaim, if specified, causes the OIDCAuthenticator to try to populate the user's
	// groups with an ID Token field. If the GrouppClaim field is present in an ID Token the value
	// must be a string or list of strings.
	GroupsClaim string
}

type OIDCAuthenticator struct {
	issuerURL string

	trustedClientID string

	usernameClaim string
	groupsClaim   string

	httpClient *http.Client

	// Contains an *oidc.Client. Do not access directly. Use client() method.
	oidcClient atomic.Value

	// Guards the close method and is used to lock during initialization and closing.
	mu    sync.Mutex
	close func() // May be nil
}

// New creates a token authenticator which validates OpenID Connect ID Tokens.
func New(opts OIDCOptions) (*OIDCAuthenticator, error) {
	url, err := url.Parse(opts.IssuerURL)
	if err != nil {
		return nil, err
	}

	if url.Scheme != "https" {
		return nil, fmt.Errorf("'oidc-issuer-url' (%q) has invalid scheme (%q), require 'https'", opts.IssuerURL, url.Scheme)
	}

	if opts.UsernameClaim == "" {
		return nil, errors.New("no username claim provided")
	}

	var roots *x509.CertPool
	if opts.CAFile != "" {
		roots, err = certutil.NewPool(opts.CAFile)
		if err != nil {
			return nil, fmt.Errorf("Failed to read the CA file: %v", err)
		}
	} else {
		glog.Info("OIDC: No x509 certificates provided, will use host's root CA set")
	}

	// Copied from http.DefaultTransport.
	tr := net.SetTransportDefaults(&http.Transport{
		// According to golang's doc, if RootCAs is nil,
		// TLS uses the host's root CA set.
		TLSClientConfig: &tls.Config{RootCAs: roots},
	})

	authenticator := &OIDCAuthenticator{
		issuerURL:       opts.IssuerURL,
		trustedClientID: opts.ClientID,
		usernameClaim:   opts.UsernameClaim,
		groupsClaim:     opts.GroupsClaim,
		httpClient:      &http.Client{Transport: tr},
	}

	// Attempt to initialize the authenticator asynchronously.
	//
	// Ignore errors instead of returning it since the OpenID Connect provider might not be
	// available yet, for instance if it's running on the cluster and needs the API server
	// to come up first. Errors will be logged within the client() method.
	go func() {
		defer runtime.HandleCrash()
		authenticator.client()
	}()

	return authenticator, nil
}

// Close stops all goroutines used by the authenticator.
func (a *OIDCAuthenticator) Close() {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.close != nil {
		a.close()
	}
	return
}

func (a *OIDCAuthenticator) client() (*oidc.Client, error) {
	// Fast check to see if client has already been initialized.
	if client := a.oidcClient.Load(); client != nil {
		return client.(*oidc.Client), nil
	}

	// Acquire lock, then recheck initialization.
	a.mu.Lock()
	defer a.mu.Unlock()
	if client := a.oidcClient.Load(); client != nil {
		return client.(*oidc.Client), nil
	}

	// Try to initialize client.
	providerConfig, err := oidc.FetchProviderConfig(a.httpClient, a.issuerURL)
	if err != nil {
		glog.Errorf("oidc authenticator: failed to fetch provider discovery data: %v", err)
		return nil, fmt.Errorf("fetch provider config: %v", err)
	}

	clientConfig := oidc.ClientConfig{
		HTTPClient:     a.httpClient,
		Credentials:    oidc.ClientCredentials{ID: a.trustedClientID},
		ProviderConfig: providerConfig,
	}

	client, err := oidc.NewClient(clientConfig)
	if err != nil {
		glog.Errorf("oidc authenticator: failed to create client: %v", err)
		return nil, fmt.Errorf("create client: %v", err)
	}

	// SyncProviderConfig will start a goroutine to periodically synchronize the provider config.
	// The synchronization interval is set by the expiration length of the config, and has a minimum
	// and maximum threshold.
	stop := client.SyncProviderConfig(a.issuerURL)
	a.oidcClient.Store(client)
	a.close = func() {
		// This assumes the stop is an unbuffered channel.
		// So instead of closing the channel, we send am empty struct here.
		// This guarantees that when this function returns, there is no flying requests,
		// because a send to an unbuffered channel happens after the receive from the channel.
		stop <- struct{}{}
	}
	return client, nil
}

// AuthenticateToken decodes and verifies an ID Token using the OIDC client, if the verification succeeds,
// then it will extract the user info from the JWT claims.
func (a *OIDCAuthenticator) AuthenticateToken(value string) (user.Info, bool, error) {
	jwt, err := jose.ParseJWT(value)
	if err != nil {
		return nil, false, err
	}

	client, err := a.client()
	if err != nil {
		return nil, false, err
	}
	if err := client.VerifyJWT(jwt); err != nil {
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
		username = fmt.Sprintf("%s#%s", a.issuerURL, claim)
	}

	// TODO(yifan): Add UID, also populate the issuer to upper layer.
	info := &user.DefaultInfo{Name: username}

	if a.groupsClaim != "" {
		groups, found, err := claims.StringsClaim(a.groupsClaim)
		if err != nil {
			// Groups type is present but is not an array of strings, try to decode as a string.
			group, _, err := claims.StringClaim(a.groupsClaim)
			if err != nil {
				// Custom claim is present, but isn't an array of strings or a string.
				return nil, false, fmt.Errorf("custom group claim contains invalid type: %T", claims[a.groupsClaim])
			}
			info.Groups = []string{group}
		} else if found {
			info.Groups = groups
		}
	}
	return info, true, nil
}
