/*
Copyright 2014 The Kubernetes Authors.

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

package authenticator

import (
	"time"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authenticator/bearertoken"
	"k8s.io/kubernetes/pkg/auth/group"
	"k8s.io/kubernetes/pkg/auth/user"
	"k8s.io/kubernetes/pkg/serviceaccount"
	certutil "k8s.io/kubernetes/pkg/util/cert"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/password/keystone"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/password/passwordfile"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/anonymous"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/basicauth"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/union"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/x509"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/anytoken"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/tokenfile"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/webhook"
)

type AuthenticatorConfig struct {
	Anonymous                   bool
	AnyToken                    bool
	BasicAuthFile               string
	ClientCAFile                string
	TokenAuthFile               string
	OIDCIssuerURL               string
	OIDCClientID                string
	OIDCCAFile                  string
	OIDCUsernameClaim           string
	OIDCGroupsClaim             string
	ServiceAccountKeyFile       string
	ServiceAccountLookup        bool
	ServiceAccountTokenGetter   serviceaccount.ServiceAccountTokenGetter
	KeystoneURL                 string
	WebhookTokenAuthnConfigFile string
	WebhookTokenAuthnCacheTTL   time.Duration
}

// New returns an authenticator.Request or an error that supports the standard
// Kubernetes authentication mechanisms.
func New(config AuthenticatorConfig) (authenticator.Request, error) {
	var authenticators []authenticator.Request

	// BasicAuth methods, local first, then remote
	if len(config.BasicAuthFile) > 0 {
		basicAuth, err := newAuthenticatorFromBasicAuthFile(config.BasicAuthFile)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, basicAuth)
	}
	if len(config.KeystoneURL) > 0 {
		keystoneAuth, err := newAuthenticatorFromKeystoneURL(config.KeystoneURL)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, keystoneAuth)
	}

	// X509 methods
	if len(config.ClientCAFile) > 0 {
		certAuth, err := newAuthenticatorFromClientCAFile(config.ClientCAFile)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, certAuth)
	}

	// Bearer token methods, local first, then remote
	if len(config.TokenAuthFile) > 0 {
		tokenAuth, err := newAuthenticatorFromTokenFile(config.TokenAuthFile)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, tokenAuth)
	}
	if len(config.ServiceAccountKeyFile) > 0 {
		serviceAccountAuth, err := newServiceAccountAuthenticator(config.ServiceAccountKeyFile, config.ServiceAccountLookup, config.ServiceAccountTokenGetter)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, serviceAccountAuth)
	}
	// NOTE(ericchiang): Keep the OpenID Connect after Service Accounts.
	//
	// Because both plugins verify JWTs whichever comes first in the union experiences
	// cache misses for all requests using the other. While the service account plugin
	// simply returns an error, the OpenID Connect plugin may query the provider to
	// update the keys, causing performance hits.
	if len(config.OIDCIssuerURL) > 0 && len(config.OIDCClientID) > 0 {
		oidcAuth, err := newAuthenticatorFromOIDCIssuerURL(config.OIDCIssuerURL, config.OIDCClientID, config.OIDCCAFile, config.OIDCUsernameClaim, config.OIDCGroupsClaim)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, oidcAuth)
	}
	if len(config.WebhookTokenAuthnConfigFile) > 0 {
		webhookTokenAuth, err := newWebhookTokenAuthenticator(config.WebhookTokenAuthnConfigFile, config.WebhookTokenAuthnCacheTTL)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, webhookTokenAuth)
	}

	// always add anytoken last, so that every other token authenticator gets to try first
	if config.AnyToken {
		authenticators = append(authenticators, bearertoken.New(anytoken.AnyTokenAuthenticator{}))
	}

	if len(authenticators) == 0 {
		if config.Anonymous {
			return anonymous.NewAuthenticator(), nil
		}
	}

	switch len(authenticators) {
	case 0:
		return nil, nil
	}

	authenticator := union.New(authenticators...)

	authenticator = group.NewGroupAdder(authenticator, []string{user.AllAuthenticated})

	if config.Anonymous {
		// If the authenticator chain returns an error, return an error (don't consider a bad bearer token anonymous).
		authenticator = union.NewFailOnError(authenticator, anonymous.NewAuthenticator())
	}

	return authenticator, nil
}

// IsValidServiceAccountKeyFile returns true if a valid public RSA key can be read from the given file
func IsValidServiceAccountKeyFile(file string) bool {
	_, err := serviceaccount.ReadPublicKey(file)
	return err == nil
}

// newAuthenticatorFromBasicAuthFile returns an authenticator.Request or an error
func newAuthenticatorFromBasicAuthFile(basicAuthFile string) (authenticator.Request, error) {
	basicAuthenticator, err := passwordfile.NewCSV(basicAuthFile)
	if err != nil {
		return nil, err
	}

	return basicauth.New(basicAuthenticator), nil
}

// newAuthenticatorFromTokenFile returns an authenticator.Request or an error
func newAuthenticatorFromTokenFile(tokenAuthFile string) (authenticator.Request, error) {
	tokenAuthenticator, err := tokenfile.NewCSV(tokenAuthFile)
	if err != nil {
		return nil, err
	}

	return bearertoken.New(tokenAuthenticator), nil
}

// newAuthenticatorFromToken returns an authenticator.Request or an error
func NewAuthenticatorFromTokens(tokens map[string]*user.DefaultInfo) authenticator.Request {
	return bearertoken.New(tokenfile.New(tokens))
}

// newAuthenticatorFromOIDCIssuerURL returns an authenticator.Request or an error.
func newAuthenticatorFromOIDCIssuerURL(issuerURL, clientID, caFile, usernameClaim, groupsClaim string) (authenticator.Request, error) {
	tokenAuthenticator, err := oidc.New(oidc.OIDCOptions{
		IssuerURL:     issuerURL,
		ClientID:      clientID,
		CAFile:        caFile,
		UsernameClaim: usernameClaim,
		GroupsClaim:   groupsClaim,
	})
	if err != nil {
		return nil, err
	}

	return bearertoken.New(tokenAuthenticator), nil
}

// newServiceAccountAuthenticator returns an authenticator.Request or an error
func newServiceAccountAuthenticator(keyfile string, lookup bool, serviceAccountGetter serviceaccount.ServiceAccountTokenGetter) (authenticator.Request, error) {
	publicKey, err := serviceaccount.ReadPublicKey(keyfile)
	if err != nil {
		return nil, err
	}

	tokenAuthenticator := serviceaccount.JWTTokenAuthenticator([]interface{}{publicKey}, lookup, serviceAccountGetter)
	return bearertoken.New(tokenAuthenticator), nil
}

// newAuthenticatorFromClientCAFile returns an authenticator.Request or an error
func newAuthenticatorFromClientCAFile(clientCAFile string) (authenticator.Request, error) {
	roots, err := certutil.NewPool(clientCAFile)
	if err != nil {
		return nil, err
	}

	opts := x509.DefaultVerifyOptions()
	opts.Roots = roots

	return x509.New(opts, x509.CommonNameUserConversion), nil
}

// newAuthenticatorFromTokenFile returns an authenticator.Request or an error
func newAuthenticatorFromKeystoneURL(keystoneURL string) (authenticator.Request, error) {
	keystoneAuthenticator, err := keystone.NewKeystoneAuthenticator(keystoneURL)
	if err != nil {
		return nil, err
	}

	return basicauth.New(keystoneAuthenticator), nil
}

func newWebhookTokenAuthenticator(webhookConfigFile string, ttl time.Duration) (authenticator.Request, error) {
	webhookTokenAuthenticator, err := webhook.New(webhookConfigFile, ttl)
	if err != nil {
		return nil, err
	}

	return bearertoken.New(webhookTokenAuthenticator), nil
}
