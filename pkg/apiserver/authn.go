/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package apiserver

import (
	"crypto/rsa"

	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/authenticator/bearertoken"
	"k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/password/passwordfile"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/basicauth"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/keystone"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/union"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/request/x509"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/tokenfile"
)

type AuthenticatorConfig struct {
	BasicAuthFile         string
	ClientCAFile          string
	TokenAuthFile         string
	OIDCIssuerURL         string
	OIDCClientID          string
	OIDCCAFile            string
	OIDCUsernameClaim     string
	ServiceAccountKeyFile string
	ServiceAccountLookup  bool
	Storage               storage.Interface
	KeystoneURL           string
}

// NewAuthenticator returns an authenticator.Request or an error
func NewAuthenticator(config AuthenticatorConfig) (authenticator.Request, error) {
	var authenticators []authenticator.Request

	if len(config.BasicAuthFile) > 0 {
		basicAuth, err := newAuthenticatorFromBasicAuthFile(config.BasicAuthFile)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, basicAuth)
	}

	if len(config.ClientCAFile) > 0 {
		certAuth, err := newAuthenticatorFromClientCAFile(config.ClientCAFile)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, certAuth)
	}

	if len(config.TokenAuthFile) > 0 {
		tokenAuth, err := newAuthenticatorFromTokenFile(config.TokenAuthFile)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, tokenAuth)
	}

	if len(config.OIDCIssuerURL) > 0 && len(config.OIDCClientID) > 0 {
		oidcAuth, err := newAuthenticatorFromOIDCIssuerURL(config.OIDCIssuerURL, config.OIDCClientID, config.OIDCCAFile, config.OIDCUsernameClaim)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, oidcAuth)
	}

	if len(config.ServiceAccountKeyFile) > 0 {
		serviceAccountAuth, err := newServiceAccountAuthenticator(config.ServiceAccountKeyFile, config.ServiceAccountLookup, config.Storage)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, serviceAccountAuth)
	}

	if len(config.KeystoneURL) > 0 {
		keystoneAuth, err := newAuthenticatorFromKeystoneURL(config.KeystoneURL)
		if err != nil {
			return nil, err
		}
		authenticators = append(authenticators, keystoneAuth)
	}

	switch len(authenticators) {
	case 0:
		return nil, nil
	case 1:
		return authenticators[0], nil
	default:
		return union.New(authenticators...), nil
	}
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

// newAuthenticatorFromOIDCIssuerURL returns an authenticator.Request or an error.
func newAuthenticatorFromOIDCIssuerURL(issuerURL, clientID, caFile, usernameClaim string) (authenticator.Request, error) {
	tokenAuthenticator, err := oidc.New(issuerURL, clientID, caFile, usernameClaim)
	if err != nil {
		return nil, err
	}

	return bearertoken.New(tokenAuthenticator), nil
}

// newServiceAccountAuthenticator returns an authenticator.Request or an error
func newServiceAccountAuthenticator(keyfile string, lookup bool, storage storage.Interface) (authenticator.Request, error) {
	publicKey, err := serviceaccount.ReadPublicKey(keyfile)
	if err != nil {
		return nil, err
	}

	var serviceAccountGetter serviceaccount.ServiceAccountTokenGetter
	if lookup {
		// If we need to look up service accounts and tokens,
		// go directly to etcd to avoid recursive auth insanity
		serviceAccountGetter = serviceaccount.NewGetterFromStorageInterface(storage)
	}

	tokenAuthenticator := serviceaccount.JWTTokenAuthenticator([]*rsa.PublicKey{publicKey}, lookup, serviceAccountGetter)
	return bearertoken.New(tokenAuthenticator), nil
}

// newAuthenticatorFromClientCAFile returns an authenticator.Request or an error
func newAuthenticatorFromClientCAFile(clientCAFile string) (authenticator.Request, error) {
	roots, err := util.CertPoolFromFile(clientCAFile)
	if err != nil {
		return nil, err
	}

	opts := x509.DefaultVerifyOptions()
	opts.Roots = roots

	return x509.New(opts, x509.CommonNameUserConversion), nil
}

// newAuthenticatorFromTokenFile returns an authenticator.Request or an error
func newAuthenticatorFromKeystoneURL(keystoneConfigFile string) (authenticator.Request, error) {
	keystoneAuthenticator, err := keystone.NewKeystoneAuthenticator(keystoneConfigFile)
	if err != nil {
		return nil, err
	}

	return basicauth.New(keystoneAuthenticator), nil
}
