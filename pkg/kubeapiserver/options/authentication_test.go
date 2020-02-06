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

package options

import (
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	"k8s.io/apiserver/pkg/authentication/request/headerrequest"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	kubeauthenticator "k8s.io/kubernetes/pkg/kubeapiserver/authenticator"
)

func TestAuthenticationValidate(t *testing.T) {
	testCases := []struct {
		name      string
		testOIDC  *OIDCAuthenticationOptions
		testSA    *ServiceAccountAuthenticationOptions
		expectErr string
	}{
		{
			name: "test when OIDC and ServiceAccounts are nil",
		},
		{
			name: "test when OIDC and ServiceAccounts are valid",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuer: "http://foo.bar.com",
			},
		},
		{
			name: "test when OIDC is invalid",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuer: "http://foo.bar.com",
			},
			expectErr: "oidc-issuer-url and oidc-client-id should be specified together",
		},
		{
			name: "test when ServiceAccount is invalid",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuer: "http://[::1]:namedport",
			},
			expectErr: "service-account-issuer contained a ':' but was not a valid URL",
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			options := NewBuiltInAuthenticationOptions()
			options.OIDC = testcase.testOIDC
			options.ServiceAccounts = testcase.testSA

			errs := options.Validate()
			if len(errs) > 0 && !strings.Contains(utilerrors.NewAggregate(errs).Error(), testcase.expectErr) {
				t.Errorf("Got err: %v, Expected err: %s", errs, testcase.expectErr)
			}

			if len(errs) == 0 && len(testcase.expectErr) != 0 {
				t.Errorf("Got err nil, Expected err: %s", testcase.expectErr)
			}
		})
	}
}

func TestToAuthenticationConfig(t *testing.T) {
	testOptions := &BuiltInAuthenticationOptions{
		Anonymous: &AnonymousAuthenticationOptions{
			Allow: false,
		},
		ClientCert: &apiserveroptions.ClientCertAuthenticationOptions{
			ClientCA: "testdata/root.pem",
		},
		WebHook: &WebHookAuthenticationOptions{
			CacheTTL:   180000000000,
			ConfigFile: "/token-webhook-config",
		},
		BootstrapToken: &BootstrapTokenAuthenticationOptions{
			Enable: false,
		},
		OIDC: &OIDCAuthenticationOptions{
			CAFile:        "/testCAFile",
			UsernameClaim: "sub",
			SigningAlgs:   []string{"RS256"},
			IssuerURL:     "testIssuerURL",
			ClientID:      "testClientID",
		},
		PasswordFile: &PasswordFileAuthenticationOptions{
			BasicAuthFile: "/testBasicAuthFile",
		},
		RequestHeader: &apiserveroptions.RequestHeaderAuthenticationOptions{
			UsernameHeaders:     []string{"x-remote-user"},
			GroupHeaders:        []string{"x-remote-group"},
			ExtraHeaderPrefixes: []string{"x-remote-extra-"},
			ClientCAFile:        "testdata/root.pem",
			AllowedNames:        []string{"kube-aggregator"},
		},
		ServiceAccounts: &ServiceAccountAuthenticationOptions{
			Lookup: true,
			Issuer: "http://foo.bar.com",
		},
		TokenFile: &TokenFileAuthenticationOptions{
			TokenFile: "/testTokenFile",
		},
		TokenSuccessCacheTTL: 10 * time.Second,
		TokenFailureCacheTTL: 0,
	}

	expectConfig := kubeauthenticator.Config{
		APIAudiences:                authenticator.Audiences{"http://foo.bar.com"},
		Anonymous:                   false,
		BasicAuthFile:               "/testBasicAuthFile",
		BootstrapToken:              false,
		ClientCAContentProvider:     nil, // this is nil because you can't compare functions
		TokenAuthFile:               "/testTokenFile",
		OIDCIssuerURL:               "testIssuerURL",
		OIDCClientID:                "testClientID",
		OIDCCAFile:                  "/testCAFile",
		OIDCUsernameClaim:           "sub",
		OIDCSigningAlgs:             []string{"RS256"},
		ServiceAccountLookup:        true,
		ServiceAccountIssuer:        "http://foo.bar.com",
		WebhookTokenAuthnConfigFile: "/token-webhook-config",
		WebhookTokenAuthnCacheTTL:   180000000000,

		TokenSuccessCacheTTL: 10 * time.Second,
		TokenFailureCacheTTL: 0,

		RequestHeaderConfig: &authenticatorfactory.RequestHeaderConfig{
			UsernameHeaders:     headerrequest.StaticStringSlice{"x-remote-user"},
			GroupHeaders:        headerrequest.StaticStringSlice{"x-remote-group"},
			ExtraHeaderPrefixes: headerrequest.StaticStringSlice{"x-remote-extra-"},
			CAContentProvider:   nil, // this is nil because you can't compare functions
			AllowedClientNames:  headerrequest.StaticStringSlice{"kube-aggregator"},
		},
	}

	resultConfig, err := testOptions.ToAuthenticationConfig()
	if err != nil {
		t.Fatal(err)
	}

	// nil these out because you cannot compare pointers.  Ensure they are non-nil first
	if resultConfig.ClientCAContentProvider == nil {
		t.Error("missing client verify")
	}
	if resultConfig.RequestHeaderConfig.CAContentProvider == nil {
		t.Error("missing requestheader verify")
	}
	resultConfig.ClientCAContentProvider = nil
	resultConfig.RequestHeaderConfig.CAContentProvider = nil

	if !reflect.DeepEqual(resultConfig, expectConfig) {
		t.Error(cmp.Diff(resultConfig, expectConfig))
	}
}
