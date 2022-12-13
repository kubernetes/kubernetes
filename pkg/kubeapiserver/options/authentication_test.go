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
	"github.com/spf13/pflag"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/authenticatorfactory"
	"k8s.io/apiserver/pkg/authentication/request/headerrequest"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	kubeauthenticator "k8s.io/kubernetes/pkg/kubeapiserver/authenticator"
)

func TestAuthenticationValidate(t *testing.T) {
	testCases := []struct {
		name        string
		testOIDC    *OIDCAuthenticationOptions
		testSA      *ServiceAccountAuthenticationOptions
		testWebHook *WebHookAuthenticationOptions
		expectErr   string
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
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuers:  []string{"http://foo.bar.com"},
				KeyFiles: []string{"testkeyfile1", "testkeyfile2"},
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
				Issuers:  []string{"http://foo.bar.com"},
				KeyFiles: []string{"testkeyfile1", "testkeyfile2"},
			},
			expectErr: "oidc-issuer-url and oidc-client-id should be specified together",
		},
		{
			name: "test when ServiceAccounts doesn't have key file",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuers: []string{"http://foo.bar.com"},
			},
			expectErr: "service-account-key-file is a required flag",
		},
		{
			name: "test when ServiceAccounts doesn't have issuer",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuers: []string{},
			},
			expectErr: "service-account-issuer is a required flag",
		},
		{
			name: "test when ServiceAccounts has empty string as issuer",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuers: []string{""},
			},
			expectErr: "service-account-issuer should not be an empty string",
		},
		{
			name: "test when ServiceAccounts has duplicate issuers",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuers: []string{"http://foo.bar.com", "http://foo.bar.com"},
			},
			expectErr: "service-account-issuer \"http://foo.bar.com\" is already specified",
		},
		{
			name: "test when ServiceAccount has bad issuer",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				Issuers: []string{"http://[::1]:namedport"},
			},
			expectErr: "service-account-issuer \"http://[::1]:namedport\" contained a ':' but was not a valid URL",
		},
		{
			name: "test when ServiceAccounts has invalid JWKSURI",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				KeyFiles: []string{"cert", "key"},
				Issuers:  []string{"http://foo.bar.com"},
				JWKSURI:  "https://host:port",
			},
			expectErr: "service-account-jwks-uri must be a valid URL: parse \"https://host:port\": invalid port \":port\" after host",
		},
		{
			name: "test when ServiceAccounts has invalid JWKSURI (not https scheme)",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				KeyFiles: []string{"cert", "key"},
				Issuers:  []string{"http://foo.bar.com"},
				JWKSURI:  "http://baz.com",
			},
			expectErr: "service-account-jwks-uri requires https scheme, parsed as: http://baz.com",
		},
		{
			name: "test when WebHook has invalid retry attempts",
			testOIDC: &OIDCAuthenticationOptions{
				UsernameClaim: "sub",
				SigningAlgs:   []string{"RS256"},
				IssuerURL:     "testIssuerURL",
				ClientID:      "testClientID",
			},
			testSA: &ServiceAccountAuthenticationOptions{
				KeyFiles: []string{"cert", "key"},
				Issuers:  []string{"http://foo.bar.com"},
				JWKSURI:  "https://baz.com",
			},
			testWebHook: &WebHookAuthenticationOptions{
				ConfigFile: "configfile",
				Version:    "v1",
				CacheTTL:   60 * time.Second,
				RetryBackoff: &wait.Backoff{
					Duration: 500 * time.Millisecond,
					Factor:   1.5,
					Jitter:   0.2,
					Steps:    0,
				},
			},
			expectErr: "number of webhook retry attempts must be greater than 0, but is: 0",
		},
	}

	for _, testcase := range testCases {
		t.Run(testcase.name, func(t *testing.T) {
			options := NewBuiltInAuthenticationOptions()
			options.OIDC = testcase.testOIDC
			options.ServiceAccounts = testcase.testSA
			options.WebHook = testcase.testWebHook

			errs := options.Validate()
			if len(errs) > 0 && (!strings.Contains(utilerrors.NewAggregate(errs).Error(), testcase.expectErr) || testcase.expectErr == "") {
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
		RequestHeader: &apiserveroptions.RequestHeaderAuthenticationOptions{
			UsernameHeaders:     []string{"x-remote-user"},
			GroupHeaders:        []string{"x-remote-group"},
			ExtraHeaderPrefixes: []string{"x-remote-extra-"},
			ClientCAFile:        "testdata/root.pem",
			AllowedNames:        []string{"kube-aggregator"},
		},
		ServiceAccounts: &ServiceAccountAuthenticationOptions{
			Lookup:  true,
			Issuers: []string{"http://foo.bar.com"},
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
		BootstrapToken:              false,
		ClientCAContentProvider:     nil, // this is nil because you can't compare functions
		TokenAuthFile:               "/testTokenFile",
		OIDCIssuerURL:               "testIssuerURL",
		OIDCClientID:                "testClientID",
		OIDCCAFile:                  "/testCAFile",
		OIDCUsernameClaim:           "sub",
		OIDCSigningAlgs:             []string{"RS256"},
		ServiceAccountLookup:        true,
		ServiceAccountIssuers:       []string{"http://foo.bar.com"},
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

func TestBuiltInAuthenticationOptionsAddFlags(t *testing.T) {
	var args = []string{
		"--api-audiences=foo",
		"--anonymous-auth=true",
		"--enable-bootstrap-token-auth=true",
		"--oidc-issuer-url=https://baz.com",
		"--oidc-client-id=client-id",
		"--oidc-ca-file=cert",
		"--oidc-username-prefix=-",
		"--client-ca-file=client-cacert",
		"--requestheader-client-ca-file=testdata/root.pem",
		"--requestheader-username-headers=x-remote-user-custom",
		"--requestheader-group-headers=x-remote-group-custom",
		"--requestheader-allowed-names=kube-aggregator",
		"--service-account-key-file=cert",
		"--service-account-key-file=key",
		"--service-account-issuer=http://foo.bar.com",
		"--service-account-jwks-uri=https://qux.com",
		"--token-auth-file=tokenfile",
		"--authentication-token-webhook-config-file=webhook_config.yaml",
		"--authentication-token-webhook-cache-ttl=180s",
	}

	expected := &BuiltInAuthenticationOptions{
		APIAudiences: []string{"foo"},
		Anonymous: &AnonymousAuthenticationOptions{
			Allow: true,
		},
		BootstrapToken: &BootstrapTokenAuthenticationOptions{
			Enable: true,
		},
		ClientCert: &apiserveroptions.ClientCertAuthenticationOptions{
			ClientCA: "client-cacert",
		},
		OIDC: &OIDCAuthenticationOptions{
			CAFile:         "cert",
			ClientID:       "client-id",
			IssuerURL:      "https://baz.com",
			UsernameClaim:  "sub",
			UsernamePrefix: "-",
			SigningAlgs:    []string{"RS256"},
		},
		RequestHeader: &apiserveroptions.RequestHeaderAuthenticationOptions{
			ClientCAFile:    "testdata/root.pem",
			UsernameHeaders: []string{"x-remote-user-custom"},
			GroupHeaders:    []string{"x-remote-group-custom"},
			AllowedNames:    []string{"kube-aggregator"},
		},
		ServiceAccounts: &ServiceAccountAuthenticationOptions{
			KeyFiles:         []string{"cert", "key"},
			Lookup:           true,
			Issuers:          []string{"http://foo.bar.com"},
			JWKSURI:          "https://qux.com",
			ExtendExpiration: true,
		},
		TokenFile: &TokenFileAuthenticationOptions{
			TokenFile: "tokenfile",
		},
		WebHook: &WebHookAuthenticationOptions{
			ConfigFile: "webhook_config.yaml",
			Version:    "v1beta1",
			CacheTTL:   180 * time.Second,
			RetryBackoff: &wait.Backoff{
				Duration: 500 * time.Millisecond,
				Factor:   1.5,
				Jitter:   0.2,
				Steps:    5,
			},
		},
		TokenSuccessCacheTTL: 10 * time.Second,
		TokenFailureCacheTTL: 0 * time.Second,
	}

	opts := NewBuiltInAuthenticationOptions().WithAll()
	pf := pflag.NewFlagSet("test-builtin-authentication-opts", pflag.ContinueOnError)
	opts.AddFlags(pf)

	if err := pf.Parse(args); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(opts, expected) {
		t.Error(cmp.Diff(opts, expected))
	}
}
