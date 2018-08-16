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
	"strings"
	"testing"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
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
