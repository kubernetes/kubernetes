package adal

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"testing"
)

const TestAuxTenantPrefix = "aux-tenant-test-"

var (
	TestAuxTenantIDs = []string{TestAuxTenantPrefix + "0", TestAuxTenantPrefix + "1", TestAuxTenantPrefix + "2"}
)

func TestNewOAuthConfig(t *testing.T) {
	config, err := NewOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID)
	if err != nil {
		t.Fatalf("autorest/adal: Unexpected error while creating oauth configuration for tenant: %v.", err)
	}

	expected := fmt.Sprintf("https://login.test.com/%s/oauth2/authorize?api-version=1.0", TestTenantID)
	if config.AuthorizeEndpoint.String() != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, config.AuthorizeEndpoint)
	}

	expected = fmt.Sprintf("https://login.test.com/%s/oauth2/token?api-version=1.0", TestTenantID)
	if config.TokenEndpoint.String() != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, config.TokenEndpoint)
	}

	expected = fmt.Sprintf("https://login.test.com/%s/oauth2/devicecode?api-version=1.0", TestTenantID)
	if config.DeviceCodeEndpoint.String() != expected {
		t.Fatalf("autorest/adal Incorrect devicecode url for Tenant from Environment. expected(%s). actual(%v).", expected, config.DeviceCodeEndpoint)
	}
}

func TestNewOAuthConfigWithAPIVersionNil(t *testing.T) {
	config, err := NewOAuthConfigWithAPIVersion(TestActiveDirectoryEndpoint, TestTenantID, nil)
	if err != nil {
		t.Fatalf("autorest/adal: Unexpected error while creating oauth configuration for tenant: %v.", err)
	}

	expected := fmt.Sprintf("https://login.test.com/%s/oauth2/authorize", TestTenantID)
	if config.AuthorizeEndpoint.String() != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, config.AuthorizeEndpoint)
	}

	expected = fmt.Sprintf("https://login.test.com/%s/oauth2/token", TestTenantID)
	if config.TokenEndpoint.String() != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, config.TokenEndpoint)
	}

	expected = fmt.Sprintf("https://login.test.com/%s/oauth2/devicecode", TestTenantID)
	if config.DeviceCodeEndpoint.String() != expected {
		t.Fatalf("autorest/adal Incorrect devicecode url for Tenant from Environment. expected(%s). actual(%v).", expected, config.DeviceCodeEndpoint)
	}
}

func TestNewOAuthConfigWithAPIVersionNotNil(t *testing.T) {
	apiVersion := "2.0"

	config, err := NewOAuthConfigWithAPIVersion(TestActiveDirectoryEndpoint, TestTenantID, &apiVersion)
	if err != nil {
		t.Fatalf("autorest/adal: Unexpected error while creating oauth configuration for tenant: %v.", err)
	}

	expected := fmt.Sprintf("https://login.test.com/%s/oauth2/authorize?api-version=2.0", TestTenantID)
	if config.AuthorizeEndpoint.String() != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, config.AuthorizeEndpoint)
	}

	expected = fmt.Sprintf("https://login.test.com/%s/oauth2/token?api-version=2.0", TestTenantID)
	if config.TokenEndpoint.String() != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, config.TokenEndpoint)
	}

	expected = fmt.Sprintf("https://login.test.com/%s/oauth2/devicecode?api-version=2.0", TestTenantID)
	if config.DeviceCodeEndpoint.String() != expected {
		t.Fatalf("autorest/adal Incorrect devicecode url for Tenant from Environment. expected(%s). actual(%v).", expected, config.DeviceCodeEndpoint)
	}
}

func TestNewMultiTenantOAuthConfig(t *testing.T) {
	cfg, err := NewMultiTenantOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID, TestAuxTenantIDs, OAuthOptions{})
	if err != nil {
		t.Fatalf("autorest/adal: unexpected error while creating multitenant config: %v", err)
	}
	expected := fmt.Sprintf("https://login.test.com/%s/oauth2/authorize?api-version=1.0", TestTenantID)
	if ep := cfg.PrimaryTenant().AuthorizeEndpoint.String(); ep != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, ep)
	}
	aux := cfg.AuxiliaryTenants()
	if len(aux) == 0 {
		t.Fatal("autorest/adal: unexpected zero-length auxiliary tenants")
	}
	for i := range aux {
		expected := fmt.Sprintf("https://login.test.com/aux-tenant-test-%d/oauth2/authorize?api-version=1.0", i)
		if ep := aux[i].AuthorizeEndpoint.String(); ep != expected {
			t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, ep)
		}
	}
}

func TestNewMultiTenantOAuthConfigFail(t *testing.T) {
	_, err := NewMultiTenantOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID, nil, OAuthOptions{})
	if err == nil {
		t.Fatal("autorest/adal: expected non-nil error")
	}
	_, err = NewMultiTenantOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID, []string{"one", "two", "three", "four"}, OAuthOptions{})
	if err == nil {
		t.Fatal("autorest/adal: expected non-nil error")
	}
}
