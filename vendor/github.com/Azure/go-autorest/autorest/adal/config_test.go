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
	"testing"
)

func TestNewOAuthConfig(t *testing.T) {
	const testActiveDirectoryEndpoint = "https://login.test.com"
	const testTenantID = "tenant-id-test"

	config, err := NewOAuthConfig(testActiveDirectoryEndpoint, testTenantID)
	if err != nil {
		t.Fatalf("autorest/adal: Unexpected error while creating oauth configuration for tenant: %v.", err)
	}

	expected := "https://login.test.com/tenant-id-test/oauth2/authorize?api-version=1.0"
	if config.AuthorizeEndpoint.String() != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, config.AuthorizeEndpoint)
	}

	expected = "https://login.test.com/tenant-id-test/oauth2/token?api-version=1.0"
	if config.TokenEndpoint.String() != expected {
		t.Fatalf("autorest/adal: Incorrect authorize url for Tenant from Environment. expected(%s). actual(%v).", expected, config.TokenEndpoint)
	}

	expected = "https://login.test.com/tenant-id-test/oauth2/devicecode?api-version=1.0"
	if config.DeviceCodeEndpoint.String() != expected {
		t.Fatalf("autorest/adal Incorrect devicecode url for Tenant from Environment. expected(%s). actual(%v).", expected, config.DeviceCodeEndpoint)
	}
}
