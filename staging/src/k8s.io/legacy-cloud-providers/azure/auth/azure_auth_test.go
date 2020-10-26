/*
Copyright 2019 The Kubernetes Authors.

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

package auth

import (
	"testing"

	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/stretchr/testify/assert"
)

var (
	CrossTenantNetworkResourceNegativeConfig = []*AzureAuthConfig{
		{
			TenantID:        "TenantID",
			AADClientID:     "AADClientID",
			AADClientSecret: "AADClientSecret",
		},
		{
			TenantID:                      "TenantID",
			AADClientID:                   "AADClientID",
			AADClientSecret:               "AADClientSecret",
			NetworkResourceTenantID:       "NetworkResourceTenantID",
			NetworkResourceSubscriptionID: "NetworkResourceSubscriptionID",
			IdentitySystem:                ADFSIdentitySystem,
		},
		{
			TenantID:                      "TenantID",
			AADClientID:                   "AADClientID",
			AADClientSecret:               "AADClientSecret",
			NetworkResourceTenantID:       "NetworkResourceTenantID",
			NetworkResourceSubscriptionID: "NetworkResourceSubscriptionID",
			UseManagedIdentityExtension:   true,
		},
	}
)

func TestGetServicePrincipalTokenFromMSIWithUserAssignedID(t *testing.T) {
	configs := []*AzureAuthConfig{
		{
			UseManagedIdentityExtension: true,
			UserAssignedIdentityID:      "UserAssignedIdentityID",
		},
		// The Azure service principal is ignored when
		// UseManagedIdentityExtension is set to true
		{
			UseManagedIdentityExtension: true,
			UserAssignedIdentityID:      "UserAssignedIdentityID",
			TenantID:                    "TenantID",
			AADClientID:                 "AADClientID",
			AADClientSecret:             "AADClientSecret",
		},
	}
	env := &azure.PublicCloud

	for _, config := range configs {
		token, err := GetServicePrincipalToken(config, env)
		assert.NoError(t, err)

		msiEndpoint, err := adal.GetMSIVMEndpoint()
		assert.NoError(t, err)

		spt, err := adal.NewServicePrincipalTokenFromMSIWithUserAssignedID(msiEndpoint,
			env.ServiceManagementEndpoint, config.UserAssignedIdentityID)
		assert.NoError(t, err)
		assert.Equal(t, token, spt)
	}
}

func TestGetServicePrincipalTokenFromMSI(t *testing.T) {
	configs := []*AzureAuthConfig{
		{
			UseManagedIdentityExtension: true,
		},
		// The Azure service principal is ignored when
		// UseManagedIdentityExtension is set to true
		{
			UseManagedIdentityExtension: true,
			TenantID:                    "TenantID",
			AADClientID:                 "AADClientID",
			AADClientSecret:             "AADClientSecret",
		},
	}
	env := &azure.PublicCloud

	for _, config := range configs {
		token, err := GetServicePrincipalToken(config, env)
		assert.NoError(t, err)

		msiEndpoint, err := adal.GetMSIVMEndpoint()
		assert.NoError(t, err)

		spt, err := adal.NewServicePrincipalTokenFromMSI(msiEndpoint, env.ServiceManagementEndpoint)
		assert.NoError(t, err)
		assert.Equal(t, token, spt)
	}

}

func TestGetServicePrincipalToken(t *testing.T) {
	config := &AzureAuthConfig{
		TenantID:        "TenantID",
		AADClientID:     "AADClientID",
		AADClientSecret: "AADClientSecret",
	}
	env := &azure.PublicCloud

	token, err := GetServicePrincipalToken(config, env)
	assert.NoError(t, err)

	oauthConfig, err := adal.NewOAuthConfigWithAPIVersion(env.ActiveDirectoryEndpoint, config.TenantID, nil)
	assert.NoError(t, err)

	spt, err := adal.NewServicePrincipalToken(*oauthConfig, config.AADClientID, config.AADClientSecret, env.ServiceManagementEndpoint)
	assert.NoError(t, err)

	assert.Equal(t, token, spt)
}

func TestGetMultiTenantServicePrincipalToken(t *testing.T) {
	config := &AzureAuthConfig{
		TenantID:                      "TenantID",
		AADClientID:                   "AADClientID",
		AADClientSecret:               "AADClientSecret",
		NetworkResourceTenantID:       "NetworkResourceTenantID",
		NetworkResourceSubscriptionID: "NetworkResourceSubscriptionID",
	}
	env := &azure.PublicCloud

	multiTenantToken, err := GetMultiTenantServicePrincipalToken(config, env)
	assert.NoError(t, err)

	multiTenantOAuthConfig, err := adal.NewMultiTenantOAuthConfig(env.ActiveDirectoryEndpoint, config.TenantID, []string{config.NetworkResourceTenantID}, adal.OAuthOptions{})
	assert.NoError(t, err)

	spt, err := adal.NewMultiTenantServicePrincipalToken(multiTenantOAuthConfig, config.AADClientID, config.AADClientSecret, env.ServiceManagementEndpoint)
	assert.NoError(t, err)

	assert.Equal(t, multiTenantToken, spt)
}

func TestGetMultiTenantServicePrincipalTokenNegative(t *testing.T) {
	env := &azure.PublicCloud
	for _, config := range CrossTenantNetworkResourceNegativeConfig {
		_, err := GetMultiTenantServicePrincipalToken(config, env)
		assert.Error(t, err)
	}
}

func TestGetNetworkResourceServicePrincipalToken(t *testing.T) {
	config := &AzureAuthConfig{
		TenantID:                      "TenantID",
		AADClientID:                   "AADClientID",
		AADClientSecret:               "AADClientSecret",
		NetworkResourceTenantID:       "NetworkResourceTenantID",
		NetworkResourceSubscriptionID: "NetworkResourceSubscriptionID",
	}
	env := &azure.PublicCloud

	token, err := GetNetworkResourceServicePrincipalToken(config, env)
	assert.NoError(t, err)

	oauthConfig, err := adal.NewOAuthConfigWithAPIVersion(env.ActiveDirectoryEndpoint, config.NetworkResourceTenantID, nil)
	assert.NoError(t, err)

	spt, err := adal.NewServicePrincipalToken(*oauthConfig, config.AADClientID, config.AADClientSecret, env.ServiceManagementEndpoint)
	assert.NoError(t, err)

	assert.Equal(t, token, spt)
}

func TestGetNetworkResourceServicePrincipalTokenNegative(t *testing.T) {
	env := &azure.PublicCloud
	for _, config := range CrossTenantNetworkResourceNegativeConfig {
		_, err := GetNetworkResourceServicePrincipalToken(config, env)
		assert.Error(t, err)
	}
}

func TestParseAzureEnvironment(t *testing.T) {
	cases := []struct {
		cloudName               string
		resourceManagerEndpoint string
		identitySystem          string
		expected                *azure.Environment
	}{
		{
			cloudName:               "",
			resourceManagerEndpoint: "",
			identitySystem:          "",
			expected:                &azure.PublicCloud,
		},
		{
			cloudName:               "AZURECHINACLOUD",
			resourceManagerEndpoint: "",
			identitySystem:          "",
			expected:                &azure.ChinaCloud,
		},
	}

	for _, c := range cases {
		env, err := ParseAzureEnvironment(c.cloudName, c.resourceManagerEndpoint, c.identitySystem)
		assert.NoError(t, err)
		assert.Equal(t, env, c.expected)
	}
}

func TestAzureStackOverrides(t *testing.T) {
	env := &azure.PublicCloud
	resourceManagerEndpoint := "https://management.test.com/"

	azureStackOverrides(env, resourceManagerEndpoint, "")
	assert.Equal(t, env.ManagementPortalURL, "https://portal.test.com/")
	assert.Equal(t, env.ServiceManagementEndpoint, env.TokenAudience)
	assert.Equal(t, env.ResourceManagerVMDNSSuffix, "cloudapp.test.com")
	assert.Equal(t, env.ActiveDirectoryEndpoint, "https://login.microsoftonline.com/")

	azureStackOverrides(env, resourceManagerEndpoint, "adfs")
	assert.Equal(t, env.ManagementPortalURL, "https://portal.test.com/")
	assert.Equal(t, env.ServiceManagementEndpoint, env.TokenAudience)
	assert.Equal(t, env.ResourceManagerVMDNSSuffix, "cloudapp.test.com")
	assert.Equal(t, env.ActiveDirectoryEndpoint, "https://login.microsoftonline.com")
}
