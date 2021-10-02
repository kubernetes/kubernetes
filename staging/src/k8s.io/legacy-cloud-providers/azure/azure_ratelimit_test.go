//go:build !providerless
// +build !providerless

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

package azure

import (
	"bytes"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/legacy-cloud-providers/azure/auth"
	azclients "k8s.io/legacy-cloud-providers/azure/clients"
)

var (
	testAzureConfig = `{
		"aadClientCertPassword": "aadClientCertPassword",
		"aadClientCertPath": "aadClientCertPath",
		"aadClientId": "aadClientId",
		"aadClientSecret": "aadClientSecret",
		"cloud":"AzurePublicCloud",
		"cloudProviderBackoff": true,
		"cloudProviderBackoffDuration": 1,
		"cloudProviderBackoffExponent": 1,
		"cloudProviderBackoffJitter": 1,
		"cloudProviderBackoffRetries": 1,
		"cloudProviderRatelimit": true,
		"cloudProviderRateLimitBucket": 1,
		"cloudProviderRateLimitBucketWrite": 1,
		"cloudProviderRateLimitQPS": 1,
		"cloudProviderRateLimitQPSWrite": 1,
		"virtualMachineScaleSetRateLimit": {
			"cloudProviderRatelimit": true,
			"cloudProviderRateLimitBucket": 2,
			"CloudProviderRateLimitBucketWrite": 2,
			"cloudProviderRateLimitQPS": 0,
			"CloudProviderRateLimitQPSWrite": 0
		},
		"loadBalancerRateLimit": {
			"cloudProviderRatelimit": false,
		},
		"networkResourceTenantId": "networkResourceTenantId",
		"networkResourceSubscriptionId": "networkResourceSubscriptionId",
		"availabilitySetNodesCacheTTLInSeconds": 100,
		"vmssCacheTTLInSeconds": 100,
		"vmssVirtualMachinesCacheTTLInSeconds": 100,
		"vmCacheTTLInSeconds": 100,
		"loadBalancerCacheTTLInSeconds": 100,
		"nsgCacheTTLInSeconds": 100,
		"routeTableCacheTTLInSeconds": 100,
		"location": "location",
		"maximumLoadBalancerRuleCount": 1,
		"primaryAvailabilitySetName": "primaryAvailabilitySetName",
		"primaryScaleSetName": "primaryScaleSetName",
		"resourceGroup": "resourceGroup",
		"routeTableName": "routeTableName",
		"routeTableResourceGroup": "routeTableResourceGroup",
		"securityGroupName": "securityGroupName",
		"securityGroupResourceGroup": "securityGroupResourceGroup",
		"subnetName": "subnetName",
		"subscriptionId": "subscriptionId",
		"tenantId": "tenantId",
		"useInstanceMetadata": true,
		"useManagedIdentityExtension": true,
		"vnetName": "vnetName",
		"vnetResourceGroup": "vnetResourceGroup",
		vmType: "standard"
	}`

	testDefaultRateLimitConfig = azclients.RateLimitConfig{
		CloudProviderRateLimit:            true,
		CloudProviderRateLimitBucket:      1,
		CloudProviderRateLimitBucketWrite: 1,
		CloudProviderRateLimitQPS:         1,
		CloudProviderRateLimitQPSWrite:    1,
	}
)

func TestParseConfig(t *testing.T) {
	expected := &Config{
		AzureAuthConfig: auth.AzureAuthConfig{
			AADClientCertPassword:         "aadClientCertPassword",
			AADClientCertPath:             "aadClientCertPath",
			AADClientID:                   "aadClientId",
			AADClientSecret:               "aadClientSecret",
			Cloud:                         "AzurePublicCloud",
			SubscriptionID:                "subscriptionId",
			TenantID:                      "tenantId",
			UseManagedIdentityExtension:   true,
			NetworkResourceTenantID:       "networkResourceTenantId",
			NetworkResourceSubscriptionID: "networkResourceSubscriptionId",
		},
		CloudProviderBackoff:         true,
		CloudProviderBackoffDuration: 1,
		CloudProviderBackoffExponent: 1,
		CloudProviderBackoffJitter:   1,
		CloudProviderBackoffRetries:  1,
		CloudProviderRateLimitConfig: CloudProviderRateLimitConfig{
			RateLimitConfig: testDefaultRateLimitConfig,
			LoadBalancerRateLimit: &azclients.RateLimitConfig{
				CloudProviderRateLimit: false,
			},
			VirtualMachineScaleSetRateLimit: &azclients.RateLimitConfig{
				CloudProviderRateLimit:            true,
				CloudProviderRateLimitBucket:      2,
				CloudProviderRateLimitBucketWrite: 2,
			},
		},
		AvailabilitySetNodesCacheTTLInSeconds: 100,
		VmssCacheTTLInSeconds:                 100,
		VmssVirtualMachinesCacheTTLInSeconds:  100,
		VMCacheTTLInSeconds:                   100,
		LoadBalancerCacheTTLInSeconds:         100,
		NsgCacheTTLInSeconds:                  100,
		RouteTableCacheTTLInSeconds:           100,
		Location:                              "location",
		MaximumLoadBalancerRuleCount:          1,
		PrimaryAvailabilitySetName:            "primaryAvailabilitySetName",
		PrimaryScaleSetName:                   "primaryScaleSetName",
		ResourceGroup:                         "resourcegroup",
		RouteTableName:                        "routeTableName",
		RouteTableResourceGroup:               "routeTableResourceGroup",
		SecurityGroupName:                     "securityGroupName",
		SecurityGroupResourceGroup:            "securityGroupResourceGroup",
		SubnetName:                            "subnetName",
		UseInstanceMetadata:                   true,
		VMType:                                "standard",
		VnetName:                              "vnetName",
		VnetResourceGroup:                     "vnetResourceGroup",
	}

	buffer := bytes.NewBufferString(testAzureConfig)
	config, err := parseConfig(buffer)
	assert.NoError(t, err)
	assert.Equal(t, expected, config)
}

func TestInitializeCloudProviderRateLimitConfig(t *testing.T) {
	buffer := bytes.NewBufferString(testAzureConfig)
	config, err := parseConfig(buffer)
	assert.NoError(t, err)

	InitializeCloudProviderRateLimitConfig(&config.CloudProviderRateLimitConfig)
	assert.Equal(t, config.LoadBalancerRateLimit, &azclients.RateLimitConfig{
		CloudProviderRateLimit: false,
	})
	assert.Equal(t, config.VirtualMachineScaleSetRateLimit, &azclients.RateLimitConfig{
		CloudProviderRateLimit:            true,
		CloudProviderRateLimitBucket:      2,
		CloudProviderRateLimitBucketWrite: 2,
		CloudProviderRateLimitQPS:         1,
		CloudProviderRateLimitQPSWrite:    1,
	})
	assert.Equal(t, config.VirtualMachineSizeRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.VirtualMachineRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.RouteRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.SubnetsRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.InterfaceRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.RouteTableRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.SecurityGroupRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.StorageAccountRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.DiskRateLimit, &testDefaultRateLimitConfig)
	assert.Equal(t, config.SnapshotRateLimit, &testDefaultRateLimitConfig)
}
