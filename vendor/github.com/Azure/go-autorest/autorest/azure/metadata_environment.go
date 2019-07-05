package azure

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/Azure/go-autorest/autorest"
)

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

type audience []string

type authentication struct {
	LoginEndpoint string   `json:"loginEndpoint"`
	Audiences     audience `json:"audiences"`
}

type environmentMetadataInfo struct {
	GalleryEndpoint string         `json:"galleryEndpoint"`
	GraphEndpoint   string         `json:"graphEndpoint"`
	PortalEndpoint  string         `json:"portalEndpoint"`
	Authentication  authentication `json:"authentication"`
}

// EnvironmentProperty represent property names that clients can override
type EnvironmentProperty string

const (
	// EnvironmentName ...
	EnvironmentName EnvironmentProperty = "name"
	// EnvironmentManagementPortalURL ..
	EnvironmentManagementPortalURL EnvironmentProperty = "managementPortalURL"
	// EnvironmentPublishSettingsURL ...
	EnvironmentPublishSettingsURL EnvironmentProperty = "publishSettingsURL"
	// EnvironmentServiceManagementEndpoint ...
	EnvironmentServiceManagementEndpoint EnvironmentProperty = "serviceManagementEndpoint"
	// EnvironmentResourceManagerEndpoint ...
	EnvironmentResourceManagerEndpoint EnvironmentProperty = "resourceManagerEndpoint"
	// EnvironmentActiveDirectoryEndpoint ...
	EnvironmentActiveDirectoryEndpoint EnvironmentProperty = "activeDirectoryEndpoint"
	// EnvironmentGalleryEndpoint ...
	EnvironmentGalleryEndpoint EnvironmentProperty = "galleryEndpoint"
	// EnvironmentKeyVaultEndpoint ...
	EnvironmentKeyVaultEndpoint EnvironmentProperty = "keyVaultEndpoint"
	// EnvironmentGraphEndpoint ...
	EnvironmentGraphEndpoint EnvironmentProperty = "graphEndpoint"
	// EnvironmentServiceBusEndpoint ...
	EnvironmentServiceBusEndpoint EnvironmentProperty = "serviceBusEndpoint"
	// EnvironmentBatchManagementEndpoint ...
	EnvironmentBatchManagementEndpoint EnvironmentProperty = "batchManagementEndpoint"
	// EnvironmentStorageEndpointSuffix ...
	EnvironmentStorageEndpointSuffix EnvironmentProperty = "storageEndpointSuffix"
	// EnvironmentSQLDatabaseDNSSuffix ...
	EnvironmentSQLDatabaseDNSSuffix EnvironmentProperty = "sqlDatabaseDNSSuffix"
	// EnvironmentTrafficManagerDNSSuffix ...
	EnvironmentTrafficManagerDNSSuffix EnvironmentProperty = "trafficManagerDNSSuffix"
	// EnvironmentKeyVaultDNSSuffix ...
	EnvironmentKeyVaultDNSSuffix EnvironmentProperty = "keyVaultDNSSuffix"
	// EnvironmentServiceBusEndpointSuffix ...
	EnvironmentServiceBusEndpointSuffix EnvironmentProperty = "serviceBusEndpointSuffix"
	// EnvironmentServiceManagementVMDNSSuffix ...
	EnvironmentServiceManagementVMDNSSuffix EnvironmentProperty = "serviceManagementVMDNSSuffix"
	// EnvironmentResourceManagerVMDNSSuffix ...
	EnvironmentResourceManagerVMDNSSuffix EnvironmentProperty = "resourceManagerVMDNSSuffix"
	// EnvironmentContainerRegistryDNSSuffix ...
	EnvironmentContainerRegistryDNSSuffix EnvironmentProperty = "containerRegistryDNSSuffix"
	// EnvironmentTokenAudience ...
	EnvironmentTokenAudience EnvironmentProperty = "tokenAudience"
)

// OverrideProperty represents property name and value that clients can override
type OverrideProperty struct {
	Key   EnvironmentProperty
	Value string
}

// EnvironmentFromURL loads an Environment from a URL
// This function is particularly useful in the Hybrid Cloud model, where one may define their own
// endpoints.
func EnvironmentFromURL(resourceManagerEndpoint string, properties ...OverrideProperty) (environment Environment, err error) {
	var metadataEnvProperties environmentMetadataInfo

	if resourceManagerEndpoint == "" {
		return environment, fmt.Errorf("Metadata resource manager endpoint is empty")
	}

	if metadataEnvProperties, err = retrieveMetadataEnvironment(resourceManagerEndpoint); err != nil {
		return environment, err
	}

	// Give priority to user's override values
	overrideProperties(&environment, properties)

	if environment.Name == "" {
		environment.Name = "HybridEnvironment"
	}
	stampDNSSuffix := environment.StorageEndpointSuffix
	if stampDNSSuffix == "" {
		stampDNSSuffix = strings.TrimSuffix(strings.TrimPrefix(strings.Replace(resourceManagerEndpoint, strings.Split(resourceManagerEndpoint, ".")[0], "", 1), "."), "/")
		environment.StorageEndpointSuffix = stampDNSSuffix
	}
	if environment.KeyVaultDNSSuffix == "" {
		environment.KeyVaultDNSSuffix = fmt.Sprintf("%s.%s", "vault", stampDNSSuffix)
	}
	if environment.KeyVaultEndpoint == "" {
		environment.KeyVaultEndpoint = fmt.Sprintf("%s%s", "https://", environment.KeyVaultDNSSuffix)
	}
	if environment.TokenAudience == "" {
		environment.TokenAudience = metadataEnvProperties.Authentication.Audiences[0]
	}
	if environment.ActiveDirectoryEndpoint == "" {
		environment.ActiveDirectoryEndpoint = metadataEnvProperties.Authentication.LoginEndpoint
	}
	if environment.ResourceManagerEndpoint == "" {
		environment.ResourceManagerEndpoint = resourceManagerEndpoint
	}
	if environment.GalleryEndpoint == "" {
		environment.GalleryEndpoint = metadataEnvProperties.GalleryEndpoint
	}
	if environment.GraphEndpoint == "" {
		environment.GraphEndpoint = metadataEnvProperties.GraphEndpoint
	}

	return environment, nil
}

func overrideProperties(environment *Environment, properties []OverrideProperty) {
	for _, property := range properties {
		switch property.Key {
		case EnvironmentName:
			{
				environment.Name = property.Value
			}
		case EnvironmentManagementPortalURL:
			{
				environment.ManagementPortalURL = property.Value
			}
		case EnvironmentPublishSettingsURL:
			{
				environment.PublishSettingsURL = property.Value
			}
		case EnvironmentServiceManagementEndpoint:
			{
				environment.ServiceManagementEndpoint = property.Value
			}
		case EnvironmentResourceManagerEndpoint:
			{
				environment.ResourceManagerEndpoint = property.Value
			}
		case EnvironmentActiveDirectoryEndpoint:
			{
				environment.ActiveDirectoryEndpoint = property.Value
			}
		case EnvironmentGalleryEndpoint:
			{
				environment.GalleryEndpoint = property.Value
			}
		case EnvironmentKeyVaultEndpoint:
			{
				environment.KeyVaultEndpoint = property.Value
			}
		case EnvironmentGraphEndpoint:
			{
				environment.GraphEndpoint = property.Value
			}
		case EnvironmentServiceBusEndpoint:
			{
				environment.ServiceBusEndpoint = property.Value
			}
		case EnvironmentBatchManagementEndpoint:
			{
				environment.BatchManagementEndpoint = property.Value
			}
		case EnvironmentStorageEndpointSuffix:
			{
				environment.StorageEndpointSuffix = property.Value
			}
		case EnvironmentSQLDatabaseDNSSuffix:
			{
				environment.SQLDatabaseDNSSuffix = property.Value
			}
		case EnvironmentTrafficManagerDNSSuffix:
			{
				environment.TrafficManagerDNSSuffix = property.Value
			}
		case EnvironmentKeyVaultDNSSuffix:
			{
				environment.KeyVaultDNSSuffix = property.Value
			}
		case EnvironmentServiceBusEndpointSuffix:
			{
				environment.ServiceBusEndpointSuffix = property.Value
			}
		case EnvironmentServiceManagementVMDNSSuffix:
			{
				environment.ServiceManagementVMDNSSuffix = property.Value
			}
		case EnvironmentResourceManagerVMDNSSuffix:
			{
				environment.ResourceManagerVMDNSSuffix = property.Value
			}
		case EnvironmentContainerRegistryDNSSuffix:
			{
				environment.ContainerRegistryDNSSuffix = property.Value
			}
		case EnvironmentTokenAudience:
			{
				environment.TokenAudience = property.Value
			}
		}
	}
}

func retrieveMetadataEnvironment(endpoint string) (environment environmentMetadataInfo, err error) {
	client := autorest.NewClientWithUserAgent("")
	managementEndpoint := fmt.Sprintf("%s%s", strings.TrimSuffix(endpoint, "/"), "/metadata/endpoints?api-version=1.0")
	req, _ := http.NewRequest("GET", managementEndpoint, nil)
	response, err := client.Do(req)
	if err != nil {
		return environment, err
	}
	defer response.Body.Close()
	jsonResponse, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return environment, err
	}
	err = json.Unmarshal(jsonResponse, &environment)
	return environment, err
}
