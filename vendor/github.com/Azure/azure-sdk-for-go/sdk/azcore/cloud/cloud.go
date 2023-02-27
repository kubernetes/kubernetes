//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package cloud

var (
	// AzureChina contains configuration for Azure China.
	AzureChina = Configuration{
		ActiveDirectoryAuthorityHost: "https://login.chinacloudapi.cn/", Services: map[ServiceName]ServiceConfiguration{},
	}
	// AzureGovernment contains configuration for Azure Government.
	AzureGovernment = Configuration{
		ActiveDirectoryAuthorityHost: "https://login.microsoftonline.us/", Services: map[ServiceName]ServiceConfiguration{},
	}
	// AzurePublic contains configuration for Azure Public Cloud.
	AzurePublic = Configuration{
		ActiveDirectoryAuthorityHost: "https://login.microsoftonline.com/", Services: map[ServiceName]ServiceConfiguration{},
	}
)

// ServiceName identifies a cloud service.
type ServiceName string

// ResourceManager is a global constant identifying Azure Resource Manager.
const ResourceManager ServiceName = "resourceManager"

// ServiceConfiguration configures a specific cloud service such as Azure Resource Manager.
type ServiceConfiguration struct {
	// Audience is the audience the client will request for its access tokens.
	Audience string
	// Endpoint is the service's base URL.
	Endpoint string
}

// Configuration configures a cloud.
type Configuration struct {
	// ActiveDirectoryAuthorityHost is the base URL of the cloud's Azure Active Directory.
	ActiveDirectoryAuthorityHost string
	// Services contains configuration for the cloud's services.
	Services map[ServiceName]ServiceConfiguration
}
