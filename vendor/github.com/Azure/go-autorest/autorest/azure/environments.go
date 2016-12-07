package azure

import (
	"fmt"
	"net/url"
	"strings"
)

const (
	activeDirectoryAPIVersion = "1.0"
)

var environments = map[string]Environment{
	"AZURECHINACLOUD":        ChinaCloud,
	"AZUREGERMANCLOUD":       GermanCloud,
	"AZUREPUBLICCLOUD":       PublicCloud,
	"AZUREUSGOVERNMENTCLOUD": USGovernmentCloud,
}

// Environment represents a set of endpoints for each of Azure's Clouds.
type Environment struct {
	Name                         string `json:"name"`
	ManagementPortalURL          string `json:"managementPortalURL"`
	PublishSettingsURL           string `json:"publishSettingsURL"`
	ServiceManagementEndpoint    string `json:"serviceManagementEndpoint"`
	ResourceManagerEndpoint      string `json:"resourceManagerEndpoint"`
	ActiveDirectoryEndpoint      string `json:"activeDirectoryEndpoint"`
	GalleryEndpoint              string `json:"galleryEndpoint"`
	KeyVaultEndpoint             string `json:"keyVaultEndpoint"`
	GraphEndpoint                string `json:"graphEndpoint"`
	StorageEndpointSuffix        string `json:"storageEndpointSuffix"`
	SQLDatabaseDNSSuffix         string `json:"sqlDatabaseDNSSuffix"`
	TrafficManagerDNSSuffix      string `json:"trafficManagerDNSSuffix"`
	KeyVaultDNSSuffix            string `json:"keyVaultDNSSuffix"`
	ServiceBusEndpointSuffix     string `json:"serviceBusEndpointSuffix"`
	ServiceManagementVMDNSSuffix string `json:"serviceManagementVMDNSSuffix"`
	ResourceManagerVMDNSSuffix   string `json:"resourceManagerVMDNSSuffix"`
}

var (
	// PublicCloud is the default public Azure cloud environment
	PublicCloud = Environment{
		Name:                         "AzurePublicCloud",
		ManagementPortalURL:          "https://manage.windowsazure.com/",
		PublishSettingsURL:           "https://manage.windowsazure.com/publishsettings/index",
		ServiceManagementEndpoint:    "https://management.core.windows.net/",
		ResourceManagerEndpoint:      "https://management.azure.com/",
		ActiveDirectoryEndpoint:      "https://login.microsoftonline.com/",
		GalleryEndpoint:              "https://gallery.azure.com/",
		KeyVaultEndpoint:             "https://vault.azure.net/",
		GraphEndpoint:                "https://graph.windows.net/",
		StorageEndpointSuffix:        "core.windows.net",
		SQLDatabaseDNSSuffix:         "database.windows.net",
		TrafficManagerDNSSuffix:      "trafficmanager.net",
		KeyVaultDNSSuffix:            "vault.azure.net",
		ServiceBusEndpointSuffix:     "servicebus.azure.com",
		ServiceManagementVMDNSSuffix: "cloudapp.net",
		ResourceManagerVMDNSSuffix:   "cloudapp.azure.com",
	}

	// USGovernmentCloud is the cloud environment for the US Government
	USGovernmentCloud = Environment{
		Name:                         "AzureUSGovernmentCloud",
		ManagementPortalURL:          "https://manage.windowsazure.us/",
		PublishSettingsURL:           "https://manage.windowsazure.us/publishsettings/index",
		ServiceManagementEndpoint:    "https://management.core.usgovcloudapi.net/",
		ResourceManagerEndpoint:      "https://management.usgovcloudapi.net/",
		ActiveDirectoryEndpoint:      "https://login.microsoftonline.com/",
		GalleryEndpoint:              "https://gallery.usgovcloudapi.net/",
		KeyVaultEndpoint:             "https://vault.usgovcloudapi.net/",
		GraphEndpoint:                "https://graph.usgovcloudapi.net/",
		StorageEndpointSuffix:        "core.usgovcloudapi.net",
		SQLDatabaseDNSSuffix:         "database.usgovcloudapi.net",
		TrafficManagerDNSSuffix:      "usgovtrafficmanager.net",
		KeyVaultDNSSuffix:            "vault.usgovcloudapi.net",
		ServiceBusEndpointSuffix:     "servicebus.usgovcloudapi.net",
		ServiceManagementVMDNSSuffix: "usgovcloudapp.net",
		ResourceManagerVMDNSSuffix:   "cloudapp.windowsazure.us",
	}

	// ChinaCloud is the cloud environment operated in China
	ChinaCloud = Environment{
		Name:                         "AzureChinaCloud",
		ManagementPortalURL:          "https://manage.chinacloudapi.com/",
		PublishSettingsURL:           "https://manage.chinacloudapi.com/publishsettings/index",
		ServiceManagementEndpoint:    "https://management.core.chinacloudapi.cn/",
		ResourceManagerEndpoint:      "https://management.chinacloudapi.cn/",
		ActiveDirectoryEndpoint:      "https://login.chinacloudapi.cn/?api-version=1.0",
		GalleryEndpoint:              "https://gallery.chinacloudapi.cn/",
		KeyVaultEndpoint:             "https://vault.azure.cn/",
		GraphEndpoint:                "https://graph.chinacloudapi.cn/",
		StorageEndpointSuffix:        "core.chinacloudapi.cn",
		SQLDatabaseDNSSuffix:         "database.chinacloudapi.cn",
		TrafficManagerDNSSuffix:      "trafficmanager.cn",
		KeyVaultDNSSuffix:            "vault.azure.cn",
		ServiceBusEndpointSuffix:     "servicebus.chinacloudapi.net",
		ServiceManagementVMDNSSuffix: "chinacloudapp.cn",
		ResourceManagerVMDNSSuffix:   "cloudapp.azure.cn",
	}

	// GermanCloud is the cloud environment operated in Germany
	GermanCloud = Environment{
		Name:                         "AzureGermanCloud",
		ManagementPortalURL:          "http://portal.microsoftazure.de/",
		PublishSettingsURL:           "https://manage.microsoftazure.de/publishsettings/index",
		ServiceManagementEndpoint:    "https://management.core.cloudapi.de/",
		ResourceManagerEndpoint:      "https://management.microsoftazure.de/",
		ActiveDirectoryEndpoint:      "https://login.microsoftonline.de/",
		GalleryEndpoint:              "https://gallery.cloudapi.de/",
		KeyVaultEndpoint:             "https://vault.microsoftazure.de/",
		GraphEndpoint:                "https://graph.cloudapi.de/",
		StorageEndpointSuffix:        "core.cloudapi.de",
		SQLDatabaseDNSSuffix:         "database.cloudapi.de",
		TrafficManagerDNSSuffix:      "azuretrafficmanager.de",
		KeyVaultDNSSuffix:            "vault.microsoftazure.de",
		ServiceBusEndpointSuffix:     "servicebus.cloudapi.de",
		ServiceManagementVMDNSSuffix: "azurecloudapp.de",
		ResourceManagerVMDNSSuffix:   "cloudapp.microsoftazure.de",
	}
)

// EnvironmentFromName returns an Environment based on the common name specified
func EnvironmentFromName(name string) (Environment, error) {
	name = strings.ToUpper(name)
	env, ok := environments[name]
	if !ok {
		return env, fmt.Errorf("autorest/azure: There is no cloud environment matching the name %q", name)
	}
	return env, nil
}

// OAuthConfigForTenant returns an OAuthConfig with tenant specific urls
func (env Environment) OAuthConfigForTenant(tenantID string) (*OAuthConfig, error) {
	return OAuthConfigForTenant(env.ActiveDirectoryEndpoint, tenantID)
}

// OAuthConfigForTenant returns an OAuthConfig with tenant specific urls for target cloud auth endpoint
func OAuthConfigForTenant(activeDirectoryEndpoint, tenantID string) (*OAuthConfig, error) {
	template := "%s/oauth2/%s?api-version=%s"
	u, err := url.Parse(activeDirectoryEndpoint)
	if err != nil {
		return nil, err
	}
	authorizeURL, err := u.Parse(fmt.Sprintf(template, tenantID, "authorize", activeDirectoryAPIVersion))
	if err != nil {
		return nil, err
	}
	tokenURL, err := u.Parse(fmt.Sprintf(template, tenantID, "token", activeDirectoryAPIVersion))
	if err != nil {
		return nil, err
	}
	deviceCodeURL, err := u.Parse(fmt.Sprintf(template, tenantID, "devicecode", activeDirectoryAPIVersion))
	if err != nil {
		return nil, err
	}

	return &OAuthConfig{
		AuthorizeEndpoint:  *authorizeURL,
		TokenEndpoint:      *tokenURL,
		DeviceCodeEndpoint: *deviceCodeURL,
	}, nil
}
