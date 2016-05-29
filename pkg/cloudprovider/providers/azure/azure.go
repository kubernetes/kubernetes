/*
Copyright 2016 The Kubernetes Authors.

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
	"io"
	"io/ioutil"

	"k8s.io/kubernetes/pkg/cloudprovider"

	"github.com/Azure/azure-sdk-for-go/arm/compute"
	"github.com/Azure/azure-sdk-for-go/arm/network"
	"github.com/Azure/go-autorest/autorest/azure"
	"github.com/ghodss/yaml"
)

// CloudProviderName is the value used for the --cloud-provider flag
const CloudProviderName = "azure"

// Config holds the configuration parsed from the --cloud-config flag
type Config struct {
	Cloud             string `json:"cloud" yaml:"cloud"`
	TenantID          string `json:"tenantId" yaml:"tenantId"`
	SubscriptionID    string `json:"subscriptionId" yaml:"subscriptionId"`
	ResourceGroup     string `json:"resourceGroup" yaml:"resourceGroup"`
	Location          string `json:"location" yaml:"location"`
	VnetName          string `json:"vnetName" yaml:"vnetName"`
	SubnetName        string `json:"subnetName" yaml:"subnetName"`
	SecurityGroupName string `json:"securityGroupName" yaml:"securityGroupName"`
	RouteTableName    string `json:"routeTableName" yaml:"routeTableName"`

	AADClientID     string `json:"aadClientId" yaml:"aadClientId"`
	AADClientSecret string `json:"aadClientSecret" yaml:"aadClientSecret"`
	AADTenantID     string `json:"aadTenantId" yaml:"aadTenantId"`
}

// Cloud holds the config and clients
type Cloud struct {
	Config
	Environment             azure.Environment
	RoutesClient            network.RoutesClient
	SubnetsClient           network.SubnetsClient
	InterfacesClient        network.InterfacesClient
	RouteTablesClient       network.RouteTablesClient
	LoadBalancerClient      network.LoadBalancersClient
	PublicIPAddressesClient network.PublicIPAddressesClient
	SecurityGroupsClient    network.SecurityGroupsClient
	VirtualMachinesClient   compute.VirtualMachinesClient
}

func init() {
	cloudprovider.RegisterCloudProvider(CloudProviderName, NewCloud)
}

// NewCloud returns a Cloud with initialized clients
func NewCloud(configReader io.Reader) (cloudprovider.Interface, error) {
	var az Cloud

	configContents, err := ioutil.ReadAll(configReader)
	if err != nil {
		return nil, err
	}
	err = yaml.Unmarshal(configContents, &az)
	if err != nil {
		return nil, err
	}

	if az.Cloud == "" {
		az.Environment = azure.PublicCloud
	} else {
		az.Environment, err = azure.EnvironmentFromName(az.Cloud)
		if err != nil {
			return nil, err
		}
	}

	oauthConfig, err := az.Environment.OAuthConfigForTenant(az.TenantID)
	if err != nil {
		return nil, err
	}

	servicePrincipalToken, err := azure.NewServicePrincipalToken(
		*oauthConfig,
		az.AADClientID,
		az.AADClientSecret,
		az.Environment.ServiceManagementEndpoint)
	if err != nil {
		return nil, err
	}

	az.SubnetsClient = network.NewSubnetsClient(az.SubscriptionID)
	az.SubnetsClient.BaseURI = az.Environment.ResourceManagerEndpoint
	az.SubnetsClient.Authorizer = servicePrincipalToken

	az.RouteTablesClient = network.NewRouteTablesClient(az.SubscriptionID)
	az.RouteTablesClient.BaseURI = az.Environment.ResourceManagerEndpoint
	az.RouteTablesClient.Authorizer = servicePrincipalToken

	az.RoutesClient = network.NewRoutesClient(az.SubscriptionID)
	az.RoutesClient.BaseURI = az.Environment.ResourceManagerEndpoint
	az.RoutesClient.Authorizer = servicePrincipalToken

	az.InterfacesClient = network.NewInterfacesClient(az.SubscriptionID)
	az.InterfacesClient.BaseURI = az.Environment.ResourceManagerEndpoint
	az.InterfacesClient.Authorizer = servicePrincipalToken

	az.LoadBalancerClient = network.NewLoadBalancersClient(az.SubscriptionID)
	az.LoadBalancerClient.BaseURI = az.Environment.ResourceManagerEndpoint
	az.LoadBalancerClient.Authorizer = servicePrincipalToken

	az.VirtualMachinesClient = compute.NewVirtualMachinesClient(az.SubscriptionID)
	az.VirtualMachinesClient.BaseURI = az.Environment.ResourceManagerEndpoint
	az.VirtualMachinesClient.Authorizer = servicePrincipalToken

	az.PublicIPAddressesClient = network.NewPublicIPAddressesClient(az.SubscriptionID)
	az.PublicIPAddressesClient.BaseURI = az.Environment.ResourceManagerEndpoint
	az.PublicIPAddressesClient.Authorizer = servicePrincipalToken

	az.SecurityGroupsClient = network.NewSecurityGroupsClient(az.SubscriptionID)
	az.SecurityGroupsClient.BaseURI = az.Environment.ResourceManagerEndpoint
	az.SecurityGroupsClient.Authorizer = servicePrincipalToken

	return &az, nil
}

// LoadBalancer returns a balancer interface. Also returns true if the interface is supported, false otherwise.
func (az *Cloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return az, true
}

// Instances returns an instances interface. Also returns true if the interface is supported, false otherwise.
func (az *Cloud) Instances() (cloudprovider.Instances, bool) {
	return az, true
}

// Zones returns a zones interface. Also returns true if the interface is supported, false otherwise.
func (az *Cloud) Zones() (cloudprovider.Zones, bool) {
	return az, true
}

// Clusters returns a clusters interface.  Also returns true if the interface is supported, false otherwise.
func (az *Cloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// Routes returns a routes interface along with whether the interface is supported.
func (az *Cloud) Routes() (cloudprovider.Routes, bool) {
	return az, true
}

// ScrubDNS provides an opportunity for cloud-provider-specific code to process DNS settings for pods.
func (az *Cloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// ProviderName returns the cloud provider ID.
func (az *Cloud) ProviderName() string {
	return CloudProviderName
}
