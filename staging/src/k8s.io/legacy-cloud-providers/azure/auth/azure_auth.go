/*
Copyright 2017 The Kubernetes Authors.

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
	"crypto/rsa"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"strings"

	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/azure"
	"golang.org/x/crypto/pkcs12"

	"k8s.io/klog/v2"
)

const (
	// ADFSIdentitySystem is the override value for tenantID on Azure Stack clouds.
	ADFSIdentitySystem = "adfs"
)

var (
	// ErrorNoAuth indicates that no credentials are provided.
	ErrorNoAuth = fmt.Errorf("no credentials provided for Azure cloud provider")
)

// AzureAuthConfig holds auth related part of cloud config
type AzureAuthConfig struct {
	// The cloud environment identifier. Takes values from https://github.com/Azure/go-autorest/blob/ec5f4903f77ed9927ac95b19ab8e44ada64c1356/autorest/azure/environments.go#L13
	Cloud string `json:"cloud,omitempty" yaml:"cloud,omitempty"`
	// The AAD Tenant ID for the Subscription that the cluster is deployed in
	TenantID string `json:"tenantId,omitempty" yaml:"tenantId,omitempty"`
	// The ClientID for an AAD application with RBAC access to talk to Azure RM APIs
	AADClientID string `json:"aadClientId,omitempty" yaml:"aadClientId,omitempty"`
	// The ClientSecret for an AAD application with RBAC access to talk to Azure RM APIs
	AADClientSecret string `json:"aadClientSecret,omitempty" yaml:"aadClientSecret,omitempty"`
	// The path of a client certificate for an AAD application with RBAC access to talk to Azure RM APIs
	AADClientCertPath string `json:"aadClientCertPath,omitempty" yaml:"aadClientCertPath,omitempty"`
	// The password of the client certificate for an AAD application with RBAC access to talk to Azure RM APIs
	AADClientCertPassword string `json:"aadClientCertPassword,omitempty" yaml:"aadClientCertPassword,omitempty"`
	// Use managed service identity for the virtual machine to access Azure ARM APIs
	UseManagedIdentityExtension bool `json:"useManagedIdentityExtension,omitempty" yaml:"useManagedIdentityExtension,omitempty"`
	// UserAssignedIdentityID contains the Client ID of the user assigned MSI which is assigned to the underlying VMs. If empty the user assigned identity is not used.
	// More details of the user assigned identity can be found at: https://docs.microsoft.com/en-us/azure/active-directory/managed-service-identity/overview
	// For the user assigned identity specified here to be used, the UseManagedIdentityExtension has to be set to true.
	UserAssignedIdentityID string `json:"userAssignedIdentityID,omitempty" yaml:"userAssignedIdentityID,omitempty"`
	// The ID of the Azure Subscription that the cluster is deployed in
	SubscriptionID string `json:"subscriptionId,omitempty" yaml:"subscriptionId,omitempty"`
	// IdentitySystem indicates the identity provider. Relevant only to hybrid clouds (Azure Stack).
	// Allowed values are 'azure_ad' (default), 'adfs'.
	IdentitySystem string `json:"identitySystem,omitempty" yaml:"identitySystem,omitempty"`
	// ResourceManagerEndpoint is the cloud's resource manager endpoint. If set, cloud provider queries this endpoint
	// in order to generate an autorest.Environment instance instead of using one of the pre-defined Environments.
	ResourceManagerEndpoint string `json:"resourceManagerEndpoint,omitempty" yaml:"resourceManagerEndpoint,omitempty"`
	// The AAD Tenant ID for the Subscription that the network resources are deployed in
	NetworkResourceTenantID string `json:"networkResourceTenantID,omitempty" yaml:"networkResourceTenantID,omitempty"`
	// The ID of the Azure Subscription that the network resources are deployed in
	NetworkResourceSubscriptionID string `json:"networkResourceSubscriptionID,omitempty" yaml:"networkResourceSubscriptionID,omitempty"`
}

// GetServicePrincipalToken creates a new service principal token based on the configuration.
//
// By default, the cluster and its network resources are deployed in the same AAD Tenant and Subscription,
// and all azure clients use this method to fetch Service Principal Token.
//
// If NetworkResourceTenantID and NetworkResourceSubscriptionID are specified to have different values than TenantID and SubscriptionID, network resources are deployed in different AAD Tenant and Subscription than those for the cluster,
// than only azure clients except VM/VMSS and network resource ones use this method to fetch Token.
// For tokens for VM/VMSS and network resource ones, please check GetMultiTenantServicePrincipalToken and GetNetworkResourceServicePrincipalToken.
func GetServicePrincipalToken(config *AzureAuthConfig, env *azure.Environment) (*adal.ServicePrincipalToken, error) {
	var tenantID string
	if strings.EqualFold(config.IdentitySystem, ADFSIdentitySystem) {
		tenantID = ADFSIdentitySystem
	} else {
		tenantID = config.TenantID
	}

	if config.UseManagedIdentityExtension {
		klog.V(2).Infoln("azure: using managed identity extension to retrieve access token")
		msiEndpoint, err := adal.GetMSIVMEndpoint()
		if err != nil {
			return nil, fmt.Errorf("Getting the managed service identity endpoint: %v", err)
		}
		if len(config.UserAssignedIdentityID) > 0 {
			klog.V(4).Info("azure: using User Assigned MSI ID to retrieve access token")
			return adal.NewServicePrincipalTokenFromMSIWithUserAssignedID(msiEndpoint,
				env.ServiceManagementEndpoint,
				config.UserAssignedIdentityID)
		}
		klog.V(4).Info("azure: using System Assigned MSI to retrieve access token")
		return adal.NewServicePrincipalTokenFromMSI(
			msiEndpoint,
			env.ServiceManagementEndpoint)
	}

	oauthConfig, err := adal.NewOAuthConfigWithAPIVersion(env.ActiveDirectoryEndpoint, tenantID, nil)
	if err != nil {
		return nil, fmt.Errorf("creating the OAuth config: %v", err)
	}

	if len(config.AADClientSecret) > 0 {
		klog.V(2).Infoln("azure: using client_id+client_secret to retrieve access token")
		return adal.NewServicePrincipalToken(
			*oauthConfig,
			config.AADClientID,
			config.AADClientSecret,
			env.ServiceManagementEndpoint)
	}

	if len(config.AADClientCertPath) > 0 && len(config.AADClientCertPassword) > 0 {
		klog.V(2).Infoln("azure: using jwt client_assertion (client_cert+client_private_key) to retrieve access token")
		certData, err := ioutil.ReadFile(config.AADClientCertPath)
		if err != nil {
			return nil, fmt.Errorf("reading the client certificate from file %s: %v", config.AADClientCertPath, err)
		}
		certificate, privateKey, err := decodePkcs12(certData, config.AADClientCertPassword)
		if err != nil {
			return nil, fmt.Errorf("decoding the client certificate: %v", err)
		}
		return adal.NewServicePrincipalTokenFromCertificate(
			*oauthConfig,
			config.AADClientID,
			certificate,
			privateKey,
			env.ServiceManagementEndpoint)
	}

	return nil, ErrorNoAuth
}

// GetMultiTenantServicePrincipalToken is used when (and only when) NetworkResourceTenantID and NetworkResourceSubscriptionID are specified to have different values than TenantID and SubscriptionID.
//
// In that scenario, network resources are deployed in different AAD Tenant and Subscription than those for the cluster,
// and this method creates a new multi-tenant service principal token based on the configuration.
//
// PrimaryToken of the returned multi-tenant token is for the AAD Tenant specified by TenantID, and AuxiliaryToken of the returned multi-tenant token is for the AAD Tenant specified by NetworkResourceTenantID.
//
// Azure VM/VMSS clients use this multi-tenant token, in order to operate those VM/VMSS in AAD Tenant specified by TenantID, and meanwhile in their payload they are referencing network resources (e.g. Load Balancer, Network Security Group, etc.) in AAD Tenant specified by NetworkResourceTenantID.
func GetMultiTenantServicePrincipalToken(config *AzureAuthConfig, env *azure.Environment) (*adal.MultiTenantServicePrincipalToken, error) {
	err := config.checkConfigWhenNetworkResourceInDifferentTenant()
	if err != nil {
		return nil, fmt.Errorf("got error(%v) in getting multi-tenant service principal token", err)
	}

	multiTenantOAuthConfig, err := adal.NewMultiTenantOAuthConfig(
		env.ActiveDirectoryEndpoint, config.TenantID, []string{config.NetworkResourceTenantID}, adal.OAuthOptions{})
	if err != nil {
		return nil, fmt.Errorf("creating the multi-tenant OAuth config: %v", err)
	}

	if len(config.AADClientSecret) > 0 {
		klog.V(2).Infoln("azure: using client_id+client_secret to retrieve multi-tenant access token")
		return adal.NewMultiTenantServicePrincipalToken(
			multiTenantOAuthConfig,
			config.AADClientID,
			config.AADClientSecret,
			env.ServiceManagementEndpoint)
	}

	if len(config.AADClientCertPath) > 0 && len(config.AADClientCertPassword) > 0 {
		return nil, fmt.Errorf("AAD Application client certificate authentication is not supported in getting multi-tenant service principal token")
	}

	return nil, ErrorNoAuth
}

// GetNetworkResourceServicePrincipalToken is used when (and only when) NetworkResourceTenantID and NetworkResourceSubscriptionID are specified to have different values than TenantID and SubscriptionID.
//
// In that scenario, network resources are deployed in different AAD Tenant and Subscription than those for the cluster,
// and this method creates a new service principal token for network resources tenant based on the configuration.
//
// Azure network resource (Load Balancer, Public IP, Route Table, Network Security Group and their sub level resources) clients use this multi-tenant token, in order to operate resources in AAD Tenant specified by NetworkResourceTenantID.
func GetNetworkResourceServicePrincipalToken(config *AzureAuthConfig, env *azure.Environment) (*adal.ServicePrincipalToken, error) {
	err := config.checkConfigWhenNetworkResourceInDifferentTenant()
	if err != nil {
		return nil, fmt.Errorf("got error(%v) in getting network resources service principal token", err)
	}

	oauthConfig, err := adal.NewOAuthConfigWithAPIVersion(env.ActiveDirectoryEndpoint, config.NetworkResourceTenantID, nil)
	if err != nil {
		return nil, fmt.Errorf("creating the OAuth config for network resources tenant: %v", err)
	}

	if len(config.AADClientSecret) > 0 {
		klog.V(2).Infoln("azure: using client_id+client_secret to retrieve access token for network resources tenant")
		return adal.NewServicePrincipalToken(
			*oauthConfig,
			config.AADClientID,
			config.AADClientSecret,
			env.ServiceManagementEndpoint)
	}

	if len(config.AADClientCertPath) > 0 && len(config.AADClientCertPassword) > 0 {
		return nil, fmt.Errorf("AAD Application client certificate authentication is not supported in getting network resources service principal token")
	}

	return nil, ErrorNoAuth
}

// ParseAzureEnvironment returns the azure environment.
// If 'resourceManagerEndpoint' is set, the environment is computed by quering the cloud's resource manager endpoint.
// Otherwise, a pre-defined Environment is looked up by name.
func ParseAzureEnvironment(cloudName, resourceManagerEndpoint, identitySystem string) (*azure.Environment, error) {
	var env azure.Environment
	var err error
	if resourceManagerEndpoint != "" {
		klog.V(4).Infof("Loading environment from resource manager endpoint: %s", resourceManagerEndpoint)
		nameOverride := azure.OverrideProperty{Key: azure.EnvironmentName, Value: cloudName}
		env, err = azure.EnvironmentFromURL(resourceManagerEndpoint, nameOverride)
		if err == nil {
			azureStackOverrides(&env, resourceManagerEndpoint, identitySystem)
		}
	} else if cloudName == "" {
		klog.V(4).Info("Using public cloud environment")
		env = azure.PublicCloud
	} else {
		klog.V(4).Infof("Using %s environment", cloudName)
		env, err = azure.EnvironmentFromName(cloudName)
	}
	return &env, err
}

// UsesNetworkResourceInDifferentTenant determines whether the AzureAuthConfig indicates to use network resources in different AAD Tenant and Subscription than those for the cluster
// Return true only when both NetworkResourceTenantID and NetworkResourceSubscriptionID are specified
// and they are not equals to TenantID and SubscriptionID
func (config *AzureAuthConfig) UsesNetworkResourceInDifferentTenant() bool {
	return len(config.NetworkResourceTenantID) > 0 &&
		len(config.NetworkResourceSubscriptionID) > 0 &&
		!strings.EqualFold(config.NetworkResourceTenantID, config.TenantID) &&
		!strings.EqualFold(config.NetworkResourceSubscriptionID, config.SubscriptionID)
}

// decodePkcs12 decodes a PKCS#12 client certificate by extracting the public certificate and
// the private RSA key
func decodePkcs12(pkcs []byte, password string) (*x509.Certificate, *rsa.PrivateKey, error) {
	privateKey, certificate, err := pkcs12.Decode(pkcs, password)
	if err != nil {
		return nil, nil, fmt.Errorf("decoding the PKCS#12 client certificate: %v", err)
	}
	rsaPrivateKey, isRsaKey := privateKey.(*rsa.PrivateKey)
	if !isRsaKey {
		return nil, nil, fmt.Errorf("PKCS#12 certificate must contain a RSA private key")
	}

	return certificate, rsaPrivateKey, nil
}

// azureStackOverrides ensures that the Environment matches what AKSe currently generates for Azure Stack
func azureStackOverrides(env *azure.Environment, resourceManagerEndpoint, identitySystem string) {
	env.ManagementPortalURL = strings.Replace(resourceManagerEndpoint, "https://management.", "https://portal.", -1)
	env.ServiceManagementEndpoint = env.TokenAudience
	env.ResourceManagerVMDNSSuffix = strings.Replace(resourceManagerEndpoint, "https://management.", "cloudapp.", -1)
	env.ResourceManagerVMDNSSuffix = strings.TrimSuffix(env.ResourceManagerVMDNSSuffix, "/")
	if strings.EqualFold(identitySystem, ADFSIdentitySystem) {
		env.ActiveDirectoryEndpoint = strings.TrimSuffix(env.ActiveDirectoryEndpoint, "/")
		env.ActiveDirectoryEndpoint = strings.TrimSuffix(env.ActiveDirectoryEndpoint, "adfs")
	}
}

// checkConfigWhenNetworkResourceInDifferentTenant checks configuration for the scenario of using network resource in different tenant
func (config *AzureAuthConfig) checkConfigWhenNetworkResourceInDifferentTenant() error {
	if !config.UsesNetworkResourceInDifferentTenant() {
		return fmt.Errorf("NetworkResourceTenantID and NetworkResourceSubscriptionID must be configured")
	}

	if strings.EqualFold(config.IdentitySystem, ADFSIdentitySystem) {
		return fmt.Errorf("ADFS identity system is not supported")
	}

	if config.UseManagedIdentityExtension {
		return fmt.Errorf("managed identity is not supported")
	}

	return nil
}
