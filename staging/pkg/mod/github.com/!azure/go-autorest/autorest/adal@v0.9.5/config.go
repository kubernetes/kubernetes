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
	"errors"
	"fmt"
	"net/url"
)

const (
	activeDirectoryEndpointTemplate = "%s/oauth2/%s%s"
)

// OAuthConfig represents the endpoints needed
// in OAuth operations
type OAuthConfig struct {
	AuthorityEndpoint  url.URL `json:"authorityEndpoint"`
	AuthorizeEndpoint  url.URL `json:"authorizeEndpoint"`
	TokenEndpoint      url.URL `json:"tokenEndpoint"`
	DeviceCodeEndpoint url.URL `json:"deviceCodeEndpoint"`
}

// IsZero returns true if the OAuthConfig object is zero-initialized.
func (oac OAuthConfig) IsZero() bool {
	return oac == OAuthConfig{}
}

func validateStringParam(param, name string) error {
	if len(param) == 0 {
		return fmt.Errorf("parameter '" + name + "' cannot be empty")
	}
	return nil
}

// NewOAuthConfig returns an OAuthConfig with tenant specific urls
func NewOAuthConfig(activeDirectoryEndpoint, tenantID string) (*OAuthConfig, error) {
	apiVer := "1.0"
	return NewOAuthConfigWithAPIVersion(activeDirectoryEndpoint, tenantID, &apiVer)
}

// NewOAuthConfigWithAPIVersion returns an OAuthConfig with tenant specific urls.
// If apiVersion is not nil the "api-version" query parameter will be appended to the endpoint URLs with the specified value.
func NewOAuthConfigWithAPIVersion(activeDirectoryEndpoint, tenantID string, apiVersion *string) (*OAuthConfig, error) {
	if err := validateStringParam(activeDirectoryEndpoint, "activeDirectoryEndpoint"); err != nil {
		return nil, err
	}
	api := ""
	// it's legal for tenantID to be empty so don't validate it
	if apiVersion != nil {
		if err := validateStringParam(*apiVersion, "apiVersion"); err != nil {
			return nil, err
		}
		api = fmt.Sprintf("?api-version=%s", *apiVersion)
	}
	u, err := url.Parse(activeDirectoryEndpoint)
	if err != nil {
		return nil, err
	}
	authorityURL, err := u.Parse(tenantID)
	if err != nil {
		return nil, err
	}
	authorizeURL, err := u.Parse(fmt.Sprintf(activeDirectoryEndpointTemplate, tenantID, "authorize", api))
	if err != nil {
		return nil, err
	}
	tokenURL, err := u.Parse(fmt.Sprintf(activeDirectoryEndpointTemplate, tenantID, "token", api))
	if err != nil {
		return nil, err
	}
	deviceCodeURL, err := u.Parse(fmt.Sprintf(activeDirectoryEndpointTemplate, tenantID, "devicecode", api))
	if err != nil {
		return nil, err
	}

	return &OAuthConfig{
		AuthorityEndpoint:  *authorityURL,
		AuthorizeEndpoint:  *authorizeURL,
		TokenEndpoint:      *tokenURL,
		DeviceCodeEndpoint: *deviceCodeURL,
	}, nil
}

// MultiTenantOAuthConfig provides endpoints for primary and aulixiary tenant IDs.
type MultiTenantOAuthConfig interface {
	PrimaryTenant() *OAuthConfig
	AuxiliaryTenants() []*OAuthConfig
}

// OAuthOptions contains optional OAuthConfig creation arguments.
type OAuthOptions struct {
	APIVersion string
}

func (c OAuthOptions) apiVersion() string {
	if c.APIVersion != "" {
		return fmt.Sprintf("?api-version=%s", c.APIVersion)
	}
	return "1.0"
}

// NewMultiTenantOAuthConfig creates an object that support multitenant OAuth configuration.
// See https://docs.microsoft.com/en-us/azure/azure-resource-manager/authenticate-multi-tenant for more information.
func NewMultiTenantOAuthConfig(activeDirectoryEndpoint, primaryTenantID string, auxiliaryTenantIDs []string, options OAuthOptions) (MultiTenantOAuthConfig, error) {
	if len(auxiliaryTenantIDs) == 0 || len(auxiliaryTenantIDs) > 3 {
		return nil, errors.New("must specify one to three auxiliary tenants")
	}
	mtCfg := multiTenantOAuthConfig{
		cfgs: make([]*OAuthConfig, len(auxiliaryTenantIDs)+1),
	}
	apiVer := options.apiVersion()
	pri, err := NewOAuthConfigWithAPIVersion(activeDirectoryEndpoint, primaryTenantID, &apiVer)
	if err != nil {
		return nil, fmt.Errorf("failed to create OAuthConfig for primary tenant: %v", err)
	}
	mtCfg.cfgs[0] = pri
	for i := range auxiliaryTenantIDs {
		aux, err := NewOAuthConfig(activeDirectoryEndpoint, auxiliaryTenantIDs[i])
		if err != nil {
			return nil, fmt.Errorf("failed to create OAuthConfig for tenant '%s': %v", auxiliaryTenantIDs[i], err)
		}
		mtCfg.cfgs[i+1] = aux
	}
	return mtCfg, nil
}

type multiTenantOAuthConfig struct {
	// first config in the slice is the primary tenant
	cfgs []*OAuthConfig
}

func (m multiTenantOAuthConfig) PrimaryTenant() *OAuthConfig {
	return m.cfgs[0]
}

func (m multiTenantOAuthConfig) AuxiliaryTenants() []*OAuthConfig {
	return m.cfgs[1:]
}
