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
	"net/url"
)

const (
	activeDirectoryAPIVersion = "1.0"
)

// OAuthConfig represents the endpoints needed
// in OAuth operations
type OAuthConfig struct {
	AuthorityEndpoint  url.URL
	AuthorizeEndpoint  url.URL
	TokenEndpoint      url.URL
	DeviceCodeEndpoint url.URL
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
	if err := validateStringParam(activeDirectoryEndpoint, "activeDirectoryEndpoint"); err != nil {
		return nil, err
	}
	// it's legal for tenantID to be empty so don't validate it
	const activeDirectoryEndpointTemplate = "%s/oauth2/%s?api-version=%s"
	u, err := url.Parse(activeDirectoryEndpoint)
	if err != nil {
		return nil, err
	}
	authorityURL, err := u.Parse(tenantID)
	if err != nil {
		return nil, err
	}
	authorizeURL, err := u.Parse(fmt.Sprintf(activeDirectoryEndpointTemplate, tenantID, "authorize", activeDirectoryAPIVersion))
	if err != nil {
		return nil, err
	}
	tokenURL, err := u.Parse(fmt.Sprintf(activeDirectoryEndpointTemplate, tenantID, "token", activeDirectoryAPIVersion))
	if err != nil {
		return nil, err
	}
	deviceCodeURL, err := u.Parse(fmt.Sprintf(activeDirectoryEndpointTemplate, tenantID, "devicecode", activeDirectoryAPIVersion))
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
