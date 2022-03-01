/*
Copyright 2021 The Kubernetes Authors.

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

package spec3

import (
	"encoding/json"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"github.com/go-openapi/swag"
)

// SecurityScheme defines reusable Security Scheme Object, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#securitySchemeObject
type SecurityScheme struct {
	spec.Refable
	SecuritySchemeProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode SecurityScheme as JSON
func (s *SecurityScheme) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(s.SecuritySchemeProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(s.VendorExtensible)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(s.Refable)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (s *SecurityScheme) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &s.SecuritySchemeProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &s.VendorExtensible); err != nil {
		return err
	}
	return json.Unmarshal(data, &s.Refable)
}

// SecuritySchemeProps defines a security scheme that can be used by the operations
type SecuritySchemeProps struct {
	// Type of the security scheme
	Type string `json:"type,omitempty"`
	// Description holds a short description for security scheme
	Description string `json:"description,omitempty"`
	// Name holds the name of the header, query or cookie parameter to be used
	Name string `json:"name,omitempty"`
	// In holds the location of the API key
	In string `json:"in,omitempty"`
	// Scheme holds the name of the HTTP Authorization scheme to be used in the Authorization header
	Scheme string `json:"scheme,omitempty"`
	// BearerFormat holds a hint to the client to identify how the bearer token is formatted
	BearerFormat string `json:"bearerFormat,omitempty"`
	// Flows contains configuration information for the flow types supported.
	Flows map[string]*OAuthFlow `json:"flows,omitempty"`
	// OpenIdConnectUrl holds an url to discover OAuth2 configuration values from
	OpenIdConnectUrl string `json:"openIdConnectUrl,omitempty"`
}

// OAuthFlow contains configuration information for the flow types supported.
type OAuthFlow struct {
	OAuthFlowProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode OAuthFlow as JSON
func (o *OAuthFlow) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(o.OAuthFlowProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(o.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (o *OAuthFlow) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &o.OAuthFlowProps); err != nil {
		return err
	}
	return json.Unmarshal(data, &o.VendorExtensible)
}

// OAuthFlowProps holds configuration details for a supported OAuth Flow
type OAuthFlowProps struct {
	// AuthorizationUrl hold the authorization URL to be used for this flow
	AuthorizationUrl string `json:"authorizationUrl,omitempty"`
	// TokenUrl holds the token URL to be used for this flow
	TokenUrl string `json:"tokenUrl,omitempty"`
	// RefreshUrl holds the URL to be used for obtaining refresh tokens
	RefreshUrl string `json:"refreshUrl,omitempty"`
	// Scopes holds the available scopes for the OAuth2 security scheme
	Scopes map[string]string `json:"scopes,omitempty"`
}
