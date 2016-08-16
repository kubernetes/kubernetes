// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

import (
	"encoding/json"

	"github.com/go-openapi/jsonpointer"
	"github.com/go-openapi/swag"
)

const (
	basic       = "basic"
	apiKey      = "apiKey"
	oauth2      = "oauth2"
	implicit    = "implicit"
	password    = "password"
	application = "application"
	accessCode  = "accessCode"
)

// BasicAuth creates a basic auth security scheme
func BasicAuth() *SecurityScheme {
	return &SecurityScheme{SecuritySchemeProps: SecuritySchemeProps{Type: basic}}
}

// APIKeyAuth creates an api key auth security scheme
func APIKeyAuth(fieldName, valueSource string) *SecurityScheme {
	return &SecurityScheme{SecuritySchemeProps: SecuritySchemeProps{Type: apiKey, Name: fieldName, In: valueSource}}
}

// OAuth2Implicit creates an implicit flow oauth2 security scheme
func OAuth2Implicit(authorizationURL string) *SecurityScheme {
	return &SecurityScheme{SecuritySchemeProps: SecuritySchemeProps{
		Type:             oauth2,
		Flow:             implicit,
		AuthorizationURL: authorizationURL,
	}}
}

// OAuth2Password creates a password flow oauth2 security scheme
func OAuth2Password(tokenURL string) *SecurityScheme {
	return &SecurityScheme{SecuritySchemeProps: SecuritySchemeProps{
		Type:     oauth2,
		Flow:     password,
		TokenURL: tokenURL,
	}}
}

// OAuth2Application creates an application flow oauth2 security scheme
func OAuth2Application(tokenURL string) *SecurityScheme {
	return &SecurityScheme{SecuritySchemeProps: SecuritySchemeProps{
		Type:     oauth2,
		Flow:     application,
		TokenURL: tokenURL,
	}}
}

// OAuth2AccessToken creates an access token flow oauth2 security scheme
func OAuth2AccessToken(authorizationURL, tokenURL string) *SecurityScheme {
	return &SecurityScheme{SecuritySchemeProps: SecuritySchemeProps{
		Type:             oauth2,
		Flow:             accessCode,
		AuthorizationURL: authorizationURL,
		TokenURL:         tokenURL,
	}}
}

type SecuritySchemeProps struct {
	Description      string            `json:"description,omitempty"`
	Type             string            `json:"type"`
	Name             string            `json:"name,omitempty"`             // api key
	In               string            `json:"in,omitempty"`               // api key
	Flow             string            `json:"flow,omitempty"`             // oauth2
	AuthorizationURL string            `json:"authorizationUrl,omitempty"` // oauth2
	TokenURL         string            `json:"tokenUrl,omitempty"`         // oauth2
	Scopes           map[string]string `json:"scopes,omitempty"`           // oauth2
}

// AddScope adds a scope to this security scheme
func (s *SecuritySchemeProps) AddScope(scope, description string) {
	if s.Scopes == nil {
		s.Scopes = make(map[string]string)
	}
	s.Scopes[scope] = description
}

// SecurityScheme allows the definition of a security scheme that can be used by the operations.
// Supported schemes are basic authentication, an API key (either as a header or as a query parameter)
// and OAuth2's common flows (implicit, password, application and access code).
//
// For more information: http://goo.gl/8us55a#securitySchemeObject
type SecurityScheme struct {
	VendorExtensible
	SecuritySchemeProps
}

// JSONLookup implements an interface to customize json pointer lookup
func (s SecurityScheme) JSONLookup(token string) (interface{}, error) {
	if ex, ok := s.Extensions[token]; ok {
		return &ex, nil
	}

	r, _, err := jsonpointer.GetForToken(s.SecuritySchemeProps, token)
	return r, err
}

// MarshalJSON marshal this to JSON
func (s SecurityScheme) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(s.SecuritySchemeProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(s.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// UnmarshalJSON marshal this from JSON
func (s *SecurityScheme) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &s.SecuritySchemeProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &s.VendorExtensible); err != nil {
		return err
	}
	return nil
}
