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

	"github.com/go-openapi/swag"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
)

// SecuritySchemeProps describes a swagger security scheme in the securityDefinitions section
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

// SecurityScheme allows the definition of a security scheme that can be used by the operations.
// Supported schemes are basic authentication, an API key (either as a header or as a query parameter)
// and OAuth2's common flows (implicit, password, application and access code).
//
// For more information: http://goo.gl/8us55a#securitySchemeObject
type SecurityScheme struct {
	VendorExtensible
	SecuritySchemeProps
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
	return json.Unmarshal(data, &s.VendorExtensible)
}

func (s *SecurityScheme) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		Extensions
		SecuritySchemeProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	x.Extensions.sanitize()
	if len(x.Extensions) == 0 {
		x.Extensions = nil
	}
	s.VendorExtensible.Extensions = x.Extensions
	s.SecuritySchemeProps = x.SecuritySchemeProps
	return nil
}
