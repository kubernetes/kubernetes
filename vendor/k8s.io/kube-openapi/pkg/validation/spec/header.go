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
)

const (
	jsonArray = "array"
)

// HeaderProps describes a response header
type HeaderProps struct {
	Description string `json:"description,omitempty"`
}

// Header describes a header for a response of the API
//
// For more information: http://goo.gl/8us55a#headerObject
type Header struct {
	CommonValidations
	SimpleSchema
	VendorExtensible
	HeaderProps
}

// MarshalJSON marshal this to JSON
func (h Header) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(h.CommonValidations)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(h.SimpleSchema)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(h.HeaderProps)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

// UnmarshalJSON unmarshals this header from JSON
func (h *Header) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &h.CommonValidations); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &h.SimpleSchema); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &h.VendorExtensible); err != nil {
		return err
	}
	return json.Unmarshal(data, &h.HeaderProps)
}
