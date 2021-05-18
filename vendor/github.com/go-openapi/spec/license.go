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

// License information for the exposed API.
//
// For more information: http://goo.gl/8us55a#licenseObject
type License struct {
	LicenseProps
	VendorExtensible
}

type LicenseProps struct {
	Name string `json:"name,omitempty"`
	URL  string `json:"url,omitempty"`
}

func (l *License) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &l.LicenseProps); err != nil {
		return err
	}
	return json.Unmarshal(data, &l.VendorExtensible)
}

func (l License) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(l.LicenseProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(l.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}
