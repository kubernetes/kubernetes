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

// TagProps describe a tag entry in the top level tags section of a swagger spec
type TagProps struct {
	Description  string                 `json:"description,omitempty"`
	Name         string                 `json:"name,omitempty"`
	ExternalDocs *ExternalDocumentation `json:"externalDocs,omitempty"`
}

// Tag allows adding meta data to a single tag that is used by the
// [Operation Object](http://goo.gl/8us55a#operationObject).
// It is not mandatory to have a Tag Object per tag used there.
//
// For more information: http://goo.gl/8us55a#tagObject
type Tag struct {
	VendorExtensible
	TagProps
}

// MarshalJSON marshal this to JSON
func (t Tag) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(t.TagProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(t.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// UnmarshalJSON marshal this from JSON
func (t *Tag) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &t.TagProps); err != nil {
		return err
	}
	return json.Unmarshal(data, &t.VendorExtensible)
}
