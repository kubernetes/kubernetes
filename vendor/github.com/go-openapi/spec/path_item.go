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

// PathItemProps the path item specific properties
type PathItemProps struct {
	Get        *Operation  `json:"get,omitempty"`
	Put        *Operation  `json:"put,omitempty"`
	Post       *Operation  `json:"post,omitempty"`
	Delete     *Operation  `json:"delete,omitempty"`
	Options    *Operation  `json:"options,omitempty"`
	Head       *Operation  `json:"head,omitempty"`
	Patch      *Operation  `json:"patch,omitempty"`
	Parameters []Parameter `json:"parameters,omitempty"`
}

// PathItem describes the operations available on a single path.
// A Path Item may be empty, due to [ACL constraints](http://goo.gl/8us55a#securityFiltering).
// The path itself is still exposed to the documentation viewer but they will
// not know which operations and parameters are available.
//
// For more information: http://goo.gl/8us55a#pathItemObject
type PathItem struct {
	Refable
	VendorExtensible
	PathItemProps
}

// JSONLookup look up a value by the json property name
func (p PathItem) JSONLookup(token string) (interface{}, error) {
	if ex, ok := p.Extensions[token]; ok {
		return &ex, nil
	}
	if token == jsonRef {
		return &p.Ref, nil
	}
	r, _, err := jsonpointer.GetForToken(p.PathItemProps, token)
	return r, err
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (p *PathItem) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &p.Refable); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &p.VendorExtensible); err != nil {
		return err
	}
	return json.Unmarshal(data, &p.PathItemProps)
}

// MarshalJSON converts this items object to JSON
func (p PathItem) MarshalJSON() ([]byte, error) {
	b3, err := json.Marshal(p.Refable)
	if err != nil {
		return nil, err
	}
	b4, err := json.Marshal(p.VendorExtensible)
	if err != nil {
		return nil, err
	}
	b5, err := json.Marshal(p.PathItemProps)
	if err != nil {
		return nil, err
	}
	concated := swag.ConcatJSON(b3, b4, b5)
	return concated, nil
}
