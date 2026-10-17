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
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	"k8s.io/kube-openapi/pkg/internal"
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

// UnmarshalJSON hydrates this items instance with the data from JSON
func (p *PathItem) UnmarshalJSON(data []byte) error {
	return jsonv2.Unmarshal(data, p)
}

func (p *PathItem) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var x struct {
		Extensions Extensions `json:",embed"`
		PathItemProps
	}

	if err := jsonv2.UnmarshalDecode(dec, &x); err != nil {
		return err
	}
	if err := p.Refable.Ref.fromMap(x.Extensions); err != nil {
		return err
	}
	p.Extensions = internal.SanitizeExtensions(x.Extensions)
	p.PathItemProps = x.PathItemProps

	return nil
}

// MarshalJSON converts this items object to JSON
func (p PathItem) MarshalJSON() ([]byte, error) {
	return internal.DeterministicMarshal(p)
}

func (p PathItem) MarshalJSONTo(enc *jsontext.Encoder) error {
	var x struct {
		Ref        string     `json:"$ref,omitempty"`
		Extensions Extensions `json:",embed"`
		PathItemProps
	}
	x.Ref = p.Refable.Ref.String()
	x.Extensions = internal.SanitizeExtensions(p.Extensions)
	x.PathItemProps = p.PathItemProps
	return jsonv2.MarshalEncode(enc, x)
}
