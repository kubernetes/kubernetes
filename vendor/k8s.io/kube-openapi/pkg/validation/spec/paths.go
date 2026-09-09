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
	"fmt"
	"strings"

	"k8s.io/kube-openapi/pkg/internal"
)

// Paths holds the relative paths to the individual endpoints.
// The path is appended to the [`basePath`](http://goo.gl/8us55a#swaggerBasePath) in order
// to construct the full URL.
// The Paths may be empty, due to [ACL constraints](http://goo.gl/8us55a#securityFiltering).
//
// For more information: http://goo.gl/8us55a#pathsObject
type Paths struct {
	VendorExtensible
	Paths map[string]PathItem `json:"-"` // custom serializer to flatten this, each entry must start with "/"
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (p *Paths) UnmarshalJSON(data []byte) error {
	return jsonv2.Unmarshal(data, p)
}

func (p *Paths) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	tok, err := dec.ReadToken()
	if err != nil {
		return err
	}
	var ext any
	var pi PathItem
	switch k := tok.Kind(); k {
	case 'n':
		return nil // noop
	case '{':
		for {
			tok, err := dec.ReadToken()
			if err != nil {
				return err
			}

			if tok.Kind() == '}' {
				return nil
			}

			switch k := tok.String(); {
			case internal.IsExtensionKey(k):
				ext = nil
				if err := jsonv2.UnmarshalDecode(dec, &ext); err != nil {
					return err
				}

				if p.Extensions == nil {
					p.Extensions = make(map[string]any)
				}
				p.Extensions[k] = ext
			case len(k) > 0 && k[0] == '/':
				pi = PathItem{}
				if err := jsonv2.UnmarshalDecode(dec, &pi); err != nil {
					return err
				}

				if p.Paths == nil {
					p.Paths = make(map[string]PathItem)
				}
				p.Paths[k] = pi
			default:
				_, err := dec.ReadValue() // skip value
				if err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("unknown JSON kind: %v", k)
	}
}

// MarshalJSON converts this items object to JSON
func (p Paths) MarshalJSON() ([]byte, error) {
	return internal.DeterministicMarshal(p)
}

func (p Paths) MarshalJSONTo(enc *jsontext.Encoder) error {
	m := make(map[string]any, len(p.Extensions)+len(p.Paths))
	for k, v := range p.Extensions {
		if internal.IsExtensionKey(k) {
			m[k] = v
		}
	}
	for k, v := range p.Paths {
		if strings.HasPrefix(k, "/") {
			m[k] = v
		}
	}
	return jsonv2.MarshalEncode(enc, m)
}
