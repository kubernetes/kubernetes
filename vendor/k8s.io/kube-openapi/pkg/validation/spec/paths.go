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
	"fmt"
	"strings"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
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
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, p)
	}

	var res map[string]json.RawMessage
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}
	for k, v := range res {
		if strings.HasPrefix(strings.ToLower(k), "x-") {
			if p.Extensions == nil {
				p.Extensions = make(map[string]interface{})
			}
			var d interface{}
			if err := json.Unmarshal(v, &d); err != nil {
				return err
			}
			p.Extensions[k] = d
		}
		if strings.HasPrefix(k, "/") {
			if p.Paths == nil {
				p.Paths = make(map[string]PathItem)
			}
			var pi PathItem
			if err := json.Unmarshal(v, &pi); err != nil {
				return err
			}
			p.Paths[k] = pi
		}
	}
	return nil
}

func (p *Paths) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
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
			case isExtensionKey(k):
				ext = nil
				if err := opts.UnmarshalNext(dec, &ext); err != nil {
					return err
				}

				if p.Extensions == nil {
					p.Extensions = make(map[string]any)
				}
				p.Extensions[k] = ext
			case len(k) > 0 && k[0] == '/':
				pi = PathItem{}
				if err := opts.UnmarshalNext(dec, &pi); err != nil {
					return err
				}

				if p.Paths == nil {
					p.Paths = make(map[string]PathItem)
				}
				p.Paths[k] = pi
			default:
				_, err := dec.ReadValue() // skip value
				return err
			}
		}
	default:
		return fmt.Errorf("unknown JSON kind: %v", k)
	}
}

// MarshalJSON converts this items object to JSON
func (p Paths) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(p.VendorExtensible)
	if err != nil {
		return nil, err
	}

	pths := make(map[string]PathItem)
	for k, v := range p.Paths {
		if strings.HasPrefix(k, "/") {
			pths[k] = v
		}
	}
	b2, err := json.Marshal(pths)
	if err != nil {
		return nil, err
	}
	concated := swag.ConcatJSON(b1, b2)
	return concated, nil
}
