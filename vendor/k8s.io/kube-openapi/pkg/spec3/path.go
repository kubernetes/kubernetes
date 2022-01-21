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
	"strings"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"github.com/go-openapi/swag"
)

// Paths describes the available paths and operations for the API, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#pathsObject
type Paths struct {
	Paths map[string]*Path
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Paths as JSON
func (p *Paths) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(p.Paths)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(p.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (p *Paths) UnmarshalJSON(data []byte) error {
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
				p.Paths = make(map[string]*Path)
			}
			var pi *Path
			if err := json.Unmarshal(v, &pi); err != nil {
				return err
			}
			p.Paths[k] = pi
		}
	}
	return nil
}

// Path describes the operations available on a single path, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#pathItemObject
//
// Note that this struct is actually a thin wrapper around PathProps to make it referable and extensible
type Path struct {
	spec.Refable
	PathProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Path as JSON
func (p *Path) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(p.Refable)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(p.PathProps)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(p.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

func (p *Path) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &p.Refable); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &p.PathProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &p.VendorExtensible); err != nil {
		return err
	}
	return nil
}

// PathProps describes the operations available on a single path, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#pathItemObject
type PathProps struct {
	// Summary holds a summary for all operations in this path
	Summary string `json:"summary,omitempty"`
	// Description holds a description for all operations in this path
	Description string `json:"description,omitempty"`
	// Get defines GET operation
	Get *Operation `json:"get,omitempty"`
	// Put defines PUT operation
	Put *Operation `json:"put,omitempty"`
	// Post defines POST operation
	Post *Operation `json:"post,omitempty"`
	// Delete defines DELETE operation
	Delete *Operation `json:"delete,omitempty"`
	// Options defines OPTIONS operation
	Options *Operation `json:"options,omitempty"`
	// Head defines HEAD operation
	Head *Operation `json:"head,omitempty"`
	// Patch defines PATCH operation
	Patch *Operation `json:"patch,omitempty"`
	// Trace defines TRACE operation
	Trace *Operation `json:"trace,omitempty"`
	// Servers is an alternative server array to service all operations in this path
	Servers []*Server `json:"servers,omitempty"`
	// Parameters a list of parameters that are applicable for this operation
	Parameters []*Parameter `json:"parameters,omitempty"`
}
