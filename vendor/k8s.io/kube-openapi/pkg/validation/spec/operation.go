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
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
)

// OperationProps describes an operation
//
// NOTES:
// - schemes, when present must be from [http, https, ws, wss]: see validate
// - Security is handled as a special case: see MarshalJSON function
type OperationProps struct {
	Description  string                 `json:"description,omitempty"`
	Consumes     []string               `json:"consumes,omitempty"`
	Produces     []string               `json:"produces,omitempty"`
	Schemes      []string               `json:"schemes,omitempty"`
	Tags         []string               `json:"tags,omitempty"`
	Summary      string                 `json:"summary,omitempty"`
	ExternalDocs *ExternalDocumentation `json:"externalDocs,omitempty"`
	ID           string                 `json:"operationId,omitempty"`
	Deprecated   bool                   `json:"deprecated,omitempty"`
	Security     []map[string][]string  `json:"security,omitempty"`
	Parameters   []Parameter            `json:"parameters,omitempty"`
	Responses    *Responses             `json:"responses,omitempty"`
}

// Marshaling structure only, always edit along with corresponding
// struct (or compilation will fail).
type operationPropsOmitZero struct {
	Description  string                 `json:"description,omitempty"`
	Consumes     []string               `json:"consumes,omitempty"`
	Produces     []string               `json:"produces,omitempty"`
	Schemes      []string               `json:"schemes,omitempty"`
	Tags         []string               `json:"tags,omitempty"`
	Summary      string                 `json:"summary,omitempty"`
	ExternalDocs *ExternalDocumentation `json:"externalDocs,omitzero"`
	ID           string                 `json:"operationId,omitempty"`
	Deprecated   bool                   `json:"deprecated,omitempty,omitzero"`
	Security     []map[string][]string  `json:"security,omitempty"`
	Parameters   []Parameter            `json:"parameters,omitempty"`
	Responses    *Responses             `json:"responses,omitzero"`
}

// MarshalJSON takes care of serializing operation properties to JSON
//
// We use a custom marhaller here to handle a special cases related to
// the Security field. We need to preserve zero length slice
// while omitting the field when the value is nil/unset.
func (op OperationProps) MarshalJSON() ([]byte, error) {
	type Alias OperationProps
	if op.Security == nil {
		return json.Marshal(&struct {
			Security []map[string][]string `json:"security,omitempty"`
			*Alias
		}{
			Security: op.Security,
			Alias:    (*Alias)(&op),
		})
	}
	return json.Marshal(&struct {
		Security []map[string][]string `json:"security"`
		*Alias
	}{
		Security: op.Security,
		Alias:    (*Alias)(&op),
	})
}

// Operation describes a single API operation on a path.
//
// For more information: http://goo.gl/8us55a#operationObject
type Operation struct {
	VendorExtensible
	OperationProps
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (o *Operation) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, o)
	}

	if err := json.Unmarshal(data, &o.OperationProps); err != nil {
		return err
	}
	return json.Unmarshal(data, &o.VendorExtensible)
}

func (o *Operation) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	type OperationPropsNoMethods OperationProps // strip MarshalJSON method
	var x struct {
		Extensions
		OperationPropsNoMethods
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	o.Extensions = internal.SanitizeExtensions(x.Extensions)
	o.OperationProps = OperationProps(x.OperationPropsNoMethods)
	return nil
}

// MarshalJSON converts this items object to JSON
func (o Operation) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(o)
	}
	b1, err := json.Marshal(o.OperationProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(o.VendorExtensible)
	if err != nil {
		return nil, err
	}
	concated := swag.ConcatJSON(b1, b2)
	return concated, nil
}

func (o Operation) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		Extensions
		OperationProps operationPropsOmitZero `json:",inline"`
	}
	x.Extensions = internal.SanitizeExtensions(o.Extensions)
	x.OperationProps = operationPropsOmitZero(o.OperationProps)
	return opts.MarshalNext(enc, x)
}
