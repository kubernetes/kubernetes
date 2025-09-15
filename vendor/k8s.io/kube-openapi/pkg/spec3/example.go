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

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

// Example https://swagger.io/specification/#example-object

type Example struct {
	spec.Refable
	ExampleProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode RequestBody as JSON
func (e *Example) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshalingV3 {
		return internal.DeterministicMarshal(e)
	}
	b1, err := json.Marshal(e.Refable)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(e.ExampleProps)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(e.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}
func (e *Example) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		Ref          string `json:"$ref,omitempty"`
		ExampleProps `json:",inline"`
		spec.Extensions
	}
	x.Ref = e.Refable.Ref.String()
	x.Extensions = internal.SanitizeExtensions(e.Extensions)
	x.ExampleProps = e.ExampleProps
	return opts.MarshalNext(enc, x)
}

func (e *Example) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, e)
	}
	if err := json.Unmarshal(data, &e.Refable); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &e.ExampleProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &e.VendorExtensible); err != nil {
		return err
	}
	return nil
}

func (e *Example) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		ExampleProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	if err := internal.JSONRefFromMap(&e.Ref.Ref, x.Extensions); err != nil {
		return err
	}
	e.Extensions = internal.SanitizeExtensions(x.Extensions)
	e.ExampleProps = x.ExampleProps

	return nil
}

type ExampleProps struct {
	// Summary holds a short description of the example
	Summary string `json:"summary,omitempty"`
	// Description holds a long description of the example
	Description string `json:"description,omitempty"`
	// Embedded literal example.
	Value interface{} `json:"value,omitempty"`
	// A URL that points to the literal example. This provides the capability to reference examples that cannot easily be included in JSON or YAML documents.
	ExternalValue string `json:"externalValue,omitempty"`
}
