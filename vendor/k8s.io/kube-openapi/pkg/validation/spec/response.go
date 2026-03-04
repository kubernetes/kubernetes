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

// ResponseProps properties specific to a response
type ResponseProps struct {
	Description string                 `json:"description,omitempty"`
	Schema      *Schema                `json:"schema,omitempty"`
	Headers     map[string]Header      `json:"headers,omitempty"`
	Examples    map[string]interface{} `json:"examples,omitempty"`
}

// Marshaling structure only, always edit along with corresponding
// struct (or compilation will fail).
type responsePropsOmitZero struct {
	Description string                 `json:"description,omitempty"`
	Schema      *Schema                `json:"schema,omitzero"`
	Headers     map[string]Header      `json:"headers,omitempty"`
	Examples    map[string]interface{} `json:"examples,omitempty"`
}

// Response describes a single response from an API Operation.
//
// For more information: http://goo.gl/8us55a#responseObject
type Response struct {
	Refable
	ResponseProps
	VendorExtensible
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (r *Response) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, r)
	}

	if err := json.Unmarshal(data, &r.ResponseProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.Refable); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.VendorExtensible); err != nil {
		return err
	}

	return nil
}

func (r *Response) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		ResponseProps
		Extensions
	}

	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}

	if err := r.Refable.Ref.fromMap(x.Extensions); err != nil {
		return err
	}
	r.Extensions = internal.SanitizeExtensions(x.Extensions)
	r.ResponseProps = x.ResponseProps

	return nil
}

// MarshalJSON converts this items object to JSON
func (r Response) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(r)
	}
	b1, err := json.Marshal(r.ResponseProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(r.Refable)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(r.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

func (r Response) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		Ref string `json:"$ref,omitempty"`
		Extensions
		ResponseProps responsePropsOmitZero `json:",inline"`
	}
	x.Ref = r.Refable.Ref.String()
	x.Extensions = internal.SanitizeExtensions(r.Extensions)
	x.ResponseProps = responsePropsOmitZero(r.ResponseProps)
	return opts.MarshalNext(enc, x)
}

// NewResponse creates a new response instance
func NewResponse() *Response {
	return new(Response)
}

// ResponseRef creates a response as a json reference
func ResponseRef(url string) *Response {
	resp := NewResponse()
	resp.Ref = MustCreateRef(url)
	return resp
}
