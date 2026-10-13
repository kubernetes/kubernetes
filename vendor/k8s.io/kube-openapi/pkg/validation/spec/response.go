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
	return jsonv2.Unmarshal(data, r)
}

func (r *Response) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var x struct {
		ResponseProps
		Extensions Extensions `json:",embed"`
	}

	if err := jsonv2.UnmarshalDecode(dec, &x); err != nil {
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
	return internal.DeterministicMarshal(r)
}

func (r Response) MarshalJSONTo(enc *jsontext.Encoder) error {
	var x struct {
		Ref           string                `json:"$ref,omitempty"`
		Extensions    Extensions            `json:",embed"`
		ResponseProps responsePropsOmitZero `json:",embed"`
	}
	x.Ref = r.Refable.Ref.String()
	x.Extensions = internal.SanitizeExtensions(r.Extensions)
	x.ResponseProps = responsePropsOmitZero(r.ResponseProps)
	return jsonv2.MarshalEncode(enc, x)
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
