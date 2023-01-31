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

	r.Extensions = x.Extensions
	r.ResponseProps = x.ResponseProps

	if err := r.Refable.Ref.fromMap(r.Extensions); err != nil {
		return err
	}

	r.Extensions.sanitize()
	if len(r.Extensions) == 0 {
		r.Extensions = nil
	}

	return nil
}

// MarshalJSON converts this items object to JSON
func (r Response) MarshalJSON() ([]byte, error) {
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
