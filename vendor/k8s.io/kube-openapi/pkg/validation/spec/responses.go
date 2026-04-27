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
	"reflect"
	"strconv"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
)

// Responses is a container for the expected responses of an operation.
// The container maps a HTTP response code to the expected response.
// It is not expected from the documentation to necessarily cover all possible HTTP response codes,
// since they may not be known in advance. However, it is expected from the documentation to cover
// a successful operation response and any known errors.
//
// The `default` can be used a default response object for all HTTP codes that are not covered
// individually by the specification.
//
// The `Responses Object` MUST contain at least one response code, and it SHOULD be the response
// for a successful operation call.
//
// For more information: http://goo.gl/8us55a#responsesObject
type Responses struct {
	VendorExtensible
	ResponsesProps
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (r *Responses) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, r)
	}

	if err := json.Unmarshal(data, &r.ResponsesProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.VendorExtensible); err != nil {
		return err
	}
	if reflect.DeepEqual(ResponsesProps{}, r.ResponsesProps) {
		r.ResponsesProps = ResponsesProps{}
	}
	return nil
}

// MarshalJSON converts this items object to JSON
func (r Responses) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(r)
	}
	b1, err := json.Marshal(r.ResponsesProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(r.VendorExtensible)
	if err != nil {
		return nil, err
	}
	concated := swag.ConcatJSON(b1, b2)
	return concated, nil
}

func (r Responses) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	type ArbitraryKeys map[string]interface{}
	var x struct {
		ArbitraryKeys
		Default *Response `json:"default,omitempty"`
	}
	x.ArbitraryKeys = make(map[string]any, len(r.Extensions)+len(r.StatusCodeResponses))
	for k, v := range r.Extensions {
		if internal.IsExtensionKey(k) {
			x.ArbitraryKeys[k] = v
		}
	}
	for k, v := range r.StatusCodeResponses {
		x.ArbitraryKeys[strconv.Itoa(k)] = v
	}
	x.Default = r.Default
	return opts.MarshalNext(enc, x)
}

// ResponsesProps describes all responses for an operation.
// It tells what is the default response and maps all responses with a
// HTTP status code.
type ResponsesProps struct {
	Default             *Response
	StatusCodeResponses map[int]Response
}

// MarshalJSON marshals responses as JSON
func (r ResponsesProps) MarshalJSON() ([]byte, error) {
	toser := map[string]Response{}
	if r.Default != nil {
		toser["default"] = *r.Default
	}
	for k, v := range r.StatusCodeResponses {
		toser[strconv.Itoa(k)] = v
	}
	return json.Marshal(toser)
}

// UnmarshalJSON unmarshals responses from JSON
func (r *ResponsesProps) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, r)
	}
	var res map[string]json.RawMessage
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}
	if v, ok := res["default"]; ok {
		value := Response{}
		if err := json.Unmarshal(v, &value); err != nil {
			return err
		}
		r.Default = &value
		delete(res, "default")
	}
	for k, v := range res {
		// Take all integral keys
		if nk, err := strconv.Atoi(k); err == nil {
			if r.StatusCodeResponses == nil {
				r.StatusCodeResponses = map[int]Response{}
			}
			value := Response{}
			if err := json.Unmarshal(v, &value); err != nil {
				return err
			}
			r.StatusCodeResponses[nk] = value
		}
	}
	return nil
}

func (r *Responses) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) (err error) {
	tok, err := dec.ReadToken()
	if err != nil {
		return err
	}
	var ext any
	var resp Response
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
				if err := opts.UnmarshalNext(dec, &ext); err != nil {
					return err
				}

				if r.Extensions == nil {
					r.Extensions = make(map[string]any)
				}
				r.Extensions[k] = ext
			case k == "default":
				resp = Response{}
				if err := opts.UnmarshalNext(dec, &resp); err != nil {
					return err
				}

				respCopy := resp
				r.ResponsesProps.Default = &respCopy
			default:
				if nk, err := strconv.Atoi(k); err == nil {
					resp = Response{}
					if err := opts.UnmarshalNext(dec, &resp); err != nil {
						return err
					}

					if r.StatusCodeResponses == nil {
						r.StatusCodeResponses = map[int]Response{}
					}
					r.StatusCodeResponses[nk] = resp
				}
			}
		}
	default:
		return fmt.Errorf("unknown JSON kind: %v", k)
	}
}
