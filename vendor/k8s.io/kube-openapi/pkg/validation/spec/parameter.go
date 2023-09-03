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

// ParamProps describes the specific attributes of an operation parameter
//
// NOTE:
// - Schema is defined when "in" == "body": see validate
// - AllowEmptyValue is allowed where "in" == "query" || "formData"
type ParamProps struct {
	Description     string  `json:"description,omitempty"`
	Name            string  `json:"name,omitempty"`
	In              string  `json:"in,omitempty"`
	Required        bool    `json:"required,omitempty"`
	Schema          *Schema `json:"schema,omitempty"`
	AllowEmptyValue bool    `json:"allowEmptyValue,omitempty"`
}

// Marshaling structure only, always edit along with corresponding
// struct (or compilation will fail).
type paramPropsOmitZero struct {
	Description     string  `json:"description,omitempty"`
	Name            string  `json:"name,omitempty"`
	In              string  `json:"in,omitempty"`
	Required        bool    `json:"required,omitzero"`
	Schema          *Schema `json:"schema,omitzero"`
	AllowEmptyValue bool    `json:"allowEmptyValue,omitzero"`
}

// Parameter a unique parameter is defined by a combination of a [name](#parameterName) and [location](#parameterIn).
//
// There are five possible parameter types.
// * Path - Used together with [Path Templating](#pathTemplating), where the parameter value is actually part
//
//	of the operation's URL. This does not include the host or base path of the API. For example, in `/items/{itemId}`,
//	the path parameter is `itemId`.
//
// * Query - Parameters that are appended to the URL. For example, in `/items?id=###`, the query parameter is `id`.
// * Header - Custom headers that are expected as part of the request.
// * Body - The payload that's appended to the HTTP request. Since there can only be one payload, there can only be
//
//	_one_ body parameter. The name of the body parameter has no effect on the parameter itself and is used for
//	documentation purposes only. Since Form parameters are also in the payload, body and form parameters cannot exist
//	together for the same operation.
//
// * Form - Used to describe the payload of an HTTP request when either `application/x-www-form-urlencoded` or
//
//	`multipart/form-data` are used as the content type of the request (in Swagger's definition,
//	the [`consumes`](#operationConsumes) property of an operation). This is the only parameter type that can be used
//	to send files, thus supporting the `file` type. Since form parameters are sent in the payload, they cannot be
//	declared together with a body parameter for the same operation. Form parameters have a different format based on
//	the content-type used (for further details, consult http://www.w3.org/TR/html401/interact/forms.html#h-17.13.4).
//	* `application/x-www-form-urlencoded` - Similar to the format of Query parameters but as a payload.
//	For example, `foo=1&bar=swagger` - both `foo` and `bar` are form parameters. This is normally used for simple
//	parameters that are being transferred.
//	* `multipart/form-data` - each parameter takes a section in the payload with an internal header.
//	For example, for the header `Content-Disposition: form-data; name="submit-name"` the name of the parameter is
//	`submit-name`. This type of form parameters is more commonly used for file transfers.
//
// For more information: http://goo.gl/8us55a#parameterObject
type Parameter struct {
	Refable
	CommonValidations
	SimpleSchema
	VendorExtensible
	ParamProps
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (p *Parameter) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, p)
	}

	if err := json.Unmarshal(data, &p.CommonValidations); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &p.Refable); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &p.SimpleSchema); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &p.VendorExtensible); err != nil {
		return err
	}
	return json.Unmarshal(data, &p.ParamProps)
}

func (p *Parameter) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		CommonValidations
		SimpleSchema
		Extensions
		ParamProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	if err := p.Refable.Ref.fromMap(x.Extensions); err != nil {
		return err
	}
	p.CommonValidations = x.CommonValidations
	p.SimpleSchema = x.SimpleSchema
	p.Extensions = internal.SanitizeExtensions(x.Extensions)
	p.ParamProps = x.ParamProps
	return nil
}

// MarshalJSON converts this items object to JSON
func (p Parameter) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(p)
	}
	b1, err := json.Marshal(p.CommonValidations)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(p.SimpleSchema)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(p.Refable)
	if err != nil {
		return nil, err
	}
	b4, err := json.Marshal(p.VendorExtensible)
	if err != nil {
		return nil, err
	}
	b5, err := json.Marshal(p.ParamProps)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b3, b1, b2, b4, b5), nil
}

func (p Parameter) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		CommonValidations commonValidationsOmitZero `json:",inline"`
		SimpleSchema      simpleSchemaOmitZero      `json:",inline"`
		ParamProps        paramPropsOmitZero        `json:",inline"`
		Ref               string                    `json:"$ref,omitempty"`
		Extensions
	}
	x.CommonValidations = commonValidationsOmitZero(p.CommonValidations)
	x.SimpleSchema = simpleSchemaOmitZero(p.SimpleSchema)
	x.Extensions = internal.SanitizeExtensions(p.Extensions)
	x.ParamProps = paramPropsOmitZero(p.ParamProps)
	x.Ref = p.Refable.Ref.String()
	return opts.MarshalNext(enc, x)
}
