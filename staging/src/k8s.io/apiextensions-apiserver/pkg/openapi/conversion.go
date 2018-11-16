/*
Copyright 2018 The Kubernetes Authors.

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

package openapi

import (
	"encoding/json"

	"github.com/go-openapi/spec"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

// ConvertJSONSchemaPropsToOpenAPIv2Schema converts our internal OpenAPI v3 schema
// (*apiextensions.JSONSchemaProps) to an OpenAPI v2 schema (*spec.Schema).
// NOTE: we use versioned type (v1beta1) here so that we can properly marshal the object
// using the JSON tags
func ConvertJSONSchemaPropsToOpenAPIv2Schema(in *v1beta1.JSONSchemaProps) (*spec.Schema, error) {
	if in == nil {
		return nil, nil
	}

	// Marshal JSONSchemaProps into JSON and unmarshal the data into spec.Schema
	data, err := json.Marshal(*in)
	if err != nil {
		return nil, err
	}
	out := new(spec.Schema)
	if err := out.UnmarshalJSON(data); err != nil {
		return nil, err
	}
	// Remove unsupported fields in OpenAPI v2
	out.OneOf = nil
	out.AnyOf = nil
	out.Not = nil
	return out, nil
}
