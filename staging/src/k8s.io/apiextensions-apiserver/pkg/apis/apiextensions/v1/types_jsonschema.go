/*
Copyright 2017 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
)

// JSONSchemaProps is a JSON-Schema following Specification Draft 4 (http://json-schema.org/).
type JSONSchemaProps = apiextensions.JSONSchemaProps

// JSONSchemaPropsOrArray represents a value that can either be a JSONSchemaProps
// or an array of JSONSchemaProps. Mainly here for serialization purposes.
type JSONSchemaPropsOrArray = apiextensions.JSONSchemaPropsOrArray

// JSONSchemaPropsOrBool represents JSONSchemaProps or a boolean value.
// Defaults to true for the boolean property.
type JSONSchemaPropsOrBool = apiextensions.JSONSchemaPropsOrBool

// ValidationRules describes a list of validation rules written in the CEL expression language.
type ValidationRules = apiextensions.ValidationRules

// ValidationRule describes a validation rule written in the CEL expression language.
type ValidationRule = apiextensions.ValidationRule

// JSON represents any valid JSON value.
// These types are supported: bool, int64, float64, string, []interface{}, map[string]interface{} and nil.
//
// +protobuf=true
// +protobuf.options.marshal=false
// +protobuf.as=ProtoJSON
// +protobuf.options.(gogoproto.goproto_stringer)=false
type JSON = apiextensions.JSON
