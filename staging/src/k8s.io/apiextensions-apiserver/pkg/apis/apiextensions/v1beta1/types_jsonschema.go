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

package v1beta1

// JSONSchemaProps is a JSON-Schema following Specification Draft 4 (http://json-schema.org/).
type JSONSchemaProps struct {
	ID                   string                     `json:"id,omitempty" protobuf:"bytes,1,opt,name=id"`
	Schema               JSONSchemaURL              `json:"$schema,omitempty" protobuf:"bytes,2,opt,name=schema"`
	Ref                  *string                    `json:"$ref,omitempty" protobuf:"bytes,3,opt,name=ref"`
	Description          string                     `json:"description,omitempty" protobuf:"bytes,4,opt,name=description"`
	Type                 string                     `json:"type,omitempty" protobuf:"bytes,5,opt,name=type"`
	Format               string                     `json:"format,omitempty" protobuf:"bytes,6,opt,name=format"`
	Title                string                     `json:"title,omitempty" protobuf:"bytes,7,opt,name=title"`
	Default              *JSON                      `json:"default,omitempty" protobuf:"bytes,8,opt,name=default"`
	Maximum              *float64                   `json:"maximum,omitempty" protobuf:"bytes,9,opt,name=maximum"`
	ExclusiveMaximum     bool                       `json:"exclusiveMaximum,omitempty" protobuf:"bytes,10,opt,name=exclusiveMaximum"`
	Minimum              *float64                   `json:"minimum,omitempty" protobuf:"bytes,11,opt,name=minimum"`
	ExclusiveMinimum     bool                       `json:"exclusiveMinimum,omitempty" protobuf:"bytes,12,opt,name=exclusiveMinimum"`
	MaxLength            *int64                     `json:"maxLength,omitempty" protobuf:"bytes,13,opt,name=maxLength"`
	MinLength            *int64                     `json:"minLength,omitempty" protobuf:"bytes,14,opt,name=minLength"`
	Pattern              string                     `json:"pattern,omitempty" protobuf:"bytes,15,opt,name=pattern"`
	MaxItems             *int64                     `json:"maxItems,omitempty" protobuf:"bytes,16,opt,name=maxItems"`
	MinItems             *int64                     `json:"minItems,omitempty" protobuf:"bytes,17,opt,name=minItems"`
	UniqueItems          bool                       `json:"uniqueItems,omitempty" protobuf:"bytes,18,opt,name=uniqueItems"`
	MultipleOf           *float64                   `json:"multipleOf,omitempty" protobuf:"bytes,19,opt,name=multipleOf"`
	Enum                 []JSON                     `json:"enum,omitempty" protobuf:"bytes,20,rep,name=enum"`
	MaxProperties        *int64                     `json:"maxProperties,omitempty" protobuf:"bytes,21,opt,name=maxProperties"`
	MinProperties        *int64                     `json:"minProperties,omitempty" protobuf:"bytes,22,opt,name=minProperties"`
	Required             []string                   `json:"required,omitempty" protobuf:"bytes,23,rep,name=required"`
	Items                *JSONSchemaPropsOrArray    `json:"items,omitempty" protobuf:"bytes,24,opt,name=items"`
	AllOf                []JSONSchemaProps          `json:"allOf,omitempty" protobuf:"bytes,25,rep,name=allOf"`
	OneOf                []JSONSchemaProps          `json:"oneOf,omitempty" protobuf:"bytes,26,rep,name=oneOf"`
	AnyOf                []JSONSchemaProps          `json:"anyOf,omitempty" protobuf:"bytes,27,rep,name=anyOf"`
	Not                  *JSONSchemaProps           `json:"not,omitempty" protobuf:"bytes,28,opt,name=not"`
	Properties           map[string]JSONSchemaProps `json:"properties,omitempty" protobuf:"bytes,29,rep,name=properties"`
	AdditionalProperties *JSONSchemaPropsOrBool     `json:"additionalProperties,omitempty" protobuf:"bytes,30,opt,name=additionalProperties"`
	PatternProperties    map[string]JSONSchemaProps `json:"patternProperties,omitempty" protobuf:"bytes,31,rep,name=patternProperties"`
	Dependencies         JSONSchemaDependencies     `json:"dependencies,omitempty" protobuf:"bytes,32,opt,name=dependencies"`
	AdditionalItems      *JSONSchemaPropsOrBool     `json:"additionalItems,omitempty" protobuf:"bytes,33,opt,name=additionalItems"`
	Definitions          JSONSchemaDefinitions      `json:"definitions,omitempty" protobuf:"bytes,34,opt,name=definitions"`
	ExternalDocs         *ExternalDocumentation     `json:"externalDocs,omitempty" protobuf:"bytes,35,opt,name=externalDocs"`
	Example              *JSON                      `json:"example,omitempty" protobuf:"bytes,36,opt,name=example"`
}

// JSON represents any valid JSON value.
// These types are supported: bool, int64, float64, string, []interface{}, map[string]interface{} and nil.
type JSON struct {
	Raw []byte `protobuf:"bytes,1,opt,name=raw"`
}

// JSONSchemaURL represents a schema url.
type JSONSchemaURL string

// JSONSchemaPropsOrArray represents a value that can either be a JSONSchemaProps
// or an array of JSONSchemaProps. Mainly here for serialization purposes.
type JSONSchemaPropsOrArray struct {
	Schema      *JSONSchemaProps  `protobuf:"bytes,1,opt,name=schema"`
	JSONSchemas []JSONSchemaProps `protobuf:"bytes,2,rep,name=jSONSchemas"`
}

// JSONSchemaPropsOrBool represents JSONSchemaProps or a boolean value.
// Defaults to true for the boolean property.
type JSONSchemaPropsOrBool struct {
	Allows bool             `protobuf:"varint,1,opt,name=allows"`
	Schema *JSONSchemaProps `protobuf:"bytes,2,opt,name=schema"`
}

// JSONSchemaDependencies represent a dependencies property.
type JSONSchemaDependencies map[string]JSONSchemaPropsOrStringArray

// JSONSchemaPropsOrStringArray represents a JSONSchemaProps or a string array.
type JSONSchemaPropsOrStringArray struct {
	Schema   *JSONSchemaProps `protobuf:"bytes,1,opt,name=schema"`
	Property []string         `protobuf:"bytes,2,rep,name=property"`
}

// JSONSchemaDefinitions contains the models explicitly defined in this spec.
type JSONSchemaDefinitions map[string]JSONSchemaProps

// ExternalDocumentation allows referencing an external resource for extended documentation.
type ExternalDocumentation struct {
	Description string `json:"description,omitempty" protobuf:"bytes,1,opt,name=description"`
	URL         string `json:"url,omitempty" protobuf:"bytes,2,opt,name=url"`
}
