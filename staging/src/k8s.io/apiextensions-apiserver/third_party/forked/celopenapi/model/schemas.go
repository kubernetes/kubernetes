// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"strings"

	"gopkg.in/yaml.v3"
)

// NewOpenAPISchema returns an empty instance of an OpenAPISchema object.
func NewOpenAPISchema() *OpenAPISchema {
	return &OpenAPISchema{
		Enum:       []interface{}{},
		Metadata:   map[string]string{},
		Properties: map[string]*OpenAPISchema{},
		Required:   []string{},
	}
}

// OpenAPISchema declares a struct capable of representing a subset of Open API Schemas
// supported by Kubernetes which can also be specified within Protocol Buffers.
//
// There are a handful of notable differences:
// - The validating constructs `allOf`, `anyOf`, `oneOf`, `not`, and type-related restrictsion are
//   not supported as they can be better validated in the template 'validator' block.
// - The $ref field supports references to other schema definitions, but such aliases
//   should be removed before being serialized.
// - The `additionalProperties` and `properties` fields are not currently mutually exclusive as is
//   the case for Kubernetes.
//
// See: https://kubernetes.io/docs/tasks/access-kubernetes-api/custom-resources/custom-resource-definitions/#validation
type OpenAPISchema struct {
	Title                string                    `yaml:"title,omitempty"`
	Description          string                    `yaml:"description,omitempty"`
	Type                 string                    `yaml:"type,omitempty"`
	TypeParam            string                    `yaml:"type_param,omitempty"`
	TypeRef              string                    `yaml:"$ref,omitempty"`
	DefaultValue         interface{}               `yaml:"default,omitempty"`
	Enum                 []interface{}             `yaml:"enum,omitempty"`
	Format               string                    `yaml:"format,omitempty"`
	Items                *OpenAPISchema            `yaml:"items,omitempty"`
	Metadata             map[string]string         `yaml:"metadata,omitempty"`
	Required             []string                  `yaml:"required,omitempty"`
	Properties           map[string]*OpenAPISchema `yaml:"properties,omitempty"`
	AdditionalProperties *OpenAPISchema            `yaml:"additionalProperties,omitempty"`
}

// DeclTypes constructs a top-down set of DeclType instances whose name is derived from the root
// type name provided on the call, if not set to a custom type.
func (s *OpenAPISchema) DeclTypes(maybeRootType string) (*DeclType, map[string]*DeclType) {
	root := s.DeclType().MaybeAssignTypeName(maybeRootType)
	types := FieldTypeMap(maybeRootType, root)
	return root, types
}

// DeclType returns the CEL Policy Templates type name associated with the schema element.
func (s *OpenAPISchema) DeclType() *DeclType {
	if s.TypeParam != "" {
		return NewTypeParam(s.TypeParam)
	}
	declType, found := openAPISchemaTypes[s.Type]
	if !found {
		return NewObjectTypeRef("*error*")
	}
	switch declType.TypeName() {
	case ListType.TypeName():
		return NewListType(s.Items.DeclType())
	case MapType.TypeName():
		if s.AdditionalProperties != nil {
			return NewMapType(StringType, s.AdditionalProperties.DeclType())
		}
		fields := make(map[string]*DeclField, len(s.Properties))
		required := make(map[string]struct{}, len(s.Required))
		for _, name := range s.Required {
			required[name] = struct{}{}
		}
		for name, prop := range s.Properties {
			_, isReq := required[name]
			fields[name] = &DeclField{
				Name:         name,
				Required:     isReq,
				Type:         prop.DeclType(),
				defaultValue: prop.DefaultValue,
				enumValues:   prop.Enum,
			}
		}
		customType, hasCustomType := s.Metadata["custom_type"]
		if !hasCustomType {
			return NewObjectType("object", fields)
		}
		return NewObjectType(customType, fields)
	case StringType.TypeName():
		switch s.Format {
		case "byte", "binary":
			return BytesType
		case "google-duration":
			return DurationType
		case "date", "date-time", "google-datetime":
			return TimestampType
		case "int64":
			return IntType
		case "uint64":
			return UintType
		}
	}

	return declType
}

// FindProperty returns the Open API Schema type for the given property name.
//
// A property may either be explicitly defined in a `properties` map or implicitly defined in an
// `additionalProperties` block.
func (s *OpenAPISchema) FindProperty(name string) (*OpenAPISchema, bool) {
	if s.DeclType() == AnyType {
		return s, true
	}
	if s.Properties != nil {
		prop, found := s.Properties[name]
		if found {
			return prop, true
		}
	}
	if s.AdditionalProperties != nil {
		return s.AdditionalProperties, true
	}
	return nil, false
}

var (
	// SchemaDef defines an Open API Schema definition in terms of an Open API Schema.
	schemaDef *OpenAPISchema

	// AnySchema indicates that the value may be of any type.
	AnySchema *OpenAPISchema

	// EnvSchema defines the schema for CEL environments referenced within Policy Templates.
	envSchema *OpenAPISchema

	// InstanceSchema defines a basic schema for defining Policy Instances where the instance rule
	// references a TemplateSchema derived from the Instance's template kind.
	instanceSchema *OpenAPISchema

	// TemplateSchema defines a schema for defining Policy Templates.
	templateSchema *OpenAPISchema

	openAPISchemaTypes map[string]*DeclType = map[string]*DeclType{
		"boolean":         BoolType,
		"number":          DoubleType,
		"integer":         IntType,
		"null":            NullType,
		"string":          StringType,
		"google-duration": DurationType,
		"google-datetime": TimestampType,
		"date":            TimestampType,
		"date-time":       TimestampType,
		"array":           ListType,
		"object":          MapType,
		"":                AnyType,
	}
)

const (
	schemaDefYaml = `
type: object
properties:
  $ref:
    type: string
  type:
    type: string
  type_param:  # prohibited unless used within an environment.
    type: string
  format:
    type: string
  description:
    type: string
  required:
    type: array
    items:
      type: string
  enum:
    type: array
    items:
      type: string
  enumDescriptions:
    type: array
    items:
      type: string
  default: {}
  items:
    $ref: "#openAPISchema"
  properties:
    type: object
    additionalProperties:
      $ref: "#openAPISchema"
  additionalProperties:
    $ref: "#openAPISchema"
  metadata:
    type: object
    additionalProperties:
      type: string
`

	templateSchemaYaml = `
type: object
required:
  - apiVersion
  - kind
  - metadata
  - evaluator
properties:
  apiVersion:
    type: string
  kind:
    type: string
  metadata:
    type: object
    required:
      - name
    properties:
      uid:
        type: string
      name:
        type: string
      namespace:
        type: string
        default: "default"
      etag:
        type: string
      labels:
        type: object
        additionalProperties:
          type: string
      pluralName:
        type: string
  description:
    type: string
  schema:
    $ref: "#openAPISchema"
  validator:
    type: object
    required:
      - productions
    properties:
      description:
        type: string
      environment:
        type: string
      terms:
        type: object
        additionalProperties: {}
      productions:
        type: array
        items:
          type: object
          required:
            - message
          properties:
            match:
              type: string
              default: true
            field:
              type: string
            message:
              type: string
            details: {}
  evaluator:
    type: object
    required:
      - productions
    properties:
      description:
        type: string
      environment:
        type: string
      ranges:
        type: array
        items:
          type: object
          required:
            - in
          properties:
            in:
              type: string
            key:
              type: string
            index:
              type: string
            value:
              type: string
      terms:
        type: object
        additionalProperties:
          type: string
      productions:
        type: array
        items:
          type: object
          properties:
            match:
              type: string
              default: "true"
            decision:
              type: string
            decisionRef:
              type: string
            output: {}
            decisions:
              type: array
              items:
                type: object
                required:
                  - output
                properties:
                  decision:
                    type: string
                  decisionRef:
                    type: string
                  output: {}
`

	instanceSchemaYaml = `
type: object
required:
  - apiVersion
  - kind
  - metadata
properties:
  apiVersion:
    type: string
  kind:
    type: string
  metadata:
    type: object
    additionalProperties:
      type: string
  description:
    type: string
  selector:
    type: object
    properties:
      matchLabels:
        type: object
        additionalProperties:
          type: string
      matchExpressions:
        type: array
        items:
          type: object
          required:
            - key
            - operator
          properties:
            key:
              type: string
            operator:
              type: string
              enum: ["DoesNotExist", "Exists", "In", "NotIn"]
            values:
              type: array
              items: {}
              default: []
  rule:
    $ref: "#templateRuleSchema"
  rules:
    type: array
    items:
      $ref: "#templateRuleSchema"
`

	// TODO: support subsetting of built-in functions and macros
	// TODO: support naming anonymous types within rule schema and making them accessible to
	// declarations.
	// TODO: consider supporting custom macros
	envSchemaYaml = `
type: object
required:
  - name
properties:
  name:
    type: string
  container:
    type: string
  variables:
    type: object
    additionalProperties:
      $ref: "#openAPISchema"
  functions:
    type: object
    properties:
      extensions:
        type: object
        additionalProperties:
          type: object   # function name
          additionalProperties:
            type: object # overload name
            required:
              - return
            properties:
              free_function:
                type: boolean
              args:
                type: array
                items:
                  $ref: "#openAPISchema"
              return:
                $ref: "#openAPISchema"
`
)

func init() {
	AnySchema = NewOpenAPISchema()

	instanceSchema = NewOpenAPISchema()
	in := strings.ReplaceAll(instanceSchemaYaml, "\t", "  ")
	err := yaml.Unmarshal([]byte(in), instanceSchema)
	if err != nil {
		panic(err)
	}
	envSchema = NewOpenAPISchema()
	in = strings.ReplaceAll(envSchemaYaml, "\t", "  ")
	err = yaml.Unmarshal([]byte(in), envSchema)
	if err != nil {
		panic(err)
	}
	schemaDef = NewOpenAPISchema()
	in = strings.ReplaceAll(schemaDefYaml, "\t", "  ")
	err = yaml.Unmarshal([]byte(in), schemaDef)
	if err != nil {
		panic(err)
	}
	templateSchema = NewOpenAPISchema()
	in = strings.ReplaceAll(templateSchemaYaml, "\t", "  ")
	err = yaml.Unmarshal([]byte(in), templateSchema)
	if err != nil {
		panic(err)
	}
}
