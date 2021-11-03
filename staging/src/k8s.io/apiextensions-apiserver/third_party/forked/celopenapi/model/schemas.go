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
	"k8s.io/kube-openapi/pkg/validation/spec"
	"strings"

	"gopkg.in/yaml.v3"
)

// SchemaDeclTypes constructs a top-down set of DeclType instances whose name is derived from the root
// type name provided on the call, if not set to a custom type.
func SchemaDeclTypes(s *spec.Schema, maybeRootType string) (*DeclType, map[string]*DeclType) {
	root := SchemaDeclType(s).MaybeAssignTypeName(maybeRootType)
	types := FieldTypeMap(maybeRootType, root)
	return root, types
}

// SchemaDeclType returns the CEL Policy Templates type name associated with the schema element.
func SchemaDeclType(s *spec.Schema) *DeclType {
	if len(s.Type) < 1 {
		return NewObjectTypeRef("*error*")
	}
	for _, schemaType := range s.Type {
		declType, found := openAPISchemaTypes[schemaType]
		if !found {
			continue
		}
		switch declType.TypeName() {
		case ListType.TypeName():
			if s.Items.Len() == 1 {
				return NewListType(SchemaDeclType(s.Items.Schema))
			}
		case MapType.TypeName():
			if s.AdditionalProperties != nil {
				return NewMapType(StringType, SchemaDeclType(s.AdditionalProperties.Schema))
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
					Type:         SchemaDeclType(&prop),
					defaultValue: prop.Default,
					enumValues:   prop.Enum,
				}
			}
			return NewObjectType("object", fields)
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
	return NewObjectTypeRef("*error*")
}

var (
	// SchemaDef defines an Open API Schema definition in terms of an Open API Schema.
	schemaDef *spec.Schema

	// AnySchema indicates that the value may be of any type.
	AnySchema *spec.Schema

	// EnvSchema defines the schema for CEL environments referenced within Policy Templates.
	envSchema *spec.Schema

	// InstanceSchema defines a basic schema for defining Policy Instances where the instance rule
	// references a TemplateSchema derived from the Instance's template kind.
	instanceSchema *spec.Schema

	// TemplateSchema defines a schema for defining Policy Templates.
	templateSchema *spec.Schema

	openAPISchemaTypes = map[string]*DeclType{
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
	AnySchema = &spec.Schema{}

	instanceSchema = &spec.Schema{}
	in := strings.ReplaceAll(instanceSchemaYaml, "\t", "  ")
	err := yaml.Unmarshal([]byte(in), instanceSchema)
	if err != nil {
		panic(err)
	}
	envSchema = &spec.Schema{}
	in = strings.ReplaceAll(envSchemaYaml, "\t", "  ")
	err = yaml.Unmarshal([]byte(in), envSchema)
	if err != nil {
		panic(err)
	}
	schemaDef = &spec.Schema{}
	in = strings.ReplaceAll(schemaDefYaml, "\t", "  ")
	err = yaml.Unmarshal([]byte(in), schemaDef)
	if err != nil {
		panic(err)
	}
	templateSchema = &spec.Schema{}
	in = strings.ReplaceAll(templateSchemaYaml, "\t", "  ")
	err = yaml.Unmarshal([]byte(in), templateSchema)
	if err != nil {
		panic(err)
	}
}
