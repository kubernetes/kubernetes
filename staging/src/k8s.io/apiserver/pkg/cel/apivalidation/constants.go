package apivalidation

import (
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// Validation library for a schema
// against a typed or unstructured object
type OpenAPISchemaType string

const (
	SchemaTypeString  OpenAPISchemaType = "string"
	SchemaTypeNumber  OpenAPISchemaType = "number"
	SchemaTypeInteger OpenAPISchemaType = "integer"
	SchemaTypeArray   OpenAPISchemaType = "array"
	SchemaTypeObject  OpenAPISchemaType = "object"
	SchemaTypeBool    OpenAPISchemaType = "boolean"
	SchemaTypeNull    OpenAPISchemaType = "null"
)

var WildcardSchema common.Schema = &openapi.Schema{Schema: &spec.Schema{VendorExtensible: spec.VendorExtensible{
	Extensions: spec.Extensions{
		"x-kubernetes-preserve-unknown-fields": true,
	},
}}}
