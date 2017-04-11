package genswagger

import (
	"github.com/grpc-ecosystem/grpc-gateway/protoc-gen-grpc-gateway/descriptor"
)

type param struct {
	*descriptor.File
	reg *descriptor.Registry
}

type binding struct {
	*descriptor.Binding
}

// http://swagger.io/specification/#infoObject
type swaggerInfoObject struct {
	Title          string `json:"title"`
	Description    string `json:"description,omitempty"`
	TermsOfService string `json:"termsOfService,omitempty"`
	Version        string `json:"version"`

	Contact      *swaggerContactObject               `json:"contact,omitempty"`
	License      *swaggerLicenseObject               `json:"license,omitempty"`
	ExternalDocs *swaggerExternalDocumentationObject `json:"externalDocs,omitempty"`
}

// http://swagger.io/specification/#contactObject
type swaggerContactObject struct {
	Name  string `json:"name,omitempty"`
	URL   string `json:"url,omitempty"`
	Email string `json:"email,omitempty"`
}

// http://swagger.io/specification/#licenseObject
type swaggerLicenseObject struct {
	Name string `json:"name,omitempty"`
	URL  string `json:"url,omitempty"`
}

// http://swagger.io/specification/#externalDocumentationObject
type swaggerExternalDocumentationObject struct {
	Description string `json:"description,omitempty"`
	URL         string `json:"url,omitempty"`
}

// http://swagger.io/specification/#swaggerObject
type swaggerObject struct {
	Swagger     string                   `json:"swagger"`
	Info        swaggerInfoObject        `json:"info"`
	Host        string                   `json:"host,omitempty"`
	BasePath    string                   `json:"basePath,omitempty"`
	Schemes     []string                 `json:"schemes"`
	Consumes    []string                 `json:"consumes"`
	Produces    []string                 `json:"produces"`
	Paths       swaggerPathsObject       `json:"paths"`
	Definitions swaggerDefinitionsObject `json:"definitions"`
}

// http://swagger.io/specification/#pathsObject
type swaggerPathsObject map[string]swaggerPathItemObject

// http://swagger.io/specification/#pathItemObject
type swaggerPathItemObject struct {
	Get    *swaggerOperationObject `json:"get,omitempty"`
	Delete *swaggerOperationObject `json:"delete,omitempty"`
	Post   *swaggerOperationObject `json:"post,omitempty"`
	Put    *swaggerOperationObject `json:"put,omitempty"`
}

// http://swagger.io/specification/#operationObject
type swaggerOperationObject struct {
	Summary     string                  `json:"summary,omitempty"`
	Description string                  `json:"description,omitempty"`
	OperationID string                  `json:"operationId"`
	Responses   swaggerResponsesObject  `json:"responses"`
	Parameters  swaggerParametersObject `json:"parameters,omitempty"`
	Tags        []string                `json:"tags,omitempty"`

	ExternalDocs *swaggerExternalDocumentationObject `json:"externalDocs,omitempty"`
}

type swaggerParametersObject []swaggerParameterObject

// http://swagger.io/specification/#parameterObject
type swaggerParameterObject struct {
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	In          string              `json:"in,omitempty"`
	Required    bool                `json:"required"`
	Type        string              `json:"type,omitempty"`
	Format      string              `json:"format,omitempty"`
	Items       *swaggerItemsObject `json:"items,omitempty"`

	// Or you can explicitly refer to another type. If this is defined all
	// other fields should be empty
	Schema *swaggerSchemaObject `json:"schema,omitempty"`
}

// core part of schema, which is common to itemsObject and schemaObject.
// http://swagger.io/specification/#itemsObject
type schemaCore struct {
	Type   string `json:"type,omitempty"`
	Format string `json:"format,omitempty"`
	Ref    string `json:"$ref,omitempty"`
}

type swaggerItemsObject schemaCore

// http://swagger.io/specification/#responsesObject
type swaggerResponsesObject map[string]swaggerResponseObject

// http://swagger.io/specification/#responseObject
type swaggerResponseObject struct {
	Description string              `json:"description"`
	Schema      swaggerSchemaObject `json:"schema"`
}

// http://swagger.io/specification/#schemaObject
type swaggerSchemaObject struct {
	schemaCore
	// Properties can be recursively defined
	Properties           map[string]swaggerSchemaObject `json:"properties,omitempty"`
	AdditionalProperties *swaggerSchemaObject           `json:"additionalProperties,omitempty"`
	Items                *swaggerItemsObject            `json:"items,omitempty"`

	// If the item is an enumeration include a list of all the *NAMES* of the
	// enum values.  I'm not sure how well this will work but assuming all enums
	// start from 0 index it will be great. I don't think that is a good assumption.
	Enum    []string `json:"enum,omitempty"`
	Default string   `json:"default,omitempty"`

	Description string `json:"description,omitempty"`
	Title       string `json:"title,omitempty"`
}

// http://swagger.io/specification/#referenceObject
type swaggerReferenceObject struct {
	Ref string `json:"$ref"`
}

// http://swagger.io/specification/#definitionsObject
type swaggerDefinitionsObject map[string]swaggerSchemaObject

// Internal type mapping from FQMN to descriptor.Message. Used as a set by the
// findServiceMessages function.
type messageMap map[string]*descriptor.Message

// Internal type mapping from FQEN to descriptor.Enum. Used as a set by the
// findServiceMessages function.
type enumMap map[string]*descriptor.Enum
