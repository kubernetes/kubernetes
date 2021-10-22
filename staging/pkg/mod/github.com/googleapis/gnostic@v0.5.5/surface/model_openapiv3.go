// Copyright 2017 Google LLC. All Rights Reserved.
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

package surface_v1

import (
	"github.com/googleapis/gnostic/compiler"
	openapiv3 "github.com/googleapis/gnostic/openapiv3"
	"log"
	"strings"
)

type OpenAPI3Builder struct {
	model    *Model
	document *openapiv3.Document
}

// NewModelFromOpenAPIv3 builds a model of an API service for use in code generation.
func NewModelFromOpenAPI3(document *openapiv3.Document, sourceName string) (*Model, error) {
	return newOpenAPI3Builder(document).buildModel(document, sourceName)
}

func newOpenAPI3Builder(document *openapiv3.Document) *OpenAPI3Builder {
	return &OpenAPI3Builder{model: &Model{}, document: document}
}

// Fills the surface model with information from a parsed OpenAPI description. The surface model provides that information
// in a way  that is more processable by plugins like gnostic-go-generator or gnostic-grpc.
// Since OpenAPI schemas can be indefinitely nested, it is a recursive approach to build all Types and Methods.
// The basic idea is that whenever we have "named OpenAPI object" (e.g.: NamedSchemaOrReference, NamedMediaType) we:
//	1. Create a Type with that name
//	2. Recursively create sub schemas (see buildFromSchema function)
// 	3. Return a FieldInfo object that describes how the created Type should be represented inside another Type as field.
func (b *OpenAPI3Builder) buildModel(document *openapiv3.Document, sourceName string) (*Model, error) {
	b.model.Types = make([]*Type, 0)
	b.model.Methods = make([]*Method, 0)
	// Set model properties from passed-in document.
	b.model.Name = document.Info.Title
	b.buildFromDocument(document)
	err := b.buildSymbolicReferences(document, sourceName)
	if err != nil {
		log.Printf("Error while building symbolic references. This might cause the plugin to fail: %v", err)
	}
	return b.model, nil
}

// Builds Types from the component section; builds Types and methods from paths;
func (b *OpenAPI3Builder) buildFromDocument(document *openapiv3.Document) {
	b.buildFromComponents(document.Components)
	b.buildFromPaths(document.Paths)
}

// Builds all Types from an "OpenAPI component" section
func (b *OpenAPI3Builder) buildFromComponents(components *openapiv3.Components) {
	if components == nil {
		return
	}

	for _, namedSchema := range components.GetSchemas().GetAdditionalProperties() {
		fInfo := b.buildFromSchemaOrReference(namedSchema.Name, namedSchema.Value)
		b.checkForExistence(namedSchema.Name, fInfo)
	}

	for _, namedParameter := range components.GetParameters().GetAdditionalProperties() {
		// Parameters in OpenAPI have a name field. See: https://swagger.io/specification/#parameterObject
		// The name gets passed up the callstack and is therefore contained inside fInfo. That is why we pass "" as fieldName
		// A type with that parameter was never created, so we still need to do that.
		t := makeType(namedParameter.Name)
		fInfo := b.buildFromParamOrRef(namedParameter.Value)
		makeFieldAndAppendToType(fInfo, t, "")
		if len(t.Fields) > 0 {
			b.model.addType(t)
		}
	}

	for _, namedResponses := range components.GetResponses().GetAdditionalProperties() {
		fInfo := b.buildFromResponseOrRef(namedResponses.Name, namedResponses.Value)
		b.checkForExistence(namedResponses.Name, fInfo)
	}

	for _, namedRequestBody := range components.GetRequestBodies().GetAdditionalProperties() {
		fInfo := b.buildFromRequestBodyOrRef(namedRequestBody.Name, namedRequestBody.Value)
		b.checkForExistence(namedRequestBody.Name, fInfo)
	}
}

// Builds Methods and Types (parameters, request bodies, responses) from all paths
func (b *OpenAPI3Builder) buildFromPaths(paths *openapiv3.Paths) {
	for _, path := range paths.Path {
		b.buildFromNamedPath(path.Name, path.Value)
	}
}

// Builds all symbolic references. A symbolic reference is an URL to another OpenAPI description. We call "document.ResolveReferences"
// inside that method. This has the same effect like: "gnostic --resolve-refs"
func (b *OpenAPI3Builder) buildSymbolicReferences(document *openapiv3.Document, sourceName string) (err error) {
	cache := compiler.GetInfoCache()
	if len(cache) == 0 && sourceName != "" {
		// Fills the compiler cache with all kind of references.
		_, err = document.ResolveReferences(sourceName)
		if err != nil {
			return err
		}
		cache = compiler.GetInfoCache()
	}

	for ref := range cache {
		if isSymbolicReference(ref) {
			b.model.SymbolicReferences = append(b.model.SymbolicReferences, ref)
		}
	}
	// Clear compiler cache for recursive calls
	compiler.ClearInfoCache()
	return nil
}

// Builds a Method and adds it to the surface model
func (b *OpenAPI3Builder) buildFromNamedPath(name string, pathItem *openapiv3.PathItem) {
	for _, method := range []string{"GET", "PUT", "POST", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"} {
		var op *openapiv3.Operation
		switch method {
		case "GET":
			op = pathItem.Get
		case "PUT":
			op = pathItem.Put
		case "POST":
			op = pathItem.Post
		case "DELETE":
			op = pathItem.Delete
		case "OPTIONS":
			op = pathItem.Options
		case "HEAD":
			op = pathItem.Head
		case "PATCH":
			op = pathItem.Patch
		case "TRACE":
			op = pathItem.Trace
		}
		if op != nil {
			m := &Method{
				Operation:   op.OperationId,
				Path:        name,
				Method:      method,
				Name:        sanitizeOperationName(op.OperationId),
				Description: op.Description,
			}
			if m.Name == "" {
				m.Name = generateOperationName(method, name)
			}
			m.ParametersTypeName, m.ResponsesTypeName = b.buildFromNamedOperation(m.Name, op)
			b.model.addMethod(m)
		}
	}
}

// Builds the "Parameters" and "Responses" types for an operation, adds them to the model, and returns the names of the types.
// If no such Type is added to the model an empty string is returned.
func (b *OpenAPI3Builder) buildFromNamedOperation(name string, operation *openapiv3.Operation) (parametersTypeName string, responseTypeName string) {
	// At first, we build the operations input parameters. This includes parameters (like PATH or QUERY parameters) and a request body
	operationParameters := makeType(name + "Parameters")
	operationParameters.Description = operationParameters.Name + " holds parameters to " + name
	for _, paramOrRef := range operation.Parameters {
		fieldInfo := b.buildFromParamOrRef(paramOrRef)
		// For parameters the name of the field is contained inside fieldInfo. That is why we pass "" as fieldName
		makeFieldAndAppendToType(fieldInfo, operationParameters, "")
	}

	if operation.RequestBody != nil {
		fInfo := b.buildFromRequestBodyOrRef(operation.OperationId+"RequestBody", operation.RequestBody)
		makeFieldAndAppendToType(fInfo, operationParameters, "request_body")
	}

	if len(operationParameters.Fields) > 0 {
		b.model.addType(operationParameters)
		parametersTypeName = operationParameters.Name
	}

	// Secondly, we build the response values for the method.
	if responses := operation.Responses; responses != nil {
		operationResponses := makeType(name + "Responses")
		operationResponses.Description = operationResponses.Name + " holds responses of " + name
		for _, namedResponse := range responses.ResponseOrReference {
			fieldInfo := b.buildFromResponseOrRef(operation.OperationId+convertStatusCodeToText(namedResponse.Name), namedResponse.Value)
			makeFieldAndAppendToType(fieldInfo, operationResponses, namedResponse.Name)
		}
		if responses.Default != nil {
			fieldInfo := b.buildFromResponseOrRef(operation.OperationId+"Default", responses.Default)
			makeFieldAndAppendToType(fieldInfo, operationResponses, "default")
		}
		if len(operationResponses.Fields) > 0 {
			b.model.addType(operationResponses)
			responseTypeName = operationResponses.Name
		}
	}
	return parametersTypeName, responseTypeName
}

// A helper method to differentiate between references and actual objects.
// The actual Field and Type are created in the functions which call this function
func (b *OpenAPI3Builder) buildFromParamOrRef(paramOrRef *openapiv3.ParameterOrReference) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if param := paramOrRef.GetParameter(); param != nil {
		fInfo = b.buildFromParam(param)
		return fInfo
	} else if ref := paramOrRef.GetReference(); ref != nil {
		t := findType(b.model.Types, validTypeForRef(ref.XRef))
		if t != nil && len(t.Fields) > 0 {
			fInfo.fieldKind, fInfo.fieldType, fInfo.fieldName, fInfo.fieldPosition = FieldKind_REFERENCE, validTypeForRef(ref.XRef), t.Name, t.Fields[0].Position
			return fInfo
		}
		// TODO: This might happen for symbolic references --> fInfo.Position defaults to 'BODY' which is wrong.
		log.Printf("Not able to find parameter information for: %v", ref)
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, validTypeForRef(ref.XRef)
		return fInfo // Lets return fInfo for now otherwise we may get null pointer exception
	}
	return nil
}

// Returns information on how to represent 'parameter' as field. This information gets propagated up the callstack.
func (b *OpenAPI3Builder) buildFromParam(parameter *openapiv3.Parameter) (fInfo *FieldInfo) {
	if schemaOrRef := parameter.Schema; schemaOrRef != nil {
		fInfo = b.buildFromSchemaOrReference(parameter.Name, schemaOrRef)
		fInfo.fieldName = parameter.Name
		switch parameter.In {
		case "body":
			fInfo.fieldPosition = Position_BODY
		case "header":
			fInfo.fieldPosition = Position_HEADER
		case "formdata":
			fInfo.fieldPosition = Position_FORMDATA
		case "query":
			fInfo.fieldPosition = Position_QUERY
		case "path":
			fInfo.fieldPosition = Position_PATH
		}
		return fInfo
	}
	return nil
}

// A helper method to differentiate between references and actual objects
func (b *OpenAPI3Builder) buildFromRequestBodyOrRef(name string, reqBodyOrRef *openapiv3.RequestBodyOrReference) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if requestBody := reqBodyOrRef.GetRequestBody(); requestBody != nil {
		fInfo = b.buildFromRequestBody(name, requestBody)
		return fInfo
	} else if ref := reqBodyOrRef.GetReference(); ref != nil {
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, validTypeForRef(ref.XRef)
		return fInfo
	}
	return nil
}

// Builds a Type for 'reqBody' and returns information on how to use this Type as field.
func (b *OpenAPI3Builder) buildFromRequestBody(name string, reqBody *openapiv3.RequestBody) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if reqBody.Content != nil {
		schemaType := makeType(name)
		for _, namedMediaType := range reqBody.Content.AdditionalProperties {
			fieldInfo := b.buildFromSchemaOrReference(name+namedMediaType.Name, namedMediaType.GetValue().GetSchema())
			makeFieldAndAppendToType(fieldInfo, schemaType, namedMediaType.Name)
		}
		b.model.addType(schemaType)
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, schemaType.Name
		return fInfo
	}
	return nil
}

// A helper method to differentiate between references and actual objects
func (b *OpenAPI3Builder) buildFromResponseOrRef(name string, responseOrRef *openapiv3.ResponseOrReference) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if response := responseOrRef.GetResponse(); response != nil {
		fInfo = b.buildFromResponse(name, response)
		return fInfo
	} else if ref := responseOrRef.GetReference(); ref != nil {
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, validTypeForRef(ref.XRef)
		return fInfo
	}
	return nil
}

// Builds a Type for 'response' and returns information on how to use this Type as field.
func (b *OpenAPI3Builder) buildFromResponse(name string, response *openapiv3.Response) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if response.Content != nil && response.Content.AdditionalProperties != nil {
		schemaType := makeType(name)
		for _, namedMediaType := range response.Content.AdditionalProperties {
			fieldInfo := b.buildFromSchemaOrReference(name+namedMediaType.Name, namedMediaType.GetValue().GetSchema())
			makeFieldAndAppendToType(fieldInfo, schemaType, namedMediaType.Name)
		}
		b.model.addType(schemaType)
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, schemaType.Name
		return fInfo
	}
	return nil
}

// A helper method to differentiate between references and actual objects
func (b *OpenAPI3Builder) buildFromSchemaOrReference(name string, schemaOrReference *openapiv3.SchemaOrReference) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if schema := schemaOrReference.GetSchema(); schema != nil {
		fInfo = b.buildFromSchema(name, schema)
		return fInfo
	} else if ref := schemaOrReference.GetReference(); ref != nil {
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, validTypeForRef(ref.XRef)
		return fInfo
	}
	return nil
}

// Given an OpenAPI schema there are two possibilities:
//  1. 	The schema is an object/array: We create a type for the object, recursively call according methods for child
//  	schemas, and then return information on how to use the created Type as field.
//	2. 	The schema has a scalar type: We return information on how to represent a scalar schema as Field. Fields are
//		created whenever Types are created (higher up in the callstack). This possibility can be considered as the "base condition"
//		for the recursive approach.
func (b *OpenAPI3Builder) buildFromSchema(name string, schema *openapiv3.Schema) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	// Data types according to: https://swagger.io/docs/specification/data-models/data-types/
	switch schema.Type {
	case "":
		fallthrough
	case "object":
		schemaType := makeType(name)

		for _, namedSchema := range schema.GetProperties().GetAdditionalProperties() {
			fieldInfo := b.buildFromSchemaOrReference(namedSchema.Name, namedSchema.Value)
			makeFieldAndAppendToType(fieldInfo, schemaType, namedSchema.Name)
		}

		if schemaOrRef := schema.AdditionalProperties.GetSchemaOrReference(); schemaOrRef != nil {
			// AdditionalProperties are represented as map
			fieldInfo := b.buildFromSchemaOrReference(name+"AdditionalProperties", schemaOrRef)
			if fieldInfo != nil {
				mapValueType := determineMapValueType(*fieldInfo)
				fieldInfo.fieldKind, fieldInfo.fieldType, fieldInfo.fieldFormat = FieldKind_MAP, "map[string]"+mapValueType, ""
				makeFieldAndAppendToType(fieldInfo, schemaType, "additional_properties")
			}
		}

		for _, schemaOrRef := range schema.AnyOf {
			b.buildFromOneOfAnyOfAndAllOf(schemaOrRef, schemaType)
		}

		for _, schemaOrRef := range schema.OneOf {
			b.buildFromOneOfAnyOfAndAllOf(schemaOrRef, schemaType)
		}

		for _, schemaOrRef := range schema.AllOf {
			b.buildFromOneOfAnyOfAndAllOf(schemaOrRef, schemaType)
		}

		if schema.Items != nil {
			for _, schemaOrRef := range schema.Items.SchemaOrReference {
				fieldInfo := b.buildFromSchemaOrReference(name+"Items", schemaOrRef)
				makeFieldAndAppendToType(fieldInfo, schemaType, "items")
			}
		}

		if len(schemaType.Fields) == 0 {
			schemaType.Kind = TypeKind_OBJECT
			schemaType.ContentType = "interface{}"
		}
		if t := findType(b.model.Types, schemaType.Name); t == nil {
			b.model.addType(schemaType)
		}
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, schemaType.Name
		return fInfo
	case "array":
		// I do believe there is a bug inside of OpenAPIv3.proto. I think the type of "schema.Items.SchemaOrReference"
		// shouldn't be an array of pointers but rather a single pointer.
		// According to: https://swagger.io/specification/#schemaObject
		// The 'items' "Value MUST be an object and not an array" and "Inline or referenced schema MUST be of a Schema Object"
		for _, schemaOrRef := range schema.Items.SchemaOrReference {
			arrayFieldInfo := b.buildFromSchemaOrReference(name, schemaOrRef)
			if arrayFieldInfo != nil {
				fInfo.fieldKind, fInfo.fieldType, fInfo.fieldFormat, fInfo.enumValues = FieldKind_ARRAY, arrayFieldInfo.fieldType, arrayFieldInfo.fieldFormat, arrayFieldInfo.enumValues
				return fInfo
			}
		}
	default:
		for _, enum := range schema.GetEnum() {
			fInfo.enumValues = append(fInfo.enumValues, strings.TrimSuffix(enum.Yaml, "\n"))
		}
		// We go a scalar value
		fInfo.fieldKind, fInfo.fieldType, fInfo.fieldFormat = FieldKind_SCALAR, schema.Type, schema.Format
		return fInfo
	}
	log.Printf("Unimplemented: could not find field info for schema: %v", schema)
	return nil
}

// buildFromOneOfAnyOfAndAllOf adds appropriate fields to the 'schemaType' given a new 'schemaOrRef'.
func (b *OpenAPI3Builder) buildFromOneOfAnyOfAndAllOf(schemaOrRef *openapiv3.SchemaOrReference, schemaType *Type) {
	// Related: https://github.com/googleapis/gnostic-grpc/issues/22
	if schema := schemaOrRef.GetSchema(); schema != nil {
		// Build a temporary type that has the required fields; add the fields to the current schema; remove the
		// temporary type
		fieldInfo := b.buildFromSchemaOrReference("ATemporaryTypeThatWillBeRemoved", schemaOrRef)
		t := findType(b.model.Types, "ATemporaryTypeThatWillBeRemoved")
		if t == nil {
			// schemaOrRef is some kind of primitive schema (e.g. of type string)
			makeFieldAndAppendToType(fieldInfo, schemaType, "value")
			return
		}
		schemaType.Fields = append(schemaType.Fields, t.Fields...)
		b.removeType(t)
	} else if ref := schemaOrRef.GetReference(); ref != nil {
		referencedSchemaName := validTypeForRef(ref.XRef)
		// Make sure that the referenced type exists, before we add the fields to the current schema
		for _, namedSchema := range b.document.GetComponents().GetSchemas().GetAdditionalProperties() {
			if referencedSchemaName == namedSchema.Name {
				fInfo := b.buildFromSchemaOrReference(namedSchema.Name, namedSchema.Value)
				b.checkForExistence(namedSchema.Name, fInfo)
				break
			}
		}
		t := findType(b.model.Types, referencedSchemaName)
		if t == nil {
			log.Printf("Unable to construct from OneOf, AnyOf, or AllOf schema. Referenced schema not found: %v", ref)
			return
		}
		schemaType.Fields = append(schemaType.Fields, t.Fields...)
	}
}

// removeType removes the Type 'toRemove' from the model.
func (b *OpenAPI3Builder) removeType(toRemove *Type) {
	res := make([]*Type, 0)
	for _, t := range b.model.Types {
		if t != toRemove {
			res = append(res, t)
		}
	}
	b.model.Types = res
}

// checkForExistence creates a type (if a type with 'name' does not already exist) and adds a field with the
// information from 'fInfo'.
func (b *OpenAPI3Builder) checkForExistence(name string, fInfo *FieldInfo) {
	// In certain cases no type will be created during the recursion. (e.g.: the schema is a primitive schema)
	if t := findType(b.model.Types, name); t == nil {
		t = makeType(name)
		makeFieldAndAppendToType(fInfo, t, "value")
		b.model.addType(t)
	}
}
