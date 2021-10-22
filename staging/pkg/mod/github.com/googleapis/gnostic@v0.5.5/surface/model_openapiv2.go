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
	openapiv2 "github.com/googleapis/gnostic/openapiv2"
	"github.com/googleapis/gnostic/compiler"
	"log"
	"strconv"
)

type OpenAPI2Builder struct {
	model *Model
}

// NewModelFromOpenAPI2 builds a model of an API service for use in code generation.
func NewModelFromOpenAPI2(document *openapiv2.Document, sourceName string) (*Model, error) {
	return newOpenAPI2Builder().buildModel(document, sourceName)
}

func newOpenAPI2Builder() *OpenAPI2Builder {
	return &OpenAPI2Builder{model: &Model{}}
}

// Fills the surface model with information from a parsed OpenAPI description. The surface model provides that information
// in a way  that is more processable by plugins like gnostic-go-generator or gnostic-grpc.
// Since OpenAPI schemas can be indefinitely nested, it is a recursive approach to build all Types and Methods.
// The basic idea is that whenever we have "named OpenAPI object" (e.g.: NamedSchemaOrReference, NamedMediaType) we:
//	1. Create a Type with that name
//	2. Recursively execute according methods on child schemas (see buildFromSchema function)
// 	3. Return a FieldInfo object that describes how the created Type should be represented inside another Type as field.
func (b *OpenAPI2Builder) buildModel(document *openapiv2.Document, sourceName string) (*Model, error) {
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

// Builds Types from definitions; builds Types and Methods from paths
func (b *OpenAPI2Builder) buildFromDocument(document *openapiv2.Document) {
	b.buildFromDefinitions(document.Definitions)
	b.buildFromParameterDefinitions(document.Parameters)
	b.buildFromResponseDefinitions(document.Responses)
	b.buildFromPaths(document.Paths)
}

// Build surface Types from OpenAPI definitions
func (b *OpenAPI2Builder) buildFromDefinitions(definitions *openapiv2.Definitions) {
	if definitions == nil {
		return
	}

	if schemas := definitions.AdditionalProperties; schemas != nil {
		for _, namedSchema := range schemas {
			fInfo := b.buildFromSchemaOrReference(namedSchema.Name, namedSchema.Value)
			// In certain cases no type will be created during the recursion: e.g.: the schema is of type scalar, array
			// or an reference. So we check whether the surface model Type already exists, and if not then we create it.
			if t := findType(b.model.Types, namedSchema.Name); t == nil {
				t = makeType(namedSchema.Name)
				makeFieldAndAppendToType(fInfo, t, "value")
				b.model.addType(t)
			}
		}
	}
}

// Build surface model Types from OpenAPI parameter definitions
func (b *OpenAPI2Builder) buildFromParameterDefinitions(parameters *openapiv2.ParameterDefinitions) {
	if parameters == nil {
		return
	}

	for _, namedParameter := range parameters.AdditionalProperties {
		// Parameters in OpenAPI have a name field. The name gets passed up the callstack and is therefore contained
		// inside fInfo. That is why we pass "" as fieldName. A type with that parameter was never created, so we still
		// need to do that.
		t := makeType(namedParameter.Name)
		fInfo := b.buildFromParam(namedParameter.Value)
		makeFieldAndAppendToType(fInfo, t, "")
		if len(t.Fields) > 0 {
			b.model.addType(t)
		}
	}
}

// Build surface model Types from OpenAPI response definitions
func (b *OpenAPI2Builder) buildFromResponseDefinitions(responses *openapiv2.ResponseDefinitions) {
	if responses == nil {
		return
	}
	for _, namedResponse := range responses.AdditionalProperties {
		fInfo := b.buildFromResponse(namedResponse.Name, namedResponse.Value)
		// In certain cases no type will be created during the recursion: e.g.: the schema is of type scalar, array
		// or an reference. So we check whether the surface model Type already exists, and if not then we create it.
		if t := findType(b.model.Types, namedResponse.Name); t == nil {
			t = makeType(namedResponse.Name)
			makeFieldAndAppendToType(fInfo, t, "value")
			b.model.addType(t)
		}
	}
}

// Builds all symbolic references. A symbolic reference is an URL to another OpenAPI description. We call "document.ResolveReferences"
// inside that method. This has the same effect like: "gnostic --resolve-refs"
func (b *OpenAPI2Builder) buildSymbolicReferences(document *openapiv2.Document, sourceName string) (err error) {
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

// Build Method and Types (parameter, request bodies, responses) from all paths
func (b *OpenAPI2Builder) buildFromPaths(paths *openapiv2.Paths) {
	for _, path := range paths.Path {
		b.buildFromNamedPath(path.Name, path.Value)
	}
}

// Builds a Method and adds it to the surface model
func (b *OpenAPI2Builder) buildFromNamedPath(name string, pathItem *openapiv2.PathItem) {
	for _, method := range []string{"GET", "PUT", "POST", "DELETE", "OPTIONS", "HEAD", "PATCH"} {
		var op *openapiv2.Operation
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
func (b *OpenAPI2Builder) buildFromNamedOperation(name string, operation *openapiv2.Operation) (parametersTypeName string, responseTypeName string) {
	// At first, we build the operations input parameters. This includes parameters (like PATH or QUERY parameters).
	operationParameters := makeType(name + "Parameters")
	operationParameters.Description = operationParameters.Name + " holds parameters to " + name
	for _, paramOrRef := range operation.Parameters {
		fieldInfo := b.buildFromParamOrRef(paramOrRef)
		// For parameters the name of the field is contained inside fieldInfo. That is why we pass "" as fieldName
		makeFieldAndAppendToType(fieldInfo, operationParameters, "")
	}
	if len(operationParameters.Fields) > 0 {
		b.model.addType(operationParameters)
		parametersTypeName = operationParameters.Name
	}

	// Secondly, we build the response values for the method.
	if responses := operation.Responses; responses != nil {
		operationResponses := makeType(name + "Responses")
		operationResponses.Description = operationResponses.Name + " holds responses of " + name
		for _, namedResponse := range responses.ResponseCode {
			fieldInfo := b.buildFromResponseOrRef(operation.OperationId+convertStatusCodeToText(namedResponse.Name), namedResponse.Value)
			makeFieldAndAppendToType(fieldInfo, operationResponses, namedResponse.Name)
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
func (b *OpenAPI2Builder) buildFromParamOrRef(paramOrRef *openapiv2.ParametersItem) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if param := paramOrRef.GetParameter(); param != nil {
		fInfo = b.buildFromParam(param)
		return fInfo
	} else if ref := paramOrRef.GetJsonReference(); ref != nil {
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
// We have to differentiate between 'body' and 'non-body' parameters
func (b *OpenAPI2Builder) buildFromParam(parameter *openapiv2.Parameter) (fInfo *FieldInfo) {
	if bodyParam := parameter.GetBodyParameter(); bodyParam != nil {
		fInfo = b.buildFromSchemaOrReference(bodyParam.Name, bodyParam.Schema)
		if fInfo != nil {
			fInfo.fieldName, fInfo.fieldPosition = bodyParam.Name, Position_BODY
			return fInfo
		}
	} else if nonBodyParam := parameter.GetNonBodyParameter(); nonBodyParam != nil {
		fInfo = b.buildFromNonBodyParameter(nonBodyParam)
		return fInfo
	}
	log.Printf("Couldn't build from parameter: %v", parameter)
	return nil
}

// Differentiates between different kind of non-body parameters
func (b *OpenAPI2Builder) buildFromNonBodyParameter(nonBodyParameter *openapiv2.NonBodyParameter) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	headerParameter := nonBodyParameter.GetHeaderParameterSubSchema()
	if headerParameter != nil {
		fInfo.fieldName, fInfo.fieldPosition, fInfo.fieldFormat = headerParameter.Name, Position_HEADER, headerParameter.Format
		b.adaptFieldKindAndFieldType(fInfo, headerParameter.Type, headerParameter.Items)
	}
	formDataParameter := nonBodyParameter.GetFormDataParameterSubSchema()
	if formDataParameter != nil {
		fInfo.fieldName, fInfo.fieldPosition, fInfo.fieldFormat = formDataParameter.Name, Position_FORMDATA, formDataParameter.Format
		b.adaptFieldKindAndFieldType(fInfo, formDataParameter.Type, formDataParameter.Items)
	}
	queryParameter := nonBodyParameter.GetQueryParameterSubSchema()
	if queryParameter != nil {
		fInfo.fieldName, fInfo.fieldPosition, fInfo.fieldFormat = queryParameter.Name, Position_QUERY, queryParameter.Format
		b.adaptFieldKindAndFieldType(fInfo, queryParameter.Type, queryParameter.Items)
	}
	pathParameter := nonBodyParameter.GetPathParameterSubSchema()
	if pathParameter != nil {
		fInfo.fieldName, fInfo.fieldPosition, fInfo.fieldFormat = pathParameter.Name, Position_PATH, pathParameter.Format
		b.adaptFieldKindAndFieldType(fInfo, pathParameter.Type, pathParameter.Items)
	}
	return fInfo
}

// Changes the fieldKind and fieldType inside of 'fInfo' based on different conditions. In case of an array we have to
// consider that it consists of indefinitely nested items.
func (b *OpenAPI2Builder) adaptFieldKindAndFieldType(fInfo *FieldInfo, parameterType string, parameterItems *openapiv2.PrimitivesItems) {
	fInfo.fieldKind, fInfo.fieldType = FieldKind_SCALAR, parameterType

	if parameterType == "array" && parameterItems != nil {
		fInfo.fieldKind, fInfo.fieldType = FieldKind_ARRAY, "string" // Default to string in case we don't find the type
		if parameterItems.Type != "" {
			// We only need the fieldType here because we know for sure that it is an array.
			fInfo.fieldType = b.buildFromPrimitiveItems(fInfo.fieldName, parameterItems, 0).fieldType
		}
	}

	if parameterType == "file" {
		fInfo.fieldKind, fInfo.fieldType = FieldKind_SCALAR, "string"
	}
}

// A recursive method that build Types for nested PrimitiveItems. The 'ctr' is used for naming the different Types.
// The base condition is if we have scalar value (not an array).
func (b *OpenAPI2Builder) buildFromPrimitiveItems(name string, items *openapiv2.PrimitivesItems, ctr int) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	switch items.Type {
	case "array":
		t := makeType(name)
		fieldInfo := b.buildFromPrimitiveItems(name+strconv.Itoa(ctr), items.Items, ctr+1)
		makeFieldAndAppendToType(fieldInfo, t, "items")

		if len(t.Fields) > 0 {
			b.model.addType(t)
			fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, t.Name
			return fInfo
		}
	default:
		// We got a scalar value
		fInfo.fieldKind, fInfo.fieldType, fInfo.fieldFormat = FieldKind_SCALAR, items.Type, items.Format
		return fInfo
	}
	return nil
}

// A helper method to differentiate between references and actual objects
func (b *OpenAPI2Builder) buildFromResponseOrRef(name string, responseOrRef *openapiv2.ResponseValue) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if response := responseOrRef.GetResponse(); response != nil {
		fInfo = b.buildFromResponse(name, response)
		return fInfo
	} else if ref := responseOrRef.GetJsonReference(); ref != nil {
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, validTypeForRef(ref.XRef)
		return fInfo
	}
	return nil
}

// A helper method to propagate the information up the call stack
func (b *OpenAPI2Builder) buildFromResponse(name string, response *openapiv2.Response) (fInfo *FieldInfo) {
	if response.Schema != nil && response.Schema.GetSchema() != nil {
		fInfo = b.buildFromSchemaOrReference(name, response.Schema.GetSchema())
		return fInfo
	}
	return nil
}

// A helper method to differentiate between references and actual objects
func (b *OpenAPI2Builder) buildFromSchemaOrReference(name string, schema *openapiv2.Schema) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}
	if schema.XRef != "" {
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, validTypeForRef(schema.XRef)
		return fInfo
	} else {
		fInfo = b.buildFromSchema(name, schema)
		return fInfo
	}
}

// Given an OpenAPI schema there are two possibilities:
//  1. 	The schema is an object/array: We create a type for the object, recursively call according methods for child
//  	schemas, and then return information on how to use the created Type as field.
//	2. 	The schema has a scalar type: We return information on how to represent a scalar schema as Field. Fields are
//		created whenever Types are created (higher up in the callstack). This possibility can be considered as the "base condition"
//		for the recursive approach.
func (b *OpenAPI2Builder) buildFromSchema(name string, schema *openapiv2.Schema) (fInfo *FieldInfo) {
	fInfo = &FieldInfo{}

	t := ""
	if schema.Type != nil && len(schema.Type.Value) == 1 && schema.Type.Value[0] != "null" {
		t = schema.Type.Value[0]
	}
	switch t {
	case "":
		fallthrough
	case "object":
		schemaType := makeType(name)
		if schema.Properties != nil && schema.Properties.AdditionalProperties != nil {
			for _, namedSchema := range schema.Properties.AdditionalProperties {
				fieldInfo := b.buildFromSchemaOrReference(namedSchema.Name, namedSchema.Value)
				makeFieldAndAppendToType(fieldInfo, schemaType, namedSchema.Name)
			}
		}
		if schema := schema.AdditionalProperties.GetSchema(); schema != nil {
			// AdditionalProperties are represented as map
			fieldInfo := b.buildFromSchemaOrReference(name+"AdditionalProperties", schema)
			if fieldInfo != nil {
				mapValueType := determineMapValueType(*fieldInfo)
				fieldInfo.fieldKind, fieldInfo.fieldType, fieldInfo.fieldFormat = FieldKind_MAP, "map[string]"+mapValueType, ""
				makeFieldAndAppendToType(fieldInfo, schemaType, "additional_properties")
			}
		}

		for idx, schemaOrRef := range schema.AllOf {
			fieldInfo := b.buildFromSchemaOrReference(name+"AllOf"+strconv.Itoa(idx+1), schemaOrRef)
			makeFieldAndAppendToType(fieldInfo, schemaType, "all_of_"+strconv.Itoa(idx+1))
		}

		if schema.Items != nil {
			for idx, schema := range schema.Items.Schema {
				fieldInfo := b.buildFromSchemaOrReference(name+"Items"+strconv.Itoa(idx+1), schema)
				makeFieldAndAppendToType(fieldInfo, schemaType, "items_"+strconv.Itoa(idx+1))
			}
		}

		if schema.Enum != nil {
			// TODO: It is not defined how enums should be represented inside the surface model
			fieldInfo := &FieldInfo{}
			fieldInfo.fieldKind, fieldInfo.fieldType, fieldInfo.fieldName = FieldKind_ANY, "string", "enum"
			makeFieldAndAppendToType(fieldInfo, schemaType, fieldInfo.fieldName)
		}

		if len(schemaType.Fields) == 0 {
			schemaType.Kind = TypeKind_OBJECT
			schemaType.ContentType = "interface{}"
		}
		b.model.addType(schemaType)
		fInfo.fieldKind, fInfo.fieldType = FieldKind_REFERENCE, schemaType.Name
		return fInfo
	case "array":
		// Same as for OpenAPI v3. I believe this is a bug: schema.Items.Schema should not be an array
		// but rather a single object describing the values of the array. Printing 'len(schema.Items.Schema)'
		// for 2000+ API descriptions from API-guru always resulted with an array of length of 1.
		for _, s := range schema.Items.Schema {
			arrayFieldInfo := b.buildFromSchemaOrReference(name, s)
			if arrayFieldInfo != nil {
				fInfo.fieldKind, fInfo.fieldType, fInfo.fieldFormat = FieldKind_ARRAY, arrayFieldInfo.fieldType, arrayFieldInfo.fieldFormat
				return fInfo
			}
		}
	default:
		// We got a scalar value
		fInfo.fieldKind, fInfo.fieldType, fInfo.fieldFormat = FieldKind_SCALAR, t, schema.Format
		return fInfo
	}
	log.Printf("Unimplemented: could not find field info for schema with name: '%v' and properties: %v", name, schema)
	return nil
}
