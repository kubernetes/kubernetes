// Copyright 2017 Google Inc. All Rights Reserved.
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

package jsonschema

import (
	"fmt"
	"log"
	"strings"
)

//
// OPERATIONS
// The following methods perform operations on Schemas.
//

// IsEmpty returns true if no members of the Schema are specified.
func (schema *Schema) IsEmpty() bool {
	return (schema.Schema == nil) &&
		(schema.ID == nil) &&
		(schema.MultipleOf == nil) &&
		(schema.Maximum == nil) &&
		(schema.ExclusiveMaximum == nil) &&
		(schema.Minimum == nil) &&
		(schema.ExclusiveMinimum == nil) &&
		(schema.MaxLength == nil) &&
		(schema.MinLength == nil) &&
		(schema.Pattern == nil) &&
		(schema.AdditionalItems == nil) &&
		(schema.Items == nil) &&
		(schema.MaxItems == nil) &&
		(schema.MinItems == nil) &&
		(schema.UniqueItems == nil) &&
		(schema.MaxProperties == nil) &&
		(schema.MinProperties == nil) &&
		(schema.Required == nil) &&
		(schema.AdditionalProperties == nil) &&
		(schema.Properties == nil) &&
		(schema.PatternProperties == nil) &&
		(schema.Dependencies == nil) &&
		(schema.Enumeration == nil) &&
		(schema.Type == nil) &&
		(schema.AllOf == nil) &&
		(schema.AnyOf == nil) &&
		(schema.OneOf == nil) &&
		(schema.Not == nil) &&
		(schema.Definitions == nil) &&
		(schema.Title == nil) &&
		(schema.Description == nil) &&
		(schema.Default == nil) &&
		(schema.Format == nil) &&
		(schema.Ref == nil)
}

// IsEqual returns true if two schemas are equal.
func (schema *Schema) IsEqual(schema2 *Schema) bool {
	return schema.String() == schema2.String()
}

// SchemaOperation represents a function that can be applied to a Schema.
type SchemaOperation func(schema *Schema, context string)

// Applies a specified function to a Schema and all of the Schemas that it contains.
func (schema *Schema) applyToSchemas(operation SchemaOperation, context string) {

	if schema.AdditionalItems != nil {
		s := schema.AdditionalItems.Schema
		if s != nil {
			s.applyToSchemas(operation, "AdditionalItems")
		}
	}

	if schema.Items != nil {
		if schema.Items.SchemaArray != nil {
			for _, s := range *(schema.Items.SchemaArray) {
				s.applyToSchemas(operation, "Items.SchemaArray")
			}
		} else if schema.Items.Schema != nil {
			schema.Items.Schema.applyToSchemas(operation, "Items.Schema")
		}
	}

	if schema.AdditionalProperties != nil {
		s := schema.AdditionalProperties.Schema
		if s != nil {
			s.applyToSchemas(operation, "AdditionalProperties")
		}
	}

	if schema.Properties != nil {
		for _, pair := range *(schema.Properties) {
			s := pair.Value
			s.applyToSchemas(operation, "Properties")
		}
	}
	if schema.PatternProperties != nil {
		for _, pair := range *(schema.PatternProperties) {
			s := pair.Value
			s.applyToSchemas(operation, "PatternProperties")
		}
	}

	if schema.Dependencies != nil {
		for _, pair := range *(schema.Dependencies) {
			schemaOrStringArray := pair.Value
			s := schemaOrStringArray.Schema
			if s != nil {
				s.applyToSchemas(operation, "Dependencies")
			}
		}
	}

	if schema.AllOf != nil {
		for _, s := range *(schema.AllOf) {
			s.applyToSchemas(operation, "AllOf")
		}
	}
	if schema.AnyOf != nil {
		for _, s := range *(schema.AnyOf) {
			s.applyToSchemas(operation, "AnyOf")
		}
	}
	if schema.OneOf != nil {
		for _, s := range *(schema.OneOf) {
			s.applyToSchemas(operation, "OneOf")
		}
	}
	if schema.Not != nil {
		schema.Not.applyToSchemas(operation, "Not")
	}

	if schema.Definitions != nil {
		for _, pair := range *(schema.Definitions) {
			s := pair.Value
			s.applyToSchemas(operation, "Definitions")
		}
	}

	operation(schema, context)
}

// CopyProperties copies all non-nil properties from the source Schema to the schema Schema.
func (schema *Schema) CopyProperties(source *Schema) {
	if source.Schema != nil {
		schema.Schema = source.Schema
	}
	if source.ID != nil {
		schema.ID = source.ID
	}
	if source.MultipleOf != nil {
		schema.MultipleOf = source.MultipleOf
	}
	if source.Maximum != nil {
		schema.Maximum = source.Maximum
	}
	if source.ExclusiveMaximum != nil {
		schema.ExclusiveMaximum = source.ExclusiveMaximum
	}
	if source.Minimum != nil {
		schema.Minimum = source.Minimum
	}
	if source.ExclusiveMinimum != nil {
		schema.ExclusiveMinimum = source.ExclusiveMinimum
	}
	if source.MaxLength != nil {
		schema.MaxLength = source.MaxLength
	}
	if source.MinLength != nil {
		schema.MinLength = source.MinLength
	}
	if source.Pattern != nil {
		schema.Pattern = source.Pattern
	}
	if source.AdditionalItems != nil {
		schema.AdditionalItems = source.AdditionalItems
	}
	if source.Items != nil {
		schema.Items = source.Items
	}
	if source.MaxItems != nil {
		schema.MaxItems = source.MaxItems
	}
	if source.MinItems != nil {
		schema.MinItems = source.MinItems
	}
	if source.UniqueItems != nil {
		schema.UniqueItems = source.UniqueItems
	}
	if source.MaxProperties != nil {
		schema.MaxProperties = source.MaxProperties
	}
	if source.MinProperties != nil {
		schema.MinProperties = source.MinProperties
	}
	if source.Required != nil {
		schema.Required = source.Required
	}
	if source.AdditionalProperties != nil {
		schema.AdditionalProperties = source.AdditionalProperties
	}
	if source.Properties != nil {
		schema.Properties = source.Properties
	}
	if source.PatternProperties != nil {
		schema.PatternProperties = source.PatternProperties
	}
	if source.Dependencies != nil {
		schema.Dependencies = source.Dependencies
	}
	if source.Enumeration != nil {
		schema.Enumeration = source.Enumeration
	}
	if source.Type != nil {
		schema.Type = source.Type
	}
	if source.AllOf != nil {
		schema.AllOf = source.AllOf
	}
	if source.AnyOf != nil {
		schema.AnyOf = source.AnyOf
	}
	if source.OneOf != nil {
		schema.OneOf = source.OneOf
	}
	if source.Not != nil {
		schema.Not = source.Not
	}
	if source.Definitions != nil {
		schema.Definitions = source.Definitions
	}
	if source.Title != nil {
		schema.Title = source.Title
	}
	if source.Description != nil {
		schema.Description = source.Description
	}
	if source.Default != nil {
		schema.Default = source.Default
	}
	if source.Format != nil {
		schema.Format = source.Format
	}
	if source.Ref != nil {
		schema.Ref = source.Ref
	}
}

// TypeIs returns true if the Type of a Schema includes the specified type
func (schema *Schema) TypeIs(typeName string) bool {
	if schema.Type != nil {
		// the schema Type is either a string or an array of strings
		if schema.Type.String != nil {
			return (*(schema.Type.String) == typeName)
		} else if schema.Type.StringArray != nil {
			for _, n := range *(schema.Type.StringArray) {
				if n == typeName {
					return true
				}
			}
		}
	}
	return false
}

// ResolveRefs resolves "$ref" elements in a Schema and its children.
// But if a reference refers to an object type, is inside a oneOf, or contains a oneOf,
// the reference is kept and we expect downstream tools to separately model these
// referenced schemas.
func (schema *Schema) ResolveRefs() {
	rootSchema := schema
	count := 1
	for count > 0 {
		count = 0
		schema.applyToSchemas(
			func(schema *Schema, context string) {
				if schema.Ref != nil {
					resolvedRef, err := rootSchema.resolveJSONPointer(*(schema.Ref))
					if err != nil {
						log.Printf("%+v", err)
					} else if resolvedRef.TypeIs("object") {
						// don't substitute for objects, we'll model the referenced schema with a class
					} else if context == "OneOf" {
						// don't substitute for references inside oneOf declarations
					} else if resolvedRef.OneOf != nil {
						// don't substitute for references that contain oneOf declarations
					} else if resolvedRef.AdditionalProperties != nil {
						// don't substitute for references that look like objects
					} else {
						schema.Ref = nil
						schema.CopyProperties(resolvedRef)
						count++
					}
				}
			}, "")
	}
}

// resolveJSONPointer resolves JSON pointers.
// This current implementation is very crude and custom for OpenAPI 2.0 schemas.
// It panics for any pointer that it is unable to resolve.
func (schema *Schema) resolveJSONPointer(ref string) (result *Schema, err error) {
	parts := strings.Split(ref, "#")
	if len(parts) == 2 {
		documentName := parts[0] + "#"
		if documentName == "#" && schema.ID != nil {
			documentName = *(schema.ID)
		}
		path := parts[1]
		document := schemas[documentName]
		pathParts := strings.Split(path, "/")

		// we currently do a very limited (hard-coded) resolution of certain paths and log errors for missed cases
		if len(pathParts) == 1 {
			return document, nil
		} else if len(pathParts) == 3 {
			switch pathParts[1] {
			case "definitions":
				dictionary := document.Definitions
				for _, pair := range *dictionary {
					if pair.Name == pathParts[2] {
						result = pair.Value
					}
				}
			case "properties":
				dictionary := document.Properties
				for _, pair := range *dictionary {
					if pair.Name == pathParts[2] {
						result = pair.Value
					}
				}
			default:
				break
			}
		}
	}
	if result == nil {
		return nil, fmt.Errorf("unresolved pointer: %+v", ref)
	}
	return result, nil
}

// ResolveAllOfs replaces "allOf" elements by merging their properties into the parent Schema.
func (schema *Schema) ResolveAllOfs() {
	schema.applyToSchemas(
		func(schema *Schema, context string) {
			if schema.AllOf != nil {
				for _, allOf := range *(schema.AllOf) {
					schema.CopyProperties(allOf)
				}
				schema.AllOf = nil
			}
		}, "resolveAllOfs")
}

// ResolveAnyOfs replaces all "anyOf" elements with "oneOf".
func (schema *Schema) ResolveAnyOfs() {
	schema.applyToSchemas(
		func(schema *Schema, context string) {
			if schema.AnyOf != nil {
				schema.OneOf = schema.AnyOf
				schema.AnyOf = nil
			}
		}, "resolveAnyOfs")
}

// return a pointer to a copy of a passed-in string
func stringptr(input string) (output *string) {
	return &input
}

// CopyOfficialSchemaProperty copies a named property from the official JSON Schema definition
func (schema *Schema) CopyOfficialSchemaProperty(name string) {
	*schema.Properties = append(*schema.Properties,
		NewNamedSchema(name,
			&Schema{Ref: stringptr("http://json-schema.org/draft-04/schema#/properties/" + name)}))
}

// CopyOfficialSchemaProperties copies named properties from the official JSON Schema definition
func (schema *Schema) CopyOfficialSchemaProperties(names []string) {
	for _, name := range names {
		schema.CopyOfficialSchemaProperty(name)
	}
}
