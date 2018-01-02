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

// schema-generator is a support tool that generates the OpenAPI v3 JSON schema.
// Yes, it's gross, but the OpenAPI 3.0 spec, which defines REST APIs with a
// rigorous JSON schema, is itself defined with a Markdown file. Ironic?
package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/googleapis/gnostic/jsonschema"
)

// convert the first character of a string to lower case
func lowerFirst(s string) string {
	if s == "" {
		return ""
	}
	r, n := utf8.DecodeRuneInString(s)
	return string(unicode.ToLower(r)) + s[n:]
}

// Section models a section of the OpenAPI specification text document.
type Section struct {
	Level    int
	Text     string
	Title    string
	Children []*Section
}

// ReadSection reads a section of the OpenAPI Specification, recursively dividing it into subsections
func ReadSection(text string, level int) (section *Section) {
	titlePattern := regexp.MustCompile("^" + strings.Repeat("#", level) + " .*$")
	subtitlePattern := regexp.MustCompile("^" + strings.Repeat("#", level+1) + " .*$")

	section = &Section{Level: level, Text: text}
	lines := strings.Split(string(text), "\n")
	subsection := ""
	for i, line := range lines {
		if i == 0 && titlePattern.Match([]byte(line)) {
			section.Title = line
		} else if subtitlePattern.Match([]byte(line)) {
			// we've found a subsection title.
			// if there's a subsection that we've already been reading, save it
			if len(subsection) != 0 {
				child := ReadSection(subsection, level+1)
				section.Children = append(section.Children, child)
			}
			// start a new subsection
			subsection = line + "\n"
		} else {
			// add to the subsection we've been reading
			subsection += line + "\n"
		}
	}
	// if this section has subsections, save the last one
	if len(section.Children) > 0 {
		child := ReadSection(subsection, level+1)
		section.Children = append(section.Children, child)
	}
	return
}

// Display recursively displays a section of the specification.
func (s *Section) Display(section string) {
	if len(s.Children) == 0 {
		//fmt.Printf("%s\n", s.Text)
	} else {
		for i, child := range s.Children {
			var subsection string
			if section == "" {
				subsection = fmt.Sprintf("%d", i)
			} else {
				subsection = fmt.Sprintf("%s.%d", section, i)
			}
			fmt.Printf("%-12s %s\n", subsection, child.NiceTitle())
			child.Display(subsection)
		}
	}
}

// remove a link from a string, leaving only the text that follows it
// if there is no link, just return the string
func stripLink(input string) (output string) {
	stringPattern := regexp.MustCompile("^(.*)$")
	stringWithLinkPattern := regexp.MustCompile("^<a .*</a>(.*)$")
	if matches := stringWithLinkPattern.FindSubmatch([]byte(input)); matches != nil {
		return string(matches[1])
	} else if matches := stringPattern.FindSubmatch([]byte(input)); matches != nil {
		return string(matches[1])
	} else {
		return input
	}
}

// NiceTitle returns a nice-to-display title for a section by removing the opening "###" and any links.
func (s *Section) NiceTitle() string {
	titlePattern := regexp.MustCompile("^#+ (.*)$")
	titleWithLinkPattern := regexp.MustCompile("^#+ <a .*</a>(.*)$")
	if matches := titleWithLinkPattern.FindSubmatch([]byte(s.Title)); matches != nil {
		return string(matches[1])
	} else if matches := titlePattern.FindSubmatch([]byte(s.Title)); matches != nil {
		return string(matches[1])
	} else {
		return ""
	}
}

// replace markdown links with their link text (removing the URL part)
func removeMarkdownLinks(input string) (output string) {
	markdownLink := regexp.MustCompile("\\[([^\\]\\[]*)\\]\\(([^\\)]*)\\)") // matches [link title](link url)
	output = string(markdownLink.ReplaceAll([]byte(input), []byte("$1")))
	return
}

// extract the fixed fields from a table in a section
func parseFixedFields(input string, schemaObject *SchemaObject) {
	lines := strings.Split(input, "\n")
	for _, line := range lines {

		// replace escaped bars with "OR", assuming these are used to describe union types
		line = strings.Replace(line, " \\| ", " OR ", -1)

		// split the table on the remaining bars
		parts := strings.Split(line, "|")
		if len(parts) > 1 {
			fieldName := strings.Trim(stripLink(parts[0]), " ")
			if fieldName != "Field Name" && fieldName != "---" {

				if len(parts) == 3 || len(parts) == 4 {
					// this is what we expect
				} else {
					log.Printf("ERROR: %+v", parts)
				}

				typeName := parts[1]
				typeName = strings.Replace(typeName, "{expression}", "Expression", -1)
				typeName = strings.Trim(typeName, " ")
				typeName = strings.Replace(typeName, "`", "", -1)
				typeName = removeMarkdownLinks(typeName)
				typeName = strings.Replace(typeName, " ", "", -1)
				typeName = strings.Replace(typeName, "Object", "", -1)
				isArray := false
				if typeName[0] == '[' && typeName[len(typeName)-1] == ']' {
					typeName = typeName[1 : len(typeName)-1]
					isArray = true
				}
				isMap := false
				mapPattern := regexp.MustCompile("^Mapstring,\\[(.*)\\]$")
				if matches := mapPattern.FindSubmatch([]byte(typeName)); matches != nil {
					typeName = string(matches[1])
					isMap = true
				} else {
					// match map[string,<typename>]
					mapPattern2 := regexp.MustCompile("^Map\\[string,(.+)\\]$")
					if matches := mapPattern2.FindSubmatch([]byte(typeName)); matches != nil {
						typeName = string(matches[1])
						isMap = true
					}
				}
				description := strings.Trim(parts[len(parts)-1], " ")
				description = removeMarkdownLinks(description)
				description = strings.Replace(description, "\n", " ", -1)

				requiredLabel1 := "**Required.** "
				requiredLabel2 := "**REQUIRED**."
				if strings.Contains(description, requiredLabel1) ||
					strings.Contains(description, requiredLabel2) {
					// only include required values if their "Validity" is "Any" or if no validity is specified
					valid := true
					if len(parts) == 4 {
						validity := parts[2]
						if strings.Contains(validity, "Any") {
							valid = true
						} else {
							valid = false
						}
					}
					if valid {
						schemaObject.RequiredFields = append(schemaObject.RequiredFields, fieldName)
					}
					description = strings.Replace(description, requiredLabel1, "", -1)
					description = strings.Replace(description, requiredLabel2, "", -1)
				}
				schemaField := SchemaObjectField{
					Name:        fieldName,
					Type:        typeName,
					IsArray:     isArray,
					IsMap:       isMap,
					Description: description,
				}
				schemaObject.FixedFields = append(schemaObject.FixedFields, schemaField)
			}
		}
	}
}

// extract the patterned fields from a table in a section
func parsePatternedFields(input string, schemaObject *SchemaObject) {
	lines := strings.Split(input, "\n")
	for _, line := range lines {

		line = strings.Replace(line, " \\| ", " OR ", -1)

		parts := strings.Split(line, "|")
		if len(parts) > 1 {
			fieldName := strings.Trim(stripLink(parts[0]), " ")
			fieldName = removeMarkdownLinks(fieldName)
			if fieldName == "HTTP Status Code" {
				fieldName = "^([0-9X]{3})$"
			}
			if fieldName != "Field Pattern" && fieldName != "---" {
				typeName := parts[1]
				typeName = strings.Trim(typeName, " ")
				typeName = strings.Replace(typeName, "`", "", -1)
				typeName = removeMarkdownLinks(typeName)
				typeName = strings.Replace(typeName, " ", "", -1)
				typeName = strings.Replace(typeName, "Object", "", -1)
				typeName = strings.Replace(typeName, "{expression}", "Expression", -1)
				isArray := false
				if typeName[0] == '[' && typeName[len(typeName)-1] == ']' {
					typeName = typeName[1 : len(typeName)-1]
					isArray = true
				}
				isMap := false
				mapPattern := regexp.MustCompile("^Mapstring,\\[(.*)\\]$")
				if matches := mapPattern.FindSubmatch([]byte(typeName)); matches != nil {
					typeName = string(matches[1])
					isMap = true
				}
				description := strings.Trim(parts[len(parts)-1], " ")
				description = removeMarkdownLinks(description)
				description = strings.Replace(description, "\n", " ", -1)

				schemaField := SchemaObjectField{
					Name:        fieldName,
					Type:        typeName,
					IsArray:     isArray,
					IsMap:       isMap,
					Description: description,
				}
				schemaObject.PatternedFields = append(schemaObject.PatternedFields, schemaField)
			}
		}
	}
}

// SchemaObjectField describes a field of a schema.
type SchemaObjectField struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	IsArray     bool   `json:"is_array"`
	IsMap       bool   `json:"is_map"`
	Description string `json:"description"`
}

// SchemaObject describes a schema.
type SchemaObject struct {
	Name            string              `json:"name"`
	ID              string              `json:"id"`
	Description     string              `json:"description"`
	Extendable      bool                `json:"extendable"`
	RequiredFields  []string            `json:"required"`
	FixedFields     []SchemaObjectField `json:"fixed"`
	PatternedFields []SchemaObjectField `json:"patterned"`
}

// SchemaModel is a collection of schemas.
type SchemaModel struct {
	Objects []SchemaObject
}

func (m *SchemaModel) objectWithID(id string) *SchemaObject {
	for _, object := range m.Objects {
		if object.ID == id {
			return &object
		}
	}
	return nil
}

// NewSchemaModel returns a new SchemaModel.
func NewSchemaModel(filename string) (schemaModel *SchemaModel, err error) {

	b, err := ioutil.ReadFile("3.0.md")
	if err != nil {
		return nil, err
	}

	// divide the specification into sections
	document := ReadSection(string(b), 1)
	document.Display("")

	// read object names and their details
	specification := document.Children[4] // fragile! the section title is "Specification"
	schema := specification.Children[7]   // fragile! the section title is "Schema"
	anchor := regexp.MustCompile("^#### <a name=\"(.*)Object\"")
	schemaObjects := make([]SchemaObject, 0)
	for _, section := range schema.Children {
		if matches := anchor.FindSubmatch([]byte(section.Title)); matches != nil {

			id := string(matches[1])

			schemaObject := SchemaObject{
				Name:           section.NiceTitle(),
				ID:             id,
				RequiredFields: nil,
			}

			if len(section.Children) > 0 {
				description := section.Children[0].Text
				description = removeMarkdownLinks(description)
				description = strings.Trim(description, " \t\n")
				description = strings.Replace(description, "\n", " ", -1)
				schemaObject.Description = description
			}

			// is the object extendable?
			if strings.Contains(section.Text, "Specification Extensions") {
				schemaObject.Extendable = true
			}

			// look for fixed fields
			for _, child := range section.Children {
				if child.NiceTitle() == "Fixed Fields" {
					parseFixedFields(child.Text, &schemaObject)
				}
			}

			// look for patterned fields
			for _, child := range section.Children {
				if child.NiceTitle() == "Patterned Fields" {
					parsePatternedFields(child.Text, &schemaObject)
				}
			}

			schemaObjects = append(schemaObjects, schemaObject)
		}
	}

	return &SchemaModel{Objects: schemaObjects}, nil
}

// UnionType represents a union of two types.
type UnionType struct {
	Name        string
	ObjectType1 string
	ObjectType2 string
}

var unionTypes map[string]*UnionType

func noteUnionType(typeName, objectType1, objectType2 string) {
	if unionTypes == nil {
		unionTypes = make(map[string]*UnionType, 0)
	}
	unionTypes[typeName] = &UnionType{
		Name:        typeName,
		ObjectType1: objectType1,
		ObjectType2: objectType2,
	}
}

// MapType represents a map of a specified type (with string keys).
type MapType struct {
	Name       string
	ObjectType string
}

var mapTypes map[string]*MapType

func noteMapType(typeName, objectType string) {
	if mapTypes == nil {
		mapTypes = make(map[string]*MapType, 0)
	}
	mapTypes[typeName] = &MapType{
		Name:       typeName,
		ObjectType: objectType,
	}
}

func definitionNameForType(typeName string) string {
	name := typeName
	switch typeName {
	case "OAuthFlows":
		name = "oauthFlows"
	case "OAuthFlow":
		name = "oauthFlow"
	case "XML":
		name = "xml"
	case "ExternalDocumentation":
		name = "externalDocs"
	default:
		// does the name contain an "OR"
		if parts := strings.Split(typeName, "OR"); len(parts) > 1 {
			name = lowerFirst(parts[0]) + "Or" + parts[1]
			noteUnionType(name, parts[0], parts[1])
		} else {
			name = lowerFirst(typeName)
		}
	}
	return "#/definitions/" + name
}

func pluralize(name string) string {
	if name == "any" {
		return "anys"
	}
	switch name[len(name)-1] {
	case 'y':
		name = name[0:len(name)-1] + "ies"
	case 's':
		name = name + "Map"
	default:
		name = name + "s"
	}
	return name
}

func definitionNameForMapOfType(typeName string) string {
	// pluralize the type name to get the name of an object representing a map of them
	var elementTypeName string
	var mapTypeName string
	if parts := strings.Split(typeName, "OR"); len(parts) > 1 {
		elementTypeName = lowerFirst(parts[0]) + "Or" + parts[1]
		noteUnionType(elementTypeName, parts[0], parts[1])
		mapTypeName = pluralize(lowerFirst(parts[0])) + "Or" + pluralize(parts[1])
	} else {
		elementTypeName = lowerFirst(typeName)
		mapTypeName = pluralize(elementTypeName)
	}
	noteMapType(mapTypeName, elementTypeName)
	return "#/definitions/" + mapTypeName
}

func updateSchemaFieldWithModelField(schemaField *jsonschema.Schema, modelField *SchemaObjectField) {
	// fmt.Printf("IN %s:%+v\n", name, schemaField)
	// update the attributes of the schema field
	if modelField.IsArray {
		// is array
		itemSchema := &jsonschema.Schema{}
		switch modelField.Type {
		case "string":
			itemSchema.Type = jsonschema.NewStringOrStringArrayWithString("string")
		case "boolean":
			itemSchema.Type = jsonschema.NewStringOrStringArrayWithString("boolean")
		case "primitive":
			itemSchema.Ref = stringptr(definitionNameForType("Primitive"))
		default:
			itemSchema.Ref = stringptr(definitionNameForType(modelField.Type))
		}
		schemaField.Items = jsonschema.NewSchemaOrSchemaArrayWithSchema(itemSchema)
		schemaField.Type = jsonschema.NewStringOrStringArrayWithString("array")
		boolValue := true // not sure about this
		schemaField.UniqueItems = &boolValue
	} else if modelField.IsMap {
		schemaField.Ref = stringptr(definitionNameForMapOfType(modelField.Type))
	} else {
		// is scalar
		switch modelField.Type {
		case "string":
			schemaField.Type = jsonschema.NewStringOrStringArrayWithString("string")
		case "boolean":
			schemaField.Type = jsonschema.NewStringOrStringArrayWithString("boolean")
		case "primitive":
			schemaField.Ref = stringptr(definitionNameForType("Primitive"))
		default:
			schemaField.Ref = stringptr(definitionNameForType(modelField.Type))
		}
	}
}

func buildSchemaWithModel(modelObject *SchemaObject) (schema *jsonschema.Schema) {

	schema = &jsonschema.Schema{}
	schema.Type = jsonschema.NewStringOrStringArrayWithString("object")

	if modelObject.RequiredFields != nil && len(modelObject.RequiredFields) > 0 {
		// copy array
		arrayCopy := modelObject.RequiredFields
		schema.Required = &arrayCopy
	}

	schema.AdditionalProperties = jsonschema.NewSchemaOrBooleanWithBoolean(false)

	schema.Description = stringptr(modelObject.Description)

	// handle fixed fields
	if modelObject.FixedFields != nil {
		newNamedSchemas := make([]*jsonschema.NamedSchema, 0)
		for _, modelField := range modelObject.FixedFields {
			schemaField := schema.PropertyWithName(modelField.Name)
			if schemaField == nil {
				// create and add the schema field
				schemaField = &jsonschema.Schema{}
				namedSchema := &jsonschema.NamedSchema{Name: modelField.Name, Value: schemaField}
				newNamedSchemas = append(newNamedSchemas, namedSchema)
			}
			updateSchemaFieldWithModelField(schemaField, &modelField)
		}
		for _, pair := range newNamedSchemas {
			if schema.Properties == nil {
				properties := make([]*jsonschema.NamedSchema, 0)
				schema.Properties = &properties
			}
			*(schema.Properties) = append(*(schema.Properties), pair)
		}

	} else {
		if schema.Properties != nil {
			fmt.Printf("SCHEMA SHOULD NOT HAVE PROPERTIES %s\n", modelObject.ID)
		}
	}

	// handle patterned fields
	if modelObject.PatternedFields != nil {
		newNamedSchemas := make([]*jsonschema.NamedSchema, 0)

		for _, modelField := range modelObject.PatternedFields {
			schemaField := schema.PatternPropertyWithName(modelField.Name)
			if schemaField == nil {
				// create and add the schema field
				schemaField = &jsonschema.Schema{}
				// Component names should match "^[a-zA-Z0-9\.\-_]+$"
				// See https://github.com/OAI/OpenAPI-Specification/blob/OpenAPI.next/versions/3.0.md#componentsObject
				nameRegex := "^[a-zA-Z0-9\\\\.\\\\-_]+$"
				if modelObject.Name == "Scopes Object" {
					nameRegex = "^"
				} else if modelObject.Name == "Headers Object" {
					nameRegex = "^[a-zA-Z0-9!#\\-\\$%&'\\*\\+\\\\\\.\\^_`\\|~]+"
				}
				propertyName := strings.Replace(modelField.Name, "{name}", nameRegex, -1)
				//  The field name MUST begin with a slash, see https://github.com/OAI/OpenAPI-Specification/blob/OpenAPI.next/versions/3.0.md#paths-object
				// JSON Schema for OpenAPI v2 uses "^/" as regex for paths, see https://github.com/OAI/OpenAPI-Specification/blob/OpenAPI.next/schemas/v2.0/schema.json#L173
				propertyName = strings.Replace(propertyName, "/{path}", "^/", -1)
				// Replace human-friendly (and regex-confusing) description with a blank pattern
				propertyName = strings.Replace(propertyName, "{expression}", "^", -1)
				propertyName = strings.Replace(propertyName, "{property}", "^", -1)
				namedSchema := &jsonschema.NamedSchema{Name: propertyName, Value: schemaField}
				newNamedSchemas = append(newNamedSchemas, namedSchema)
			}
			updateSchemaFieldWithModelField(schemaField, &modelField)
		}

		for _, pair := range newNamedSchemas {
			if schema.PatternProperties == nil {
				properties := make([]*jsonschema.NamedSchema, 0)
				schema.PatternProperties = &properties
			}
			*(schema.PatternProperties) = append(*(schema.PatternProperties), pair)
		}

	} else {
		if schema.PatternProperties != nil && !modelObject.Extendable {
			fmt.Printf("SCHEMA SHOULD NOT HAVE PATTERN PROPERTIES %s\n", modelObject.ID)
		}
	}

	if modelObject.Extendable {
		schemaField := schema.PatternPropertyWithName("^x-")
		if schemaField != nil {
			schemaField.Ref = stringptr("#/definitions/specificationExtension")
		} else {
			schemaField = &jsonschema.Schema{}
			schemaField.Ref = stringptr("#/definitions/specificationExtension")
			namedSchema := &jsonschema.NamedSchema{Name: "^x-", Value: schemaField}
			if schema.PatternProperties == nil {
				properties := make([]*jsonschema.NamedSchema, 0)
				schema.PatternProperties = &properties
			}
			*(schema.PatternProperties) = append(*(schema.PatternProperties), namedSchema)
		}
	} else {
		schemaField := schema.PatternPropertyWithName("^x-")
		if schemaField != nil {
			fmt.Printf("INVALID EXTENSION SUPPORT %s:%s\n", modelObject.ID, "^x-")
		}
	}

	return schema
}

// return a pointer to a copy of a passed-in string
func stringptr(input string) (output *string) {
	return &input
}

func int64ptr(input int64) (output *int64) {
	return &input
}

func arrayOfSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		Type:     jsonschema.NewStringOrStringArrayWithString("array"),
		MinItems: int64ptr(1),
		Items:    jsonschema.NewSchemaOrSchemaArrayWithSchema(&jsonschema.Schema{Ref: stringptr("#/definitions/schemaOrReference")}),
	}
}

func main() {
	// read and parse the text specification into a model structure
	model, err := NewSchemaModel("3.0.md")
	if err != nil {
		panic(err)
	}

	// write the model as JSON (for debugging)
	modelJSON, _ := json.MarshalIndent(model, "", "  ")
	err = ioutil.WriteFile("model.json", modelJSON, 0644)
	if err != nil {
		panic(err)
	}

	// build the top-level schema using the "OAS" model
	oasModel := model.objectWithID("oas")
	if oasModel == nil {
		log.Printf("Unable to find OAS model. Has the source document structure changed?")
		os.Exit(-1)
	}
	schema := buildSchemaWithModel(oasModel)

	// manually set a few fields
	schema.Title = stringptr("A JSON Schema for OpenAPI 3.0.")
	schema.ID = stringptr("http://openapis.org/v3/schema.json#")
	schema.Schema = stringptr("http://json-schema.org/draft-04/schema#")

	// loop over all models and create the corresponding schema objects
	definitions := make([]*jsonschema.NamedSchema, 0)
	schema.Definitions = &definitions

	for _, modelObject := range model.Objects {
		if modelObject.ID == "oas" {
			continue
		}
		definitionSchema := buildSchemaWithModel(&modelObject)
		name := modelObject.ID
		if name == "externalDocumentation" {
			name = "externalDocs"
		}
		*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema(name, definitionSchema))
	}

	// copy the properties of headerObject from parameterObject
	headerObject := schema.DefinitionWithName("header")
	parameterObject := schema.DefinitionWithName("parameter")
	if parameterObject != nil {
		newArray := make([]*jsonschema.NamedSchema, 0)
		for _, property := range *(parameterObject.Properties) {
			// we need to remove a few properties...
			if property.Name != "name" && property.Name != "in" {
				newArray = append(newArray, property)
			}
		}
		headerObject.Properties = &newArray
		// "So a shorthand for copying array arr would be tmp := append([]int{}, arr...)"
		ppArray := make([]*jsonschema.NamedSchema, 0)
		ppArray = append(ppArray, *(parameterObject.PatternProperties)...)
		headerObject.PatternProperties = &ppArray
	}

	// generate implied union types
	unionTypeKeys := make([]string, 0, len(unionTypes))
	for key := range unionTypes {
		unionTypeKeys = append(unionTypeKeys, key)
	}
	sort.Strings(unionTypeKeys)
	for _, unionTypeKey := range unionTypeKeys {
		unionType := unionTypes[unionTypeKey]
		objectSchema := schema.DefinitionWithName(unionType.Name)
		if objectSchema == nil {
			objectSchema = &jsonschema.Schema{}
			oneOf := make([]*jsonschema.Schema, 0)
			oneOf = append(oneOf, &jsonschema.Schema{Ref: stringptr("#/definitions/" + lowerFirst(unionType.ObjectType1))})
			oneOf = append(oneOf, &jsonschema.Schema{Ref: stringptr("#/definitions/" + lowerFirst(unionType.ObjectType2))})
			objectSchema.OneOf = &oneOf
			*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema(unionType.Name, objectSchema))
		}
	}

	// generate implied map types
	mapTypeKeys := make([]string, 0, len(mapTypes))
	for key := range mapTypes {
		mapTypeKeys = append(mapTypeKeys, key)
	}
	sort.Strings(mapTypeKeys)
	for _, mapTypeKey := range mapTypeKeys {
		mapType := mapTypes[mapTypeKey]
		objectSchema := schema.DefinitionWithName(mapType.Name)
		if objectSchema == nil {
			objectSchema = &jsonschema.Schema{}
			objectSchema.Type = jsonschema.NewStringOrStringArrayWithString("object")
			additionalPropertiesSchema := &jsonschema.Schema{}
			if mapType.ObjectType == "string" {
				additionalPropertiesSchema.Type = jsonschema.NewStringOrStringArrayWithString("string")
			} else {
				additionalPropertiesSchema.Ref = stringptr("#/definitions/" + lowerFirst(mapType.ObjectType))
			}
			objectSchema.AdditionalProperties = jsonschema.NewSchemaOrBooleanWithSchema(additionalPropertiesSchema)
			*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema(mapType.Name, objectSchema))
		}
	}

	// add schema objects for "object", "any", and "expression"
	if true {
		objectSchema := &jsonschema.Schema{}
		objectSchema.Type = jsonschema.NewStringOrStringArrayWithString("object")
		objectSchema.AdditionalProperties = jsonschema.NewSchemaOrBooleanWithBoolean(true)
		*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema("object", objectSchema))
	}
	if true {
		objectSchema := &jsonschema.Schema{}
		objectSchema.AdditionalProperties = jsonschema.NewSchemaOrBooleanWithBoolean(true)
		*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema("any", objectSchema))
	}
	if true {
		objectSchema := &jsonschema.Schema{}
		objectSchema.Type = jsonschema.NewStringOrStringArrayWithString("object")
		objectSchema.AdditionalProperties = jsonschema.NewSchemaOrBooleanWithBoolean(true)
		*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema("expression", objectSchema))
	}

	// add schema objects for "specificationExtension"
	if true {
		objectSchema := &jsonschema.Schema{}
		objectSchema.Description = stringptr("Any property starting with x- is valid.")
		oneOf := make([]*jsonschema.Schema, 0)
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("null")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("number")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("boolean")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("string")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("object")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("array")})
		objectSchema.OneOf = &oneOf
		*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema("specificationExtension", objectSchema))
	}

	// add schema objects for "defaultType"
	if true {
		objectSchema := &jsonschema.Schema{}
		oneOf := make([]*jsonschema.Schema, 0)
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("null")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("array")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("object")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("number")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("boolean")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("string")})
		objectSchema.OneOf = &oneOf
		*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema("defaultType", objectSchema))
	}

	// add schema objects for "primitive"
	if false { // we don't seem to need these for 3.0 RC2
		objectSchema := &jsonschema.Schema{}
		oneOf := make([]*jsonschema.Schema, 0)
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("number")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("boolean")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("string")})
		objectSchema.OneOf = &oneOf
		*schema.Definitions = append(*schema.Definitions, jsonschema.NewNamedSchema("primitive", objectSchema))
	}

	// force a few more things into the "schema" schema
	schemaObject := schema.DefinitionWithName("schema")
	schemaObject.CopyOfficialSchemaProperties(
		[]string{
			"title",
			"multipleOf",
			"maximum",
			"exclusiveMaximum",
			"minimum",
			"exclusiveMinimum",
			"maxLength",
			"minLength",
			"pattern",
			"maxItems",
			"minItems",
			"uniqueItems",
			"maxProperties",
			"minProperties",
			"required",
			"enum",
		})
	schemaObject.AdditionalProperties = jsonschema.NewSchemaOrBooleanWithBoolean(false)
	schemaObject.AddProperty("type", &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("string")})
	schemaObject.AddProperty("allOf", arrayOfSchema())
	schemaObject.AddProperty("oneOf", arrayOfSchema())
	schemaObject.AddProperty("anyOf", arrayOfSchema())
	schemaObject.AddProperty("not", &jsonschema.Schema{Ref: stringptr("#/definitions/schema")})
	anyOf := make([]*jsonschema.Schema, 0)
	anyOf = append(anyOf, &jsonschema.Schema{Ref: stringptr("#/definitions/schemaOrReference")})
	anyOf = append(anyOf, arrayOfSchema())
	schemaObject.AddProperty("items",
		&jsonschema.Schema{AnyOf: &anyOf})
	schemaObject.AddProperty("properties", &jsonschema.Schema{
		Type: jsonschema.NewStringOrStringArrayWithString("object"),
		AdditionalProperties: jsonschema.NewSchemaOrBooleanWithSchema(
			&jsonschema.Schema{Ref: stringptr("#/definitions/schemaOrReference")})})

	if true { // add additionalProperties schema object
		oneOf := make([]*jsonschema.Schema, 0)
		oneOf = append(oneOf, &jsonschema.Schema{Ref: stringptr("#/definitions/schemaOrReference")})
		oneOf = append(oneOf, &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("boolean")})
		schemaObject.AddProperty("additionalProperties", &jsonschema.Schema{OneOf: &oneOf})
	}

	schemaObject.AddProperty("default", &jsonschema.Schema{Ref: stringptr("#/definitions/defaultType")})
	schemaObject.AddProperty("description", &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("string")})
	schemaObject.AddProperty("format", &jsonschema.Schema{Type: jsonschema.NewStringOrStringArrayWithString("string")})

	// fix the content object
	contentObject := schema.DefinitionWithName("content")
	if contentObject != nil {
		pairs := make([]*jsonschema.NamedSchema, 0)
		contentObject.PatternProperties = &pairs
		namedSchema := &jsonschema.NamedSchema{Name: "^", Value: &jsonschema.Schema{Ref: stringptr("#/definitions/mediaType")}}
		*(contentObject.PatternProperties) = append(*(contentObject.PatternProperties), namedSchema)
	}

	// write the updated schema
	output := schema.JSONString()
	err = ioutil.WriteFile("schema.json", []byte(output), 0644)
	if err != nil {
		panic(err)
	}
}
