package swagger

import (
	"encoding/json"
	"reflect"
	"strings"
)

// ModelBuildable is used for extending Structs that need more control over
// how the Model appears in the Swagger api declaration.
type ModelBuildable interface {
	PostBuildModel(m *Model) *Model
}

type modelBuilder struct {
	Models *ModelList
	Config *Config
}

type documentable interface {
	SwaggerDoc() map[string]string
}

// Check if this structure has a method with signature func (<theModel>) SwaggerDoc() map[string]string
// If it exists, retrive the documentation and overwrite all struct tag descriptions
func getDocFromMethodSwaggerDoc2(model reflect.Type) map[string]string {
	if docable, ok := reflect.New(model).Elem().Interface().(documentable); ok {
		return docable.SwaggerDoc()
	}
	return make(map[string]string)
}

// addModelFrom creates and adds a Model to the builder and detects and calls
// the post build hook for customizations
func (b modelBuilder) addModelFrom(sample interface{}) {
	if modelOrNil := b.addModel(reflect.TypeOf(sample), ""); modelOrNil != nil {
		// allow customizations
		if buildable, ok := sample.(ModelBuildable); ok {
			modelOrNil = buildable.PostBuildModel(modelOrNil)
			b.Models.Put(modelOrNil.Id, *modelOrNil)
		}
	}
}

func (b modelBuilder) addModel(st reflect.Type, nameOverride string) *Model {
	modelName := b.keyFrom(st)
	if nameOverride != "" {
		modelName = nameOverride
	}
	// no models needed for primitive types
	if b.isPrimitiveType(modelName) {
		return nil
	}
	// see if we already have visited this model
	if _, ok := b.Models.At(modelName); ok {
		return nil
	}
	sm := Model{
		Id:         modelName,
		Required:   []string{},
		Properties: ModelPropertyList{}}

	// reference the model before further initializing (enables recursive structs)
	b.Models.Put(modelName, sm)

	// check for slice or array
	if st.Kind() == reflect.Slice || st.Kind() == reflect.Array {
		b.addModel(st.Elem(), "")
		return &sm
	}
	// check for structure or primitive type
	if st.Kind() != reflect.Struct {
		return &sm
	}

	fullDoc := getDocFromMethodSwaggerDoc2(st)
	modelDescriptions := []string{}

	for i := 0; i < st.NumField(); i++ {
		field := st.Field(i)
		jsonName, modelDescription, prop := b.buildProperty(field, &sm, modelName)
		if len(modelDescription) > 0 {
			modelDescriptions = append(modelDescriptions, modelDescription)
		}

		// add if not omitted
		if len(jsonName) != 0 {
			// update description
			if fieldDoc, ok := fullDoc[jsonName]; ok {
				prop.Description = fieldDoc
			}
			// update Required
			if b.isPropertyRequired(field) {
				sm.Required = append(sm.Required, jsonName)
			}
			sm.Properties.Put(jsonName, prop)
		}
	}

	// We always overwrite documentation if SwaggerDoc method exists
	// "" is special for documenting the struct itself
	if modelDoc, ok := fullDoc[""]; ok {
		sm.Description = modelDoc
	} else if len(modelDescriptions) != 0 {
		sm.Description = strings.Join(modelDescriptions, "\n")
	}

	// update model builder with completed model
	b.Models.Put(modelName, sm)

	return &sm
}

func (b modelBuilder) isPropertyRequired(field reflect.StructField) bool {
	required := true
	if jsonTag := field.Tag.Get("json"); jsonTag != "" {
		s := strings.Split(jsonTag, ",")
		if len(s) > 1 && s[1] == "omitempty" {
			return false
		}
	}
	return required
}

func (b modelBuilder) buildProperty(field reflect.StructField, model *Model, modelName string) (jsonName, modelDescription string, prop ModelProperty) {
	jsonName = b.jsonNameOfField(field)
	if len(jsonName) == 0 {
		// empty name signals skip property
		return "", "", prop
	}

	if tag := field.Tag.Get("modelDescription"); tag != "" {
		modelDescription = tag
	}

	prop.setPropertyMetadata(field)
	if prop.Type != nil {
		return jsonName, modelDescription, prop
	}
	fieldType := field.Type

	// check if type is doing its own marshalling
	marshalerType := reflect.TypeOf((*json.Marshaler)(nil)).Elem()
	if fieldType.Implements(marshalerType) {
		var pType = "string"
		if prop.Type == nil {
			prop.Type = &pType
		}
		if prop.Format == "" {
			prop.Format = b.jsonSchemaFormat(fieldType.String())
		}
		return jsonName, modelDescription, prop
	}

	// check if annotation says it is a string
	if jsonTag := field.Tag.Get("json"); jsonTag != "" {
		s := strings.Split(jsonTag, ",")
		if len(s) > 1 && s[1] == "string" {
			stringt := "string"
			prop.Type = &stringt
			return jsonName, modelDescription, prop
		}
	}

	fieldKind := fieldType.Kind()
	switch {
	case fieldKind == reflect.Struct:
		jsonName, prop := b.buildStructTypeProperty(field, jsonName, model)
		return jsonName, modelDescription, prop
	case fieldKind == reflect.Slice || fieldKind == reflect.Array:
		jsonName, prop := b.buildArrayTypeProperty(field, jsonName, modelName)
		return jsonName, modelDescription, prop
	case fieldKind == reflect.Ptr:
		jsonName, prop := b.buildPointerTypeProperty(field, jsonName, modelName)
		return jsonName, modelDescription, prop
	case fieldKind == reflect.String:
		stringt := "string"
		prop.Type = &stringt
		return jsonName, modelDescription, prop
	case fieldKind == reflect.Map:
		// if it's a map, it's unstructured, and swagger 1.2 can't handle it
		objectType := "object"
		prop.Type = &objectType
		return jsonName, modelDescription, prop
	}

	if b.isPrimitiveType(fieldType.String()) {
		mapped := b.jsonSchemaType(fieldType.String())
		prop.Type = &mapped
		prop.Format = b.jsonSchemaFormat(fieldType.String())
		return jsonName, modelDescription, prop
	}
	modelType := fieldType.String()
	prop.Ref = &modelType

	if fieldType.Name() == "" { // override type of anonymous structs
		nestedTypeName := modelName + "." + jsonName
		prop.Ref = &nestedTypeName
		b.addModel(fieldType, nestedTypeName)
	}
	return jsonName, modelDescription, prop
}

func hasNamedJSONTag(field reflect.StructField) bool {
	parts := strings.Split(field.Tag.Get("json"), ",")
	if len(parts) == 0 {
		return false
	}
	for _, s := range parts[1:] {
		if s == "inline" {
			return false
		}
	}
	return len(parts[0]) > 0
}

func (b modelBuilder) buildStructTypeProperty(field reflect.StructField, jsonName string, model *Model) (nameJson string, prop ModelProperty) {
	prop.setPropertyMetadata(field)
	// Check for type override in tag
	if prop.Type != nil {
		return jsonName, prop
	}
	fieldType := field.Type
	// check for anonymous
	if len(fieldType.Name()) == 0 {
		// anonymous
		anonType := model.Id + "." + jsonName
		b.addModel(fieldType, anonType)
		prop.Ref = &anonType
		return jsonName, prop
	}

	if field.Name == fieldType.Name() && field.Anonymous && !hasNamedJSONTag(field) {
		// embedded struct
		sub := modelBuilder{new(ModelList), b.Config}
		sub.addModel(fieldType, "")
		subKey := sub.keyFrom(fieldType)
		// merge properties from sub
		subModel, _ := sub.Models.At(subKey)
		subModel.Properties.Do(func(k string, v ModelProperty) {
			model.Properties.Put(k, v)
			// if subModel says this property is required then include it
			required := false
			for _, each := range subModel.Required {
				if k == each {
					required = true
					break
				}
			}
			if required {
				model.Required = append(model.Required, k)
			}
		})
		// add all new referenced models
		sub.Models.Do(func(key string, sub Model) {
			if key != subKey {
				if _, ok := b.Models.At(key); !ok {
					b.Models.Put(key, sub)
				}
			}
		})
		// empty name signals skip property
		return "", prop
	}
	// simple struct
	b.addModel(fieldType, "")
	var pType = fieldType.String()
	prop.Ref = &pType
	return jsonName, prop
}

func (b modelBuilder) buildArrayTypeProperty(field reflect.StructField, jsonName, modelName string) (nameJson string, prop ModelProperty) {
	// check for type override in tags
	prop.setPropertyMetadata(field)
	if prop.Type != nil {
		return jsonName, prop
	}
	fieldType := field.Type
	var pType = "array"
	prop.Type = &pType
	isPrimitive := b.isPrimitiveType(fieldType.Elem().Name())
	elemTypeName := b.getElementTypeName(modelName, jsonName, fieldType.Elem())
	prop.Items = new(Item)
	if isPrimitive {
		mapped := b.jsonSchemaType(elemTypeName)
		prop.Items.Type = &mapped
	} else {
		prop.Items.Ref = &elemTypeName
	}
	// add|overwrite model for element type
	if fieldType.Elem().Kind() == reflect.Ptr {
		fieldType = fieldType.Elem()
	}
	if !isPrimitive {
		b.addModel(fieldType.Elem(), elemTypeName)
	}
	return jsonName, prop
}

func (b modelBuilder) buildPointerTypeProperty(field reflect.StructField, jsonName, modelName string) (nameJson string, prop ModelProperty) {
	prop.setPropertyMetadata(field)
	// Check for type override in tags
	if prop.Type != nil {
		return jsonName, prop
	}
	fieldType := field.Type

	// override type of pointer to list-likes
	if fieldType.Elem().Kind() == reflect.Slice || fieldType.Elem().Kind() == reflect.Array {
		var pType = "array"
		prop.Type = &pType
		isPrimitive := b.isPrimitiveType(fieldType.Elem().Elem().Name())
		elemName := b.getElementTypeName(modelName, jsonName, fieldType.Elem().Elem())
		if isPrimitive {
			primName := b.jsonSchemaType(elemName)
			prop.Items = &Item{Ref: &primName}
		} else {
			prop.Items = &Item{Ref: &elemName}
		}
		if !isPrimitive {
			// add|overwrite model for element type
			b.addModel(fieldType.Elem().Elem(), elemName)
		}
	} else {
		// non-array, pointer type
		var pType = b.jsonSchemaType(fieldType.String()[1:]) // no star, include pkg path
		if b.isPrimitiveType(fieldType.String()[1:]) {
			prop.Type = &pType
			prop.Format = b.jsonSchemaFormat(fieldType.String()[1:])
			return jsonName, prop
		}
		prop.Ref = &pType
		elemName := ""
		if fieldType.Elem().Name() == "" {
			elemName = modelName + "." + jsonName
			prop.Ref = &elemName
		}
		b.addModel(fieldType.Elem(), elemName)
	}
	return jsonName, prop
}

func (b modelBuilder) getElementTypeName(modelName, jsonName string, t reflect.Type) string {
	if t.Kind() == reflect.Ptr {
		return t.String()[1:]
	}
	if t.Name() == "" {
		return modelName + "." + jsonName
	}
	return b.keyFrom(t)
}

func (b modelBuilder) keyFrom(st reflect.Type) string {
	key := st.String()
	if len(st.Name()) == 0 { // unnamed type
		// Swagger UI has special meaning for [
		key = strings.Replace(key, "[]", "||", -1)
	}
	return key
}

// see also https://golang.org/ref/spec#Numeric_types
func (b modelBuilder) isPrimitiveType(modelName string) bool {
	if len(modelName) == 0 {
		return false
	}
	return strings.Contains("uint uint8 uint16 uint32 uint64 int int8 int16 int32 int64 float32 float64 bool string byte rune time.Time", modelName)
}

// jsonNameOfField returns the name of the field as it should appear in JSON format
// An empty string indicates that this field is not part of the JSON representation
func (b modelBuilder) jsonNameOfField(field reflect.StructField) string {
	if jsonTag := field.Tag.Get("json"); jsonTag != "" {
		s := strings.Split(jsonTag, ",")
		if s[0] == "-" {
			// empty name signals skip property
			return ""
		} else if s[0] != "" {
			return s[0]
		}
	}
	return field.Name
}

// see also http://json-schema.org/latest/json-schema-core.html#anchor8
func (b modelBuilder) jsonSchemaType(modelName string) string {
	schemaMap := map[string]string{
		"uint":   "integer",
		"uint8":  "integer",
		"uint16": "integer",
		"uint32": "integer",
		"uint64": "integer",

		"int":   "integer",
		"int8":  "integer",
		"int16": "integer",
		"int32": "integer",
		"int64": "integer",

		"byte":      "integer",
		"float64":   "number",
		"float32":   "number",
		"bool":      "boolean",
		"time.Time": "string",
	}
	mapped, ok := schemaMap[modelName]
	if !ok {
		return modelName // use as is (custom or struct)
	}
	return mapped
}

func (b modelBuilder) jsonSchemaFormat(modelName string) string {
	if b.Config != nil && b.Config.SchemaFormatHandler != nil {
		if mapped := b.Config.SchemaFormatHandler(modelName); mapped != "" {
			return mapped
		}
	}
	schemaMap := map[string]string{
		"int":        "int32",
		"int32":      "int32",
		"int64":      "int64",
		"byte":       "byte",
		"uint":       "integer",
		"uint8":      "byte",
		"float64":    "double",
		"float32":    "float",
		"time.Time":  "date-time",
		"*time.Time": "date-time",
	}
	mapped, ok := schemaMap[modelName]
	if !ok {
		return "" // no format
	}
	return mapped
}
