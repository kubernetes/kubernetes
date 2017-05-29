package client

import (
	"reflect"
	"strings"
)

type Schemas struct {
	Collection
	Data          []Schema `json:"data,omitempty"`
	schemasByName map[string]*Schema
}

func (s *Schema) CheckField(name string) (Field, bool) {
	for fieldName := range s.ResourceFields {
		if fieldName == name {
			v, ok := s.ResourceFields[fieldName]
			return v, ok
		}
	}
	return Field{}, false
}

func (s *Schema) Field(name string) Field {
	f, _ := s.CheckField(name)
	return f
}

func (s *Schemas) CheckSchema(name string) (Schema, bool) {
	for i := range s.Data {
		if s.Data[i].Id == name {
			return s.Data[i], true
		}
	}
	return Schema{}, false
}

func (s *Schemas) Schema(name string) Schema {
	r, _ := s.CheckSchema(name)
	return r
}

func typeToFields(t reflect.Type) map[string]Field {
	result := map[string]Field{}

	for i := 0; i < t.NumField(); i++ {
		schemaField := Field{}

		typeField := t.Field(i)
		if typeField.Anonymous && typeField.Type.Kind() == reflect.Struct {
			parentFields := typeToFields(typeField.Type)
			for k, v := range result {
				parentFields[k] = v
			}
			result = parentFields
			continue
		} else if typeField.Anonymous {
			continue
		}

		fieldString := strings.ToLower(typeField.Type.Kind().String())

		switch {
		case strings.HasPrefix(fieldString, "int") || strings.HasPrefix(fieldString, "uint"):
			schemaField.Type = "int"
		case fieldString == "bool":
			schemaField.Type = fieldString
		case fieldString == "float32" || fieldString == "float64":
			schemaField.Type = "float"
		case fieldString == "string":
			schemaField.Type = "string"
		case fieldString == "map":
			// HACK
			schemaField.Type = "map[string]"
		case fieldString == "slice":
			// HACK
			schemaField.Type = "array[string]"
		}

		name := strings.Split(typeField.Tag.Get("json"), ",")[0]
		if name == "" && len(typeField.Name) > 1 {
			name = strings.ToLower(typeField.Name[0:1]) + typeField.Name[1:]
		} else if name == "" {
			name = typeField.Name
		}

		if schemaField.Type != "" {
			result[name] = schemaField
		}
	}

	return result
}

func (s *Schemas) AddType(schemaName string, obj interface{}) *Schema {
	t := reflect.TypeOf(obj)
	schema := Schema{
		Resource: Resource{
			Id:    schemaName,
			Type:  "schema",
			Links: map[string]string{},
		},
		PluralName:        guessPluralName(schemaName),
		ResourceFields:    typeToFields(t),
		CollectionMethods: []string{"GET"},
		ResourceMethods:   []string{"GET"},
	}

	if s.Data == nil {
		s.Data = []Schema{}
	}

	s.Data = append(s.Data, schema)

	return &s.Data[len(s.Data)-1]
}

func guessPluralName(name string) string {
	if name == "" {
		return ""
	}

	if strings.HasSuffix(name, "s") ||
		strings.HasSuffix(name, "ch") ||
		strings.HasSuffix(name, "x") {
		return name + "es"
	}
	return name + "s"
}
