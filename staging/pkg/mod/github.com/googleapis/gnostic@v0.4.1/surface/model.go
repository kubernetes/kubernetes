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

package surface_v1

import (
	"log"
	nethttp "net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
)

// The structure to transport information during the recursive calls inside model_openapiv2.go
// and model_openapiv3.go
type FieldInfo struct {
	fieldKind   FieldKind
	fieldType   string
	fieldFormat string
	// For parameters
	fieldPosition Position
	fieldName     string
}

func (m *Model) addType(t *Type) {
	m.Types = append(m.Types, t)
}

func (m *Model) addMethod(method *Method) {
	m.Methods = append(m.Methods, method)
}

func (m *Model) TypeWithTypeName(name string) *Type {
	if name == "" {
		return nil
	}
	for _, t := range m.Types {
		if t.TypeName == name {
			return t
		}
	}
	return nil
}

func generateOperationName(method, path string) string {
	filteredPath := strings.Replace(path, "/", "_", -1)
	filteredPath = strings.Replace(filteredPath, ".", "_", -1)
	filteredPath = strings.Replace(filteredPath, "{", "", -1)
	filteredPath = strings.Replace(filteredPath, "}", "", -1)
	return strings.Title(method) + filteredPath
}

func sanitizeOperationName(name string) string {
	name = strings.Title(name)
	name = strings.Replace(name, ".", "_", -1)
	return name
}

func typeForRef(ref string) (typeName string) {
	return path.Base(ref)
}

// Helper method to build a surface model Type
func makeType(name string) *Type {
	t := &Type{
		Name:   name,
		Kind:   TypeKind_STRUCT,
		Fields: make([]*Field, 0),
	}
	return t
}

// Helper method to build a surface model Field
func makeFieldAndAppendToType(info *FieldInfo, schemaType *Type, fieldName string) {
	if info != nil {
		f := &Field{Name: info.fieldName}
		if fieldName != "" {
			f.Name = fieldName
		}
		f.Type, f.Kind, f.Format, f.Position = info.fieldType, info.fieldKind, info.fieldFormat, info.fieldPosition
		schemaType.Fields = append(schemaType.Fields, f)
	}
}

// Helper method to determine the type of the value property for a map.
func determineMapValueType(fInfo FieldInfo) (mapValueType string) {
	if fInfo.fieldKind == FieldKind_ARRAY {
		mapValueType = "[]"
	}
	if fInfo.fieldFormat != "" {
		fInfo.fieldType = fInfo.fieldFormat
	}
	mapValueType += fInfo.fieldType
	return mapValueType
}

// Converts a string status code like: "504" into the corresponding text ("Gateway_Timeout")
func convertStatusCodeToText(c string) (statusText string) {
	code, err := strconv.Atoi(c)
	if err == nil {
		statusText = nethttp.StatusText(code)
		if statusText == "" {
			log.Println("Status code " + c + " is not known to net.http.StatusText. This might cause unpredictable behavior.")
			statusText = "unknownStatusCode"
		}
		statusText = strings.Replace(statusText, " ", "_", -1)
	}
	return statusText
}

// Searches all created types so far and returns the Type where 'typeName' matches.
func findType(types []*Type, typeName string) *Type {
	for _, t := range types {
		if typeName == t.Name {
			return t
		}
	}
	return nil
}

// Returns true if s is a valid URL.
func isSymbolicReference(s string) bool {
	_, err := url.ParseRequestURI(s)
	if err != nil {
		return false
	}
	return true
}

// Replace encoded URLS with actual characters
func validTypeForRef(XRef string) string {
	t, _ := url.QueryUnescape(typeForRef(XRef))
	return t
}
