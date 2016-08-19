// Copyright 2015 go-swagger maintainers
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

package validate

import (
	"reflect"
	"strings"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/runtime"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/swag"
)

type typeValidator struct {
	Type   spec.StringOrArray
	Format string
	In     string
	Path   string
}

var jsonTypeNames = map[string]struct{}{
	"array":   struct{}{},
	"boolean": struct{}{},
	"integer": struct{}{},
	"null":    struct{}{},
	"number":  struct{}{},
	"object":  struct{}{},
	"string":  struct{}{},
}

func (t *typeValidator) schemaInfoForType(data interface{}) (string, string) {
	switch data.(type) {
	case []byte:
		return "string", "byte"
	case strfmt.Date, *strfmt.Date:
		return "string", "date"
	case strfmt.DateTime, *strfmt.DateTime:
		return "string", "datetime"
	case runtime.File, *runtime.File:
		return "file", ""
	case strfmt.URI, *strfmt.URI:
		return "string", "uri"
	case strfmt.Email, *strfmt.Email:
		return "string", "email"
	case strfmt.Hostname, *strfmt.Hostname:
		return "string", "hostname"
	case strfmt.IPv4, *strfmt.IPv4:
		return "string", "ipv4"
	case strfmt.IPv6, *strfmt.IPv6:
		return "string", "ipv6"
	case strfmt.UUID, *strfmt.UUID:
		return "string", "uuid"
	case strfmt.UUID3, *strfmt.UUID3:
		return "string", "uuid3"
	case strfmt.UUID4, *strfmt.UUID4:
		return "string", "uuid4"
	case strfmt.UUID5, *strfmt.UUID5:
		return "string", "uuid5"
	case strfmt.ISBN, *strfmt.ISBN:
		return "string", "isbn"
	case strfmt.ISBN10, *strfmt.ISBN10:
		return "string", "isbn10"
	case strfmt.ISBN13, *strfmt.ISBN13:
		return "string", "isbn13"
	case strfmt.CreditCard, *strfmt.CreditCard:
		return "string", "creditcard"
	case strfmt.SSN, *strfmt.SSN:
		return "string", "ssn"
	case strfmt.HexColor, *strfmt.HexColor:
		return "string", "hexcolor"
	case strfmt.RGBColor, *strfmt.RGBColor:
		return "string", "rgbcolor"
	default:
		val := reflect.ValueOf(data)
		tpe := val.Type()
		switch tpe.Kind() {
		case reflect.Bool:
			return "boolean", ""
		case reflect.String:
			return "string", ""
		case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Uint8, reflect.Uint16, reflect.Uint32:
			return "integer", "int32"
		case reflect.Int, reflect.Int64, reflect.Uint, reflect.Uint64:
			return "integer", "int64"
		case reflect.Float32:
			return "number", "float32"
		case reflect.Float64:
			return "number", "float64"
		case reflect.Slice:
			return "array", ""
		case reflect.Map, reflect.Struct:
			return "object", ""
		case reflect.Interface:
			// What to do here?
			panic("dunno what to do here")
		case reflect.Ptr:
			return t.schemaInfoForType(reflect.Indirect(val).Interface())
		}
	}
	return "", ""
}

func (t *typeValidator) SetPath(path string) {
	t.Path = path
}

func (t *typeValidator) Applies(source interface{}, kind reflect.Kind) bool {
	stpe := reflect.TypeOf(source)
	r := (len(t.Type) > 0 || t.Format != "") && (stpe == specSchemaType || stpe == specParameterType || stpe == specHeaderType)
	//fmt.Printf("type validator for %q applies %t for %T (kind: %v)\n", t.Path, r, source, kind)
	return r
}

func (t *typeValidator) Validate(data interface{}) *Result {
	result := new(Result)
	result.Inc()
	if data == nil || reflect.DeepEqual(reflect.Zero(reflect.TypeOf(data)), reflect.ValueOf(data)) {
		if len(t.Type) > 0 && !t.Type.Contains("null") { // TODO: if a property is not required it also passes this
			return sErr(errors.InvalidType(t.Path, t.In, strings.Join(t.Type, ","), "null"))
		}
		return result
	}

	// check if the type matches, should be used in every validator chain as first item
	val := reflect.Indirect(reflect.ValueOf(data))
	kind := val.Kind()

	schType, format := t.schemaInfoForType(data)
	//fmt.Println("path:", t.Path, "schType:", schType, "format:", format, "expType:", t.Type, "expFmt:", t.Format, "kind:", val.Kind().String())
	isLowerInt := t.Format == "int64" && format == "int32"
	isLowerFloat := t.Format == "float64" && format == "float32"
	isFloatInt := schType == "number" && swag.IsFloat64AJSONInteger(val.Float()) && t.Type.Contains("integer")
	isIntFloat := schType == "integer" && t.Type.Contains("number")

	if kind != reflect.String && kind != reflect.Slice && t.Format != "" && !(t.Type.Contains(schType) || format == t.Format || isFloatInt || isIntFloat || isLowerInt || isLowerFloat) {
		return sErr(errors.InvalidType(t.Path, t.In, t.Format, format))
	}
	if !(t.Type.Contains("number") || t.Type.Contains("integer")) && t.Format != "" && (kind == reflect.String || kind == reflect.Slice) {
		return result
	}

	if !(t.Type.Contains(schType) || isFloatInt || isIntFloat) {
		return sErr(errors.InvalidType(t.Path, t.In, strings.Join(t.Type, ","), schType))
	}
	return result
}
