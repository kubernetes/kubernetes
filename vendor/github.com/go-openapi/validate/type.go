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
	Type     spec.StringOrArray
	Nullable bool
	Format   string
	In       string
	Path     string
}

func (t *typeValidator) schemaInfoForType(data interface{}) (string, string) {
	// internal type to JSON type with swagger 2.0 format (with go-openapi/strfmt extensions),
	// see https://github.com/go-openapi/strfmt/blob/master/README.md
	// TODO: this switch really is some sort of reverse lookup for formats. It should be provided by strfmt.
	switch data.(type) {
	case []byte, strfmt.Base64, *strfmt.Base64:
		return "string", "byte"
	case strfmt.CreditCard, *strfmt.CreditCard:
		return "string", "creditcard"
	case strfmt.Date, *strfmt.Date:
		return "string", "date"
	case strfmt.DateTime, *strfmt.DateTime:
		return "string", "date-time"
	case strfmt.Duration, *strfmt.Duration:
		return "string", "duration"
	case runtime.File, *runtime.File:
		return "file", ""
	case strfmt.Email, *strfmt.Email:
		return "string", "email"
	case strfmt.HexColor, *strfmt.HexColor:
		return "string", "hexcolor"
	case strfmt.Hostname, *strfmt.Hostname:
		return "string", "hostname"
	case strfmt.IPv4, *strfmt.IPv4:
		return "string", "ipv4"
	case strfmt.IPv6, *strfmt.IPv6:
		return "string", "ipv6"
	case strfmt.ISBN, *strfmt.ISBN:
		return "string", "isbn"
	case strfmt.ISBN10, *strfmt.ISBN10:
		return "string", "isbn10"
	case strfmt.ISBN13, *strfmt.ISBN13:
		return "string", "isbn13"
	case strfmt.MAC, *strfmt.MAC:
		return "string", "mac"
	case strfmt.ObjectId, *strfmt.ObjectId:
		return "string", "bsonobjectid"
	case strfmt.Password, *strfmt.Password:
		return "string", "password"
	case strfmt.RGBColor, *strfmt.RGBColor:
		return "string", "rgbcolor"
	case strfmt.SSN, *strfmt.SSN:
		return "string", "ssn"
	case strfmt.URI, *strfmt.URI:
		return "string", "uri"
	case strfmt.UUID, *strfmt.UUID:
		return "string", "uuid"
	case strfmt.UUID3, *strfmt.UUID3:
		return "string", "uuid3"
	case strfmt.UUID4, *strfmt.UUID4:
		return "string", "uuid4"
	case strfmt.UUID5, *strfmt.UUID5:
		return "string", "uuid5"
	// TODO: missing binary (io.ReadCloser)
	// TODO: missing json.Number
	default:
		val := reflect.ValueOf(data)
		tpe := val.Type()
		switch tpe.Kind() {
		case reflect.Bool:
			return "boolean", ""
		case reflect.String:
			return "string", ""
		case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Uint8, reflect.Uint16, reflect.Uint32:
			// NOTE: that is the spec. With go-openapi, is that not uint32 for unsigned integers?
			return "integer", "int32"
		case reflect.Int, reflect.Int64, reflect.Uint, reflect.Uint64:
			return "integer", "int64"
		case reflect.Float32:
			// NOTE: is that not "float"?
			return "number", "float32"
		case reflect.Float64:
			// NOTE: is that not "double"?
			return "number", "float64"
		// NOTE: go arrays (reflect.Array) are not supported (fixed length)
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
	// typeValidator applies to Schema, Parameter and Header objects
	stpe := reflect.TypeOf(source)
	r := (len(t.Type) > 0 || t.Format != "") && (stpe == specSchemaType || stpe == specParameterType || stpe == specHeaderType)
	debugLog("type validator for %q applies %t for %T (kind: %v)\n", t.Path, r, source, kind)
	return r
}

func (t *typeValidator) Validate(data interface{}) *Result {
	result := new(Result)
	result.Inc()
	if data == nil || reflect.DeepEqual(reflect.Zero(reflect.TypeOf(data)), reflect.ValueOf(data)) {
		// nil or zero value for the passed structure require Type: null
		if len(t.Type) > 0 && !t.Type.Contains("null") && !t.Nullable { // TODO: if a property is not required it also passes this
			return errorHelp.sErr(errors.InvalidType(t.Path, t.In, strings.Join(t.Type, ","), "null"))
		}
		return result
	}

	// check if the type matches, should be used in every validator chain as first item
	val := reflect.Indirect(reflect.ValueOf(data))
	kind := val.Kind()

	// infer schema type (JSON) and format from passed data type
	schType, format := t.schemaInfoForType(data)

	debugLog("path: %s, schType: %s,  format: %s, expType: %s, expFmt: %s, kind: %s", t.Path, schType, format, t.Type, t.Format, val.Kind().String())

	// check numerical types
	// TODO: check unsigned ints
	// TODO: check json.Number (see schema.go)
	isLowerInt := t.Format == "int64" && format == "int32"
	isLowerFloat := t.Format == "float64" && format == "float32"
	isFloatInt := schType == "number" && swag.IsFloat64AJSONInteger(val.Float()) && t.Type.Contains("integer")
	isIntFloat := schType == "integer" && t.Type.Contains("number")

	if kind != reflect.String && kind != reflect.Slice && t.Format != "" && !(t.Type.Contains(schType) || format == t.Format || isFloatInt || isIntFloat || isLowerInt || isLowerFloat) {
		// TODO: test case
		return errorHelp.sErr(errors.InvalidType(t.Path, t.In, t.Format, format))
	}

	if !(t.Type.Contains("number") || t.Type.Contains("integer")) && t.Format != "" && (kind == reflect.String || kind == reflect.Slice) {
		return result
	}

	if !(t.Type.Contains(schType) || isFloatInt || isIntFloat) {
		return errorHelp.sErr(errors.InvalidType(t.Path, t.In, strings.Join(t.Type, ","), schType))
	}
	return result
}
