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

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/validation/errors"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
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
		return stringType, stringFormatByte
	case strfmt.CreditCard, *strfmt.CreditCard:
		return stringType, stringFormatCreditCard
	case strfmt.Date, *strfmt.Date:
		return stringType, stringFormatDate
	case strfmt.DateTime, *strfmt.DateTime:
		return stringType, stringFormatDateTime
	case strfmt.Duration, *strfmt.Duration:
		return stringType, stringFormatDuration
	case strfmt.Email, *strfmt.Email:
		return stringType, stringFormatEmail
	case strfmt.HexColor, *strfmt.HexColor:
		return stringType, stringFormatHexColor
	case strfmt.Hostname, *strfmt.Hostname:
		return stringType, stringFormatHostname
	case strfmt.IPv4, *strfmt.IPv4:
		return stringType, stringFormatIPv4
	case strfmt.IPv6, *strfmt.IPv6:
		return stringType, stringFormatIPv6
	case strfmt.ISBN, *strfmt.ISBN:
		return stringType, stringFormatISBN
	case strfmt.ISBN10, *strfmt.ISBN10:
		return stringType, stringFormatISBN10
	case strfmt.ISBN13, *strfmt.ISBN13:
		return stringType, stringFormatISBN13
	case strfmt.MAC, *strfmt.MAC:
		return stringType, stringFormatMAC
	case strfmt.Password, *strfmt.Password:
		return stringType, stringFormatPassword
	case strfmt.RGBColor, *strfmt.RGBColor:
		return stringType, stringFormatRGBColor
	case strfmt.SSN, *strfmt.SSN:
		return stringType, stringFormatSSN
	case strfmt.URI, *strfmt.URI:
		return stringType, stringFormatURI
	case strfmt.UUID, *strfmt.UUID:
		return stringType, stringFormatUUID
	case strfmt.UUID3, *strfmt.UUID3:
		return stringType, stringFormatUUID3
	case strfmt.UUID4, *strfmt.UUID4:
		return stringType, stringFormatUUID4
	case strfmt.UUID5, *strfmt.UUID5:
		return stringType, stringFormatUUID5
	// TODO: missing binary (io.ReadCloser)
	// TODO: missing json.Number
	default:
		val := reflect.ValueOf(data)
		tpe := val.Type()
		switch tpe.Kind() {
		case reflect.Bool:
			return booleanType, ""
		case reflect.String:
			return stringType, ""
		case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Uint8, reflect.Uint16, reflect.Uint32:
			// NOTE: that is the spec. With go-openapi, is that not uint32 for unsigned integers?
			return integerType, integerFormatInt32
		case reflect.Int, reflect.Int64, reflect.Uint, reflect.Uint64:
			return integerType, integerFormatInt64
		case reflect.Float32:
			// NOTE: is that not numberFormatFloat?
			return numberType, numberFormatFloat32
		case reflect.Float64:
			// NOTE: is that not "double"?
			return numberType, numberFormatFloat64
		// NOTE: go arrays (reflect.Array) are not supported (fixed length)
		case reflect.Slice:
			return arrayType, ""
		case reflect.Map, reflect.Struct:
			return objectType, ""
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
	r := (len(t.Type) > 0 || t.Format != "") && stpe == specSchemaType
	debugLog("type validator for %q applies %t for %T (kind: %v)\n", t.Path, r, source, kind)
	return r
}

func (t *typeValidator) Validate(data interface{}) *Result {
	result := new(Result)
	result.Inc()
	if data == nil || reflect.DeepEqual(reflect.Zero(reflect.TypeOf(data)), reflect.ValueOf(data)) {
		// nil or zero value for the passed structure require Type: null
		if len(t.Type) > 0 && !t.Type.Contains(nullType) && !t.Nullable { // TODO: if a property is not required it also passes this
			return errorHelp.sErr(errors.InvalidType(t.Path, t.In, strings.Join(t.Type, ","), nullType))
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
	isLowerInt := t.Format == integerFormatInt64 && format == integerFormatInt32
	isLowerFloat := t.Format == numberFormatFloat64 && format == numberFormatFloat32
	isFloatInt := schType == numberType && swag.IsFloat64AJSONInteger(val.Float()) && t.Type.Contains(integerType)
	isIntFloat := schType == integerType && t.Type.Contains(numberType)

	if kind != reflect.String && kind != reflect.Slice && t.Format != "" && !(t.Type.Contains(schType) || format == t.Format || isFloatInt || isIntFloat || isLowerInt || isLowerFloat) {
		// TODO: test case
		return errorHelp.sErr(errors.InvalidType(t.Path, t.In, t.Format, format))
	}

	if !(t.Type.Contains(numberType) || t.Type.Contains(integerType)) && t.Format != "" && (kind == reflect.String || kind == reflect.Slice) {
		return result
	}

	if !(t.Type.Contains(schType) || isFloatInt || isIntFloat) {
		return errorHelp.sErr(errors.InvalidType(t.Path, t.In, strings.Join(t.Type, ","), schType))
	}
	return result
}
