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

// TODO: define this as package validate/internal
// This must be done while keeping CI intact with all tests and test coverage

import (
	"reflect"

	"k8s.io/kube-openapi/pkg/validation/errors"
)

const (
	swaggerExample  = "example"
	swaggerExamples = "examples"
)

const (
	objectType  = "object"
	arrayType   = "array"
	stringType  = "string"
	integerType = "integer"
	numberType  = "number"
	booleanType = "boolean"
	nullType    = "null"
)

const (
	jsonProperties = "properties"
	jsonDefault    = "default"
)

const (
	stringFormatDate       = "date"
	stringFormatDateTime   = "date-time"
	stringFormatPassword   = "password"
	stringFormatByte       = "byte"
	stringFormatCreditCard = "creditcard"
	stringFormatDuration   = "duration"
	stringFormatEmail      = "email"
	stringFormatHexColor   = "hexcolor"
	stringFormatHostname   = "hostname"
	stringFormatIPv4       = "ipv4"
	stringFormatIPv6       = "ipv6"
	stringFormatISBN       = "isbn"
	stringFormatISBN10     = "isbn10"
	stringFormatISBN13     = "isbn13"
	stringFormatMAC        = "mac"
	stringFormatRGBColor   = "rgbcolor"
	stringFormatSSN        = "ssn"
	stringFormatURI        = "uri"
	stringFormatUUID       = "uuid"
	stringFormatUUID3      = "uuid3"
	stringFormatUUID4      = "uuid4"
	stringFormatUUID5      = "uuid5"

	integerFormatInt32  = "int32"
	integerFormatInt64  = "int64"
	integerFormatUInt32 = "uint32"
	integerFormatUInt64 = "uint64"

	numberFormatFloat32 = "float32"
	numberFormatFloat64 = "float64"
	numberFormatFloat   = "float"
	numberFormatDouble  = "double"
)

// Helpers available at the package level
var (
	valueHelp *valueHelper
	errorHelp *errorHelper
)

type errorHelper struct {
	// A collection of unexported helpers for error construction
}

func (h *errorHelper) sErr(err errors.Error) *Result {
	// Builds a Result from standard errors.Error
	return &Result{Errors: []error{err}}
}

type valueHelper struct {
	// A collection of unexported helpers for value validation
}

func (h *valueHelper) asInt64(val interface{}) int64 {
	// Number conversion function for int64, without error checking
	// (implements an implicit type upgrade).
	v := reflect.ValueOf(val)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return v.Int()
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return int64(v.Uint())
	case reflect.Float32, reflect.Float64:
		return int64(v.Float())
	default:
		//panic("Non numeric value in asInt64()")
		return 0
	}
}

func (h *valueHelper) asUint64(val interface{}) uint64 {
	// Number conversion function for uint64, without error checking
	// (implements an implicit type upgrade).
	v := reflect.ValueOf(val)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return uint64(v.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return v.Uint()
	case reflect.Float32, reflect.Float64:
		return uint64(v.Float())
	default:
		//panic("Non numeric value in asUint64()")
		return 0
	}
}

// Same for unsigned floats
func (h *valueHelper) asFloat64(val interface{}) float64 {
	// Number conversion function for float64, without error checking
	// (implements an implicit type upgrade).
	v := reflect.ValueOf(val)
	switch v.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(v.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(v.Uint())
	case reflect.Float32, reflect.Float64:
		return v.Float()
	default:
		//panic("Non numeric value in asFloat64()")
		return 0
	}
}
