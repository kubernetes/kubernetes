// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package bigquery

import (
	"errors"
	"reflect"

	bq "google.golang.org/api/bigquery/v2"
)

// Schema describes the fields in a table or query result.
type Schema []*FieldSchema

type FieldSchema struct {
	// The field name.
	// Must contain only letters (a-z, A-Z), numbers (0-9), or underscores (_),
	// and must start with a letter or underscore.
	// The maximum length is 128 characters.
	Name string

	// A description of the field. The maximum length is 16,384 characters.
	Description string

	// Whether the field may contain multiple values.
	Repeated bool
	// Whether the field is required.  Ignored if Repeated is true.
	Required bool

	// The field data type.  If Type is Record, then this field contains a nested schema,
	// which is described by Schema.
	Type FieldType
	// Describes the nested schema if Type is set to Record.
	Schema Schema
}

func (fs *FieldSchema) asTableFieldSchema() *bq.TableFieldSchema {
	tfs := &bq.TableFieldSchema{
		Description: fs.Description,
		Name:        fs.Name,
		Type:        string(fs.Type),
	}

	if fs.Repeated {
		tfs.Mode = "REPEATED"
	} else if fs.Required {
		tfs.Mode = "REQUIRED"
	} // else leave as default, which is interpreted as NULLABLE.

	for _, f := range fs.Schema {
		tfs.Fields = append(tfs.Fields, f.asTableFieldSchema())
	}

	return tfs
}

func (s Schema) asTableSchema() *bq.TableSchema {
	var fields []*bq.TableFieldSchema
	for _, f := range s {
		fields = append(fields, f.asTableFieldSchema())
	}
	return &bq.TableSchema{Fields: fields}
}

// customizeCreateTable allows a Schema to be used directly as an option to CreateTable.
func (s Schema) customizeCreateTable(conf *createTableConf) {
	conf.schema = s.asTableSchema()
}

func convertTableFieldSchema(tfs *bq.TableFieldSchema) *FieldSchema {
	fs := &FieldSchema{
		Description: tfs.Description,
		Name:        tfs.Name,
		Repeated:    tfs.Mode == "REPEATED",
		Required:    tfs.Mode == "REQUIRED",
		Type:        FieldType(tfs.Type),
	}

	for _, f := range tfs.Fields {
		fs.Schema = append(fs.Schema, convertTableFieldSchema(f))
	}
	return fs
}

func convertTableSchema(ts *bq.TableSchema) Schema {
	var s Schema
	for _, f := range ts.Fields {
		s = append(s, convertTableFieldSchema(f))
	}
	return s
}

type FieldType string

const (
	StringFieldType    FieldType = "STRING"
	IntegerFieldType   FieldType = "INTEGER"
	FloatFieldType     FieldType = "FLOAT"
	BooleanFieldType   FieldType = "BOOLEAN"
	TimestampFieldType FieldType = "TIMESTAMP"
	RecordFieldType    FieldType = "RECORD"
)

var errNoStruct = errors.New("bigquery: can only infer schema from struct or pointer to struct")
var errUnsupportedFieldType = errors.New("bigquery: unsupported type of field in struct")

// InferSchema tries to derive a BigQuery schema from the supplied struct value.
// NOTE: All fields in the returned Schema are configured to be required,
// unless the corresponding field in the supplied struct is a slice or array.
// It is considered an error if the struct (including nested structs) contains
// any exported fields that are pointers or one of the following types:
// map, interface, complex64, complex128, func, chan.
// In these cases, an error will be returned.
// Future versions may handle these cases without error.
func InferSchema(st interface{}) (Schema, error) {
	return inferStruct(reflect.TypeOf(st))
}

func inferStruct(rt reflect.Type) (Schema, error) {
	switch rt.Kind() {
	case reflect.Struct:
		return inferFields(rt)
	default:
		return nil, errNoStruct
	}

}

// inferFieldSchema infers the FieldSchema for a Go type
func inferFieldSchema(rt reflect.Type) (*FieldSchema, error) {
	switch {
	case isByteSlice(rt):
		return &FieldSchema{Required: true, Type: StringFieldType}, nil
	case isTimeTime(rt):
		return &FieldSchema{Required: true, Type: TimestampFieldType}, nil
	case isRepeated(rt):
		et := rt.Elem()

		if isRepeated(et) && !isByteSlice(et) {
			// Multi dimensional slices/arrays are not supported by BigQuery
			return nil, errUnsupportedFieldType
		}

		f, err := inferFieldSchema(et)
		if err != nil {
			return nil, err
		}
		f.Repeated = true
		f.Required = false
		return f, nil
	case isStruct(rt):
		nested, err := inferFields(rt)
		if err != nil {
			return nil, err
		}
		return &FieldSchema{Required: true, Type: RecordFieldType, Schema: nested}, nil
	}

	switch rt.Kind() {
	case reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Int,
		reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uint, reflect.Uintptr:
		return &FieldSchema{Required: true, Type: IntegerFieldType}, nil
	case reflect.String:
		return &FieldSchema{Required: true, Type: StringFieldType}, nil
	case reflect.Bool:
		return &FieldSchema{Required: true, Type: BooleanFieldType}, nil
	case reflect.Float32, reflect.Float64:
		return &FieldSchema{Required: true, Type: FloatFieldType}, nil
	default:
		return nil, errUnsupportedFieldType
	}
}

// inferFields extracts all exported field types from struct type.
func inferFields(rt reflect.Type) (Schema, error) {
	var s Schema

	for i := 0; i < rt.NumField(); i++ {
		field := rt.Field(i)
		if field.PkgPath != "" {
			// field is unexported.
			continue
		}

		if field.Anonymous {
			// TODO(nightlyone) support embedded (see https://github.com/GoogleCloudPlatform/google-cloud-go/issues/238)
			return nil, errUnsupportedFieldType
		}

		f, err := inferFieldSchema(field.Type)
		if err != nil {
			return nil, err
		}
		f.Name = field.Name

		s = append(s, f)
	}

	return s, nil
}

func isByteSlice(rt reflect.Type) bool {
	return rt.Kind() == reflect.Slice && rt.Elem().Kind() == reflect.Uint8
}

func isTimeTime(rt reflect.Type) bool {
	return rt.PkgPath() == "time" && rt.Name() == "Time"
}

func isStruct(rt reflect.Type) bool {
	return rt.Kind() == reflect.Struct
}

func isRepeated(rt reflect.Type) bool {
	switch rt.Kind() {
	case reflect.Slice, reflect.Array:
		return true
	default:
		return false
	}
}
