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
	"fmt"
	"reflect"
	"testing"
	"time"

	bq "google.golang.org/api/bigquery/v2"
)

func (fs *FieldSchema) GoString() string {
	if fs == nil {
		return "<nil>"
	}

	return fmt.Sprintf("{Name:%s Description:%s Repeated:%t Required:%t Type:%s Schema:%s}",
		fs.Name,
		fs.Description,
		fs.Repeated,
		fs.Required,
		fs.Type,
		fmt.Sprintf("%#v", fs.Schema),
	)
}

func bqTableFieldSchema(desc, name, typ, mode string) *bq.TableFieldSchema {
	return &bq.TableFieldSchema{
		Description: desc,
		Name:        name,
		Mode:        mode,
		Type:        typ,
	}
}

func fieldSchema(desc, name, typ string, repeated, required bool) *FieldSchema {
	return &FieldSchema{
		Description: desc,
		Name:        name,
		Repeated:    repeated,
		Required:    required,
		Type:        FieldType(typ),
	}
}

func TestSchemaConversion(t *testing.T) {
	testCases := []struct {
		schema   Schema
		bqSchema *bq.TableSchema
	}{
		{
			// required
			bqSchema: &bq.TableSchema{
				Fields: []*bq.TableFieldSchema{
					bqTableFieldSchema("desc", "name", "STRING", "REQUIRED"),
				},
			},
			schema: Schema{
				fieldSchema("desc", "name", "STRING", false, true),
			},
		},
		{
			// repeated
			bqSchema: &bq.TableSchema{
				Fields: []*bq.TableFieldSchema{
					bqTableFieldSchema("desc", "name", "STRING", "REPEATED"),
				},
			},
			schema: Schema{
				fieldSchema("desc", "name", "STRING", true, false),
			},
		},
		{
			// nullable, string
			bqSchema: &bq.TableSchema{
				Fields: []*bq.TableFieldSchema{
					bqTableFieldSchema("desc", "name", "STRING", ""),
				},
			},
			schema: Schema{
				fieldSchema("desc", "name", "STRING", false, false),
			},
		},
		{
			// integer
			bqSchema: &bq.TableSchema{
				Fields: []*bq.TableFieldSchema{
					bqTableFieldSchema("desc", "name", "INTEGER", ""),
				},
			},
			schema: Schema{
				fieldSchema("desc", "name", "INTEGER", false, false),
			},
		},
		{
			// float
			bqSchema: &bq.TableSchema{
				Fields: []*bq.TableFieldSchema{
					bqTableFieldSchema("desc", "name", "FLOAT", ""),
				},
			},
			schema: Schema{
				fieldSchema("desc", "name", "FLOAT", false, false),
			},
		},
		{
			// boolean
			bqSchema: &bq.TableSchema{
				Fields: []*bq.TableFieldSchema{
					bqTableFieldSchema("desc", "name", "BOOLEAN", ""),
				},
			},
			schema: Schema{
				fieldSchema("desc", "name", "BOOLEAN", false, false),
			},
		},
		{
			// timestamp
			bqSchema: &bq.TableSchema{
				Fields: []*bq.TableFieldSchema{
					bqTableFieldSchema("desc", "name", "TIMESTAMP", ""),
				},
			},
			schema: Schema{
				fieldSchema("desc", "name", "TIMESTAMP", false, false),
			},
		},
		{
			// nested
			bqSchema: &bq.TableSchema{
				Fields: []*bq.TableFieldSchema{
					{
						Description: "An outer schema wrapping a nested schema",
						Name:        "outer",
						Mode:        "REQUIRED",
						Type:        "RECORD",
						Fields: []*bq.TableFieldSchema{
							bqTableFieldSchema("inner field", "inner", "STRING", ""),
						},
					},
				},
			},
			schema: Schema{
				&FieldSchema{
					Description: "An outer schema wrapping a nested schema",
					Name:        "outer",
					Required:    true,
					Type:        "RECORD",
					Schema: []*FieldSchema{
						{
							Description: "inner field",
							Name:        "inner",
							Type:        "STRING",
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		bqSchema := tc.schema.asTableSchema()
		if !reflect.DeepEqual(bqSchema, tc.bqSchema) {
			t.Errorf("converting to TableSchema: got:\n%v\nwant:\n%v", bqSchema, tc.bqSchema)
		}
		schema := convertTableSchema(tc.bqSchema)
		if !reflect.DeepEqual(schema, tc.schema) {
			t.Errorf("converting to Schema: got:\n%v\nwant:\n%v", schema, tc.schema)
		}
	}
}

type allStrings struct {
	String    string
	ByteSlice []byte
}

type allSignedIntegers struct {
	Int64 int64
	Int32 int32
	Int16 int16
	Int8  int8
	Int   int
}

type allUnsignedIntegers struct {
	Uint64  uint64
	Uint32  uint32
	Uint16  uint16
	Uint8   uint8
	Uintptr uintptr
	Uint    uint
}

type allFloat struct {
	Float64 float64
	Float32 float32
	// NOTE: Complex32 and Complex64 are unsupported by BigQuery
}

type allBoolean struct {
	Bool bool
}

type allTime struct {
	Time time.Time
}

func TestSimpleInference(t *testing.T) {
	testCases := []struct {
		in   interface{}
		want Schema
	}{
		{
			in: allSignedIntegers{},
			want: Schema{
				fieldSchema("", "Int64", "INTEGER", false, true),
				fieldSchema("", "Int32", "INTEGER", false, true),
				fieldSchema("", "Int16", "INTEGER", false, true),
				fieldSchema("", "Int8", "INTEGER", false, true),
				fieldSchema("", "Int", "INTEGER", false, true),
			},
		},
		{
			in: allUnsignedIntegers{},
			want: Schema{
				fieldSchema("", "Uint64", "INTEGER", false, true),
				fieldSchema("", "Uint32", "INTEGER", false, true),
				fieldSchema("", "Uint16", "INTEGER", false, true),
				fieldSchema("", "Uint8", "INTEGER", false, true),
				fieldSchema("", "Uintptr", "INTEGER", false, true),
				fieldSchema("", "Uint", "INTEGER", false, true),
			},
		},
		{
			in: allFloat{},
			want: Schema{
				fieldSchema("", "Float64", "FLOAT", false, true),
				fieldSchema("", "Float32", "FLOAT", false, true),
			},
		},
		{
			in: allBoolean{},
			want: Schema{
				fieldSchema("", "Bool", "BOOLEAN", false, true),
			},
		},
		{
			in: allTime{},
			want: Schema{
				fieldSchema("", "Time", "TIMESTAMP", false, true),
			},
		},
		{
			in: allStrings{},
			want: Schema{
				fieldSchema("", "String", "STRING", false, true),
				fieldSchema("", "ByteSlice", "STRING", false, true),
			},
		},
	}
	for i, tc := range testCases {
		got, err := InferSchema(tc.in)
		if err != nil {
			t.Fatalf("%d: error inferring TableSchema: %v", i, err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%d: inferring TableSchema: got:\n%#v\nwant:\n%#v", i, got, tc.want)
		}
	}
}

type containsNested struct {
	hidden    string
	NotNested int
	Nested    struct {
		Inside int
	}
}

type containsDoubleNested struct {
	NotNested int
	Nested    struct {
		InsideNested struct {
			Inside int
		}
	}
}

func TestNestedInference(t *testing.T) {
	testCases := []struct {
		in   interface{}
		want Schema
	}{
		{
			in: containsNested{},
			want: Schema{
				fieldSchema("", "NotNested", "INTEGER", false, true),
				&FieldSchema{
					Name:     "Nested",
					Required: true,
					Type:     "RECORD",
					Schema: []*FieldSchema{
						{
							Name:     "Inside",
							Type:     "INTEGER",
							Required: true,
						},
					},
				},
			},
		},
		{
			in: containsDoubleNested{},
			want: Schema{
				fieldSchema("", "NotNested", "INTEGER", false, true),
				&FieldSchema{
					Name:     "Nested",
					Required: true,
					Type:     "RECORD",
					Schema: []*FieldSchema{
						{
							Name:     "InsideNested",
							Required: true,
							Type:     "RECORD",
							Schema: []*FieldSchema{
								{
									Name:     "Inside",
									Type:     "INTEGER",
									Required: true,
								},
							},
						},
					},
				},
			},
		},
	}

	for i, tc := range testCases {
		got, err := InferSchema(tc.in)
		if err != nil {
			t.Fatalf("%d: error inferring TableSchema: %v", i, err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%d: inferring TableSchema: got:\n%#v\nwant:\n%#v", i, got, tc.want)
		}
	}
}

type simpleRepeated struct {
	NotRepeated       []byte
	RepeatedByteSlice [][]byte
	Repeated          []int
}

type simpleNestedRepeated struct {
	NotRepeated int
	Repeated    []struct {
		Inside int
	}
}

func TestRepeatedInference(t *testing.T) {
	testCases := []struct {
		in   interface{}
		want Schema
	}{
		{
			in: simpleRepeated{},
			want: Schema{
				fieldSchema("", "NotRepeated", "STRING", false, true),
				fieldSchema("", "RepeatedByteSlice", "STRING", true, false),
				fieldSchema("", "Repeated", "INTEGER", true, false),
			},
		},
		{
			in: simpleNestedRepeated{},
			want: Schema{
				fieldSchema("", "NotRepeated", "INTEGER", false, true),
				&FieldSchema{
					Name:     "Repeated",
					Repeated: true,
					Type:     "RECORD",
					Schema: []*FieldSchema{
						{
							Name:     "Inside",
							Type:     "INTEGER",
							Required: true,
						},
					},
				},
			},
		},
	}

	for i, tc := range testCases {
		got, err := InferSchema(tc.in)
		if err != nil {
			t.Fatalf("%d: error inferring TableSchema: %v", i, err)
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%d: inferring TableSchema: got:\n%#v\nwant:\n%#v", i, got, tc.want)
		}
	}
}

type Embedded struct {
	Embedded int
}

type nestedEmbedded struct {
	Embedded
}

func TestSchemaErrors(t *testing.T) {
	testCases := []struct {
		in  interface{}
		err error
	}{
		{
			in:  []byte{},
			err: errNoStruct,
		},
		{
			in:  new(int),
			err: errNoStruct,
		},
		{
			in:  new(allStrings),
			err: errNoStruct,
		},
		{
			in:  struct{ Complex complex64 }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  struct{ Map map[string]int }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  struct{ Chan chan bool }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  struct{ Ptr *int }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  struct{ Interface interface{} }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  struct{ MultiDimensional [][]int }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  struct{ MultiDimensional [][][]byte }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  struct{ ChanSlice []chan bool }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  struct{ NestedChan struct{ Chan []chan bool } }{},
			err: errUnsupportedFieldType,
		},
		{
			in:  nestedEmbedded{},
			err: errUnsupportedFieldType,
		},
	}
	for i, tc := range testCases {
		want := tc.err
		_, got := InferSchema(tc.in)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%d: inferring TableSchema: got:\n%#v\nwant:\n%#v", i, got, want)
		}
	}
}
