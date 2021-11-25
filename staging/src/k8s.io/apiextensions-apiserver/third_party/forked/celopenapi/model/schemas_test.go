// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"reflect"
	"testing"

	"github.com/google/cel-go/common/types"

	"google.golang.org/protobuf/proto"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

func TestSchemaDeclType(t *testing.T) {
	ts := testSchema()
	cust := SchemaDeclType(ts, false)
	if cust.TypeName() != "object" {
		t.Errorf("incorrect type name, got %v, wanted object", cust.TypeName())
	}
	if len(cust.Fields) != 4 {
		t.Errorf("incorrect number of fields, got %d, wanted 4", len(cust.Fields))
	}
	for _, f := range cust.Fields {
		prop, found := ts.Properties[f.Name]
		if !found {
			t.Errorf("type field not found in schema, field: %s", f.Name)
		}
		fdv := f.DefaultValue()
		if prop.Default.Object != nil {
			pdv := types.DefaultTypeAdapter.NativeToValue(prop.Default.Object)
			if !reflect.DeepEqual(fdv, pdv) {
				t.Errorf("field and schema do not agree on default value for field: %s, field value: %v, schema default: %v", f.Name, fdv, pdv)
			}
		}
		if (prop.ValueValidation == nil || len(prop.ValueValidation.Enum) == 0) && len(f.EnumValues()) != 0 {
			t.Errorf("field had more enum values than the property. field: %s", f.Name)
		}
		if prop.ValueValidation != nil {
			fevs := f.EnumValues()
			for _, fev := range fevs {
				found := false
				for _, pev := range prop.ValueValidation.Enum {
					celpev := types.DefaultTypeAdapter.NativeToValue(pev.Object)
					if reflect.DeepEqual(fev, celpev) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf(
						"could not find field enum value in property definition. field: %s, enum: %v",
						f.Name, fev)
				}
			}
		}
	}
	if ts.ValueValidation != nil {
		for _, name := range ts.ValueValidation.Required {
			df, found := cust.FindField(name)
			if !found {
				t.Errorf("custom type missing required field. field=%s", name)
			}
			if !df.Required {
				t.Errorf("field marked as required in schema, but optional in type. field=%s", df.Name)
			}
		}
	}
}

func TestSchemaDeclTypes(t *testing.T) {
	ts := testSchema()
	cust := SchemaDeclType(ts, true).MaybeAssignTypeName("CustomObject")
	typeMap := FieldTypeMap("CustomObject", cust)
	nested, _ := cust.FindField("nested")
	metadata, _ := cust.FindField("metadata")
	expectedObjTypeMap := map[string]*DeclType{
		"CustomObject":          cust,
		"CustomObject.nested":   nested.Type,
		"CustomObject.metadata": metadata.Type,
	}
	objTypeMap := map[string]*DeclType{}
	for name, t := range typeMap {
		if t.IsObject() {
			objTypeMap[name] = t
		}
	}
	if len(objTypeMap) != len(expectedObjTypeMap) {
		t.Errorf("got different type set. got=%v, wanted=%v", objTypeMap, expectedObjTypeMap)
	}
	for exp, expType := range expectedObjTypeMap {
		actType, found := objTypeMap[exp]
		if !found {
			t.Errorf("missing type in rule types: %s", exp)
			continue
		}
		if !proto.Equal(expType.ExprType(), actType.ExprType()) {
			t.Errorf("incompatible CEL types. got=%v, wanted=%v", actType.ExprType(), expType.ExprType())
		}
	}
}

func testSchema() *schema.Structural {
	// Manual construction of a schema with the following definition:
	//
	// schema:
	//   type: object
	//   metadata:
	//     custom_type: "CustomObject"
	//   required:
	//     - name
	//     - value
	//   properties:
	//     name:
	//       type: string
	//     nested:
	//       type: object
	//       properties:
	//         subname:
	//           type: string
	//         flags:
	//           type: object
	//           additionalProperties:
	//             type: boolean
	//         dates:
	//           type: array
	//           items:
	//             type: string
	//             format: date-time
	//      metadata:
	//        type: object
	//        additionalProperties:
	//          type: object
	//          properties:
	//            key:
	//              type: string
	//            values:
	//              type: array
	//              items: string
	//     value:
	//       type: integer
	//       format: int64
	//       default: 1
	//       enum: [1,2,3]
	ts := &schema.Structural{
		Generic: schema.Generic{
			Type: "object",
		},
		Properties: map[string]schema.Structural{
			"name": {
				Generic: schema.Generic{
					Type: "string",
				},
			},
			"value": {
				Generic: schema.Generic{
					Type:    "integer",
					Default: schema.JSON{Object: int64(1)},
				},
				ValueValidation: &schema.ValueValidation{
					Format: "int64",
					Enum:   []schema.JSON{{Object: int64(1)}, {Object: int64(2)}, {Object: int64(3)}},
				},
			},
			"nested": {
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"subname": {
						Generic: schema.Generic{
							Type: "string",
						},
					},
					"flags": {
						Generic: schema.Generic{
							Type: "object",
							AdditionalProperties: &schema.StructuralOrBool{
								Structural: &schema.Structural{
									Generic: schema.Generic{
										Type: "boolean",
									},
								},
							},
						},
					},
					"dates": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "string",
							},
							ValueValidation: &schema.ValueValidation{
								Format: "date-time",
							},
						},
					},
				},
			},
			"metadata": {
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"name": {
						Generic: schema.Generic{
							Type: "string",
						},
					},
					"value": {
						Generic: schema.Generic{
							Type: "array",
						},
						Items: &schema.Structural{
							Generic: schema.Generic{
								Type: "string",
							},
						},
					},
				},
			},
		},
	}
	return ts
}
