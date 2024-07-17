/*
Copyright 2022 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package model

import (
	"reflect"
	"testing"

	"github.com/google/cel-go/common/types"

	"google.golang.org/protobuf/proto"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	apiservercel "k8s.io/apiserver/pkg/cel"
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
	typeMap := apiservercel.FieldTypeMap("CustomObject", cust)
	nested, _ := cust.FindField("nested")
	metadata, _ := cust.FindField("metadata")
	expectedObjTypeMap := map[string]*apiservercel.DeclType{
		"CustomObject":          cust,
		"CustomObject.nested":   nested.Type,
		"CustomObject.metadata": metadata.Type,
	}
	objTypeMap := map[string]*apiservercel.DeclType{}
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
		expT, err := expType.ExprType()
		if err != nil {
			t.Errorf("fail to get cel type: %s", err)
		}
		actT, err := actType.ExprType()
		if err != nil {
			t.Errorf("fail to get cel type: %s", err)
		}
		if !proto.Equal(expT, actT) {
			t.Errorf("incompatible CEL types. got=%v, wanted=%v", expT, actT)
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

func arraySchema(arrayType, format string, maxItems *int64) *schema.Structural {
	return &schema.Structural{
		Generic: schema.Generic{
			Type: "array",
		},
		Items: &schema.Structural{
			Generic: schema.Generic{
				Type: arrayType,
			},
			ValueValidation: &schema.ValueValidation{
				Format: format,
			},
		},
		ValueValidation: &schema.ValueValidation{
			MaxItems: maxItems,
		},
	}
}

func TestEstimateMaxLengthJSON(t *testing.T) {
	type maxLengthTest struct {
		Name                string
		InputSchema         *schema.Structural
		ExpectedMaxElements int64
		ExpectNilType       bool
	}
	tests := []maxLengthTest{
		{
			Name:        "booleanArray",
			InputSchema: arraySchema("boolean", "", nil),
			// expected JSON is [true,true,...], so our length should be (maxRequestSizeBytes - 2) / 5
			ExpectedMaxElements: 629145,
		},
		{
			Name:        "durationArray",
			InputSchema: arraySchema("string", "duration", nil),
			// expected JSON is ["0","0",...] so our length should be (maxRequestSizeBytes - 2) / 4
			ExpectedMaxElements: 786431,
		},
		{
			Name:        "datetimeArray",
			InputSchema: arraySchema("string", "date-time", nil),
			// expected JSON is ["2000-01-01T01:01:01","2000-01-01T01:01:01",...] so our length should be (maxRequestSizeBytes - 2) / 22
			ExpectedMaxElements: 142987,
		},
		{
			Name:        "dateArray",
			InputSchema: arraySchema("string", "date", nil),
			// expected JSON is ["2000-01-01","2000-01-02",...] so our length should be (maxRequestSizeBytes - 2) / 13
			ExpectedMaxElements: 241978,
		},
		{
			Name:        "numberArray",
			InputSchema: arraySchema("integer", "", nil),
			// expected JSON is [0,0,...] so our length should be (maxRequestSizeBytes - 2) / 2
			ExpectedMaxElements: 1572863,
		},
		{
			Name:        "stringArray",
			InputSchema: arraySchema("string", "", nil),
			// expected JSON is ["","",...] so our length should be (maxRequestSizeBytes - 2) / 3
			ExpectedMaxElements: 1048575,
		},
		{
			Name: "stringMap",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
					AdditionalProperties: &schema.StructuralOrBool{Structural: &schema.Structural{
						Generic: schema.Generic{
							Type: "string",
						},
					}},
				},
			},
			// expected JSON is {"":"","":"",...} so our length should be (3000000 - 2) / 6
			ExpectedMaxElements: 393215,
		},
		{
			Name: "objectOptionalPropertyArray",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Items: &schema.Structural{
					Generic: schema.Generic{
						Type: "object",
					},
					Properties: map[string]schema.Structural{
						"required": {
							Generic: schema.Generic{
								Type: "string",
							},
						},
						"optional": {
							Generic: schema.Generic{
								Type: "string",
							},
						},
					},
					ValueValidation: &schema.ValueValidation{
						Required: []string{"required"},
					},
				},
			},
			// expected JSON is [{"required":"",},{"required":"",},...] so our length should be (maxRequestSizeBytes - 2) / 17
			ExpectedMaxElements: 185042,
		},
		{
			Name:        "arrayWithLength",
			InputSchema: arraySchema("integer", "int64", maxPtr(10)),
			// manually set by MaxItems
			ExpectedMaxElements: 10,
		},
		{
			Name: "stringWithLength",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					MaxLength: maxPtr(20),
				},
			},
			// manually set by MaxLength, but we expect a 4x multiplier compared to the original input
			// since OpenAPIv3 maxLength uses code points, but DeclType works with bytes
			ExpectedMaxElements: 80,
		},
		{
			Name: "mapWithLength",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
					AdditionalProperties: &schema.StructuralOrBool{Structural: &schema.Structural{
						Generic: schema.Generic{
							Type: "string",
						},
					}},
				},
				ValueValidation: &schema.ValueValidation{
					Format:        "string",
					MaxProperties: maxPtr(15),
				},
			},
			// manually set by MaxProperties
			ExpectedMaxElements: 15,
		},
		{
			Name: "durationMaxSize",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					Format: "duration",
				},
			},
			// should be exactly equal to maxDurationSizeJSON
			ExpectedMaxElements: apiservercel.MaxDurationSizeJSON,
		},
		{
			Name: "dateSize",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					Format: "date",
				},
			},
			// should be exactly equal to dateSizeJSON
			ExpectedMaxElements: apiservercel.JSONDateSize,
		},
		{
			Name: "maxdatetimeSize",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					Format: "date-time",
				},
			},
			// should be exactly equal to maxDatetimeSizeJSON
			ExpectedMaxElements: apiservercel.MaxDatetimeSizeJSON,
		},
		{
			Name: "maxintOrStringSize",
			InputSchema: &schema.Structural{
				Extensions: schema.Extensions{
					XIntOrString: true,
				},
			},
			// should be exactly equal to maxRequestSizeBytes - 2 (to allow for quotes in the case of a string)
			ExpectedMaxElements: apiservercel.DefaultMaxRequestSizeBytes - 2,
		},
		{
			Name: "objectDefaultFieldArray",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Items: &schema.Structural{
					Generic: schema.Generic{
						Type: "object",
					},
					Properties: map[string]schema.Structural{
						"field": {
							Generic: schema.Generic{
								Type:    "string",
								Default: schema.JSON{Object: "default"},
							},
						},
					},
					ValueValidation: &schema.ValueValidation{
						Required: []string{"field"},
					},
				},
			},
			// expected JSON is [{},{},...] so our length should be (maxRequestSizeBytes - 2) / 3
			ExpectedMaxElements: 1048575,
		},
		{
			Name: "byteStringSize",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					Format: "byte",
				},
			},
			// expected JSON is "" so our length should be (maxRequestSizeBytes - 2)
			ExpectedMaxElements: 3145726,
		},
		{
			Name: "byteStringSetMaxLength",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					Format:    "byte",
					MaxLength: maxPtr(20),
				},
			},
			// note that unlike regular strings we don't have to take unicode into account,
			// so we expect the max length to be exactly equal to the user-supplied one
			ExpectedMaxElements: 20,
		},
		{
			Name: "Property under array",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "array",
				},
				Properties: map[string]schema.Structural{
					"field": {
						Generic: schema.Generic{
							Type:    "string",
							Default: schema.JSON{Object: "default"},
						},
					},
				},
			},
			// Got nil for delType
			ExpectedMaxElements: 0,
			ExpectNilType:       true,
		},
		{
			Name: "Items under object",
			InputSchema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Items: &schema.Structural{
					Generic: schema.Generic{
						Type: "array",
					},
					Properties: map[string]schema.Structural{
						"field": {
							Generic: schema.Generic{
								Type:    "string",
								Default: schema.JSON{Object: "default"},
							},
						},
					},
					ValueValidation: &schema.ValueValidation{
						Required: []string{"field"},
					},
				},
			},
			// Skip items under object for schema conversion.
			ExpectedMaxElements: 0,
		},
	}
	for _, testCase := range tests {
		t.Run(testCase.Name, func(t *testing.T) {
			decl := SchemaDeclType(testCase.InputSchema, false)
			if decl != nil && decl.MaxElements != testCase.ExpectedMaxElements {
				t.Errorf("wrong maxElements (got %d, expected %d)", decl.MaxElements, testCase.ExpectedMaxElements)
			}
			if testCase.ExpectNilType && decl != nil {
				t.Errorf("expected nil type, got %v", decl)
			}
		})
	}
}

func maxPtr(max int64) *int64 {
	return &max
}

func genNestedSchema(depth int) *schema.Structural {
	var generator func(d int) schema.Structural
	generator = func(d int) schema.Structural {
		nodeTemplate := schema.Structural{
			Generic: schema.Generic{
				Type:                 "object",
				AdditionalProperties: &schema.StructuralOrBool{},
			},
		}
		if d == 1 {
			return nodeTemplate
		} else {
			mapType := generator(d - 1)
			nodeTemplate.Generic.AdditionalProperties.Structural = &mapType
			return nodeTemplate
		}
	}
	schema := generator(depth)
	return &schema
}

func BenchmarkDeeplyNestedSchemaDeclType(b *testing.B) {
	benchmarkSchema := genNestedSchema(10)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		SchemaDeclType(benchmarkSchema, false)
	}
}
