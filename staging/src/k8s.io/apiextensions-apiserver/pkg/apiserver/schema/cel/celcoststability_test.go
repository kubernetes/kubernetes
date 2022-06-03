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

package cel

import (
	"context"
	"fmt"
	"math"
	"testing"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestCelCostStability(t *testing.T) {
	cases := []struct {
		name       string
		schema     *schema.Structural
		obj        map[string]interface{}
		expectCost map[string]int64
	}{
		{name: "integers",
			// 1st obj and schema args are for "self.val1" field, 2nd for "self.val2" and so on.
			obj:    objs(math.MaxInt64, math.MaxInt64, math.MaxInt32, math.MaxInt32, math.MaxInt64, math.MaxInt64),
			schema: schemas(integerType, integerType, int32Type, int32Type, int64Type, int64Type),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", fmt.Sprintf("%d", math.MaxInt64)): 11,
				"self.val1 == self.val6":                              5, // integer with no format is the same as int64
				"type(self.val1) == int":                              4,
				fmt.Sprintf("self.val3 + 1 == %d + 1", math.MaxInt32): 5, // CEL integers are 64 bit
			},
		},
		{name: "numbers",
			obj:    objs(math.MaxFloat64, math.MaxFloat64, math.MaxFloat32, math.MaxFloat32, math.MaxFloat64, math.MaxFloat64, int64(1)),
			schema: schemas(numberType, numberType, floatType, floatType, doubleType, doubleType, doubleType),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", fmt.Sprintf("%f", math.MaxFloat64)): 11,
				"self.val1 == self.val6":    5, // number with no format is the same as float64
				"type(self.val1) == double": 4,

				// Use a int64 value with a number openAPI schema type since float representations of whole numbers
				// (e.g. 1.0, 0.0) can convert to int representations (e.g. 1, 0) in yaml to json translation, and
				// then get parsed as int64s.
				"type(self.val7) == double": 4,
				"self.val7 == 1.0":          3,
			},
		},
		{name: "numeric comparisons",
			obj: objs(
				int64(5),      // val1, integer type, integer value
				float64(10.0), // val4, number type, parsed from decimal literal
				float64(10.0), // val5, float type, parsed from decimal literal
				float64(10.0), // val6, double type, parsed from decimal literal
				int64(10),     // val7, number type, parsed from integer literal
				int64(10),     // val8, float type, parsed from integer literal
				int64(10),     // val9, double type, parsed from integer literal
			),
			schema: schemas(integerType, numberType, floatType, doubleType, numberType, floatType, doubleType),
			expectCost: map[string]int64{
				// xref: https://github.com/google/cel-spec/wiki/proposal-210

				// compare integers with all float types
				"double(self.val1) < self.val4": 6,
				"self.val1 < int(self.val4)":    6,
				"double(self.val1) < self.val5": 6,
				"self.val1 < int(self.val5)":    6,
				"double(self.val1) < self.val6": 6,
				"self.val1 < int(self.val6)":    6,

				// compare literal integers and floats
				"double(5) < 10.0": 1,
				"5 < int(10.0)":    1,

				// compare integers with literal floats
				"double(self.val1) < 10.0": 4,
			},
		},
		{name: "unicode strings",
			obj:    objs("Rook takes 👑", "Rook takes 👑"),
			schema: schemas(stringType, stringType),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", "'Rook takes 👑'"): 14,
				"self.val1.startsWith('Rook')":    4,
				"!self.val1.startsWith('knight')": 5,
				"self.val1.matches('^[^0-9]*$')":  8,
				"!self.val1.matches('^[0-9]*$')":  7,
				"type(self.val1) == string":       4,
				"size(self.val1) == 12":           4,

				// string functions (https://github.com/google/cel-go/blob/v0.9.0/ext/strings.go)
				"self.val1.charAt(3) == 'k'":                       4,
				"self.val1.indexOf('o') == 1":                      4,
				"self.val1.indexOf('o', 2) == 2":                   4,
				"self.val1.replace(' ', 'x') == 'Rookxtakesx👑'":    7,
				"self.val1.replace(' ', 'x', 1) == 'Rookxtakes 👑'": 7,
				"self.val1.split(' ') == ['Rook', 'takes', '👑']":   6,
				"self.val1.split(' ', 2) == ['Rook', 'takes 👑']":   6,
				"self.val1.substring(5) == 'takes 👑'":              5,
				"self.val1.substring(0, 4) == 'Rook'":              5,
				"self.val1.substring(4, 10).trim() == 'takes'":     6,
				"self.val1.upperAscii() == 'ROOK TAKES 👑'":         6,
				"self.val1.lowerAscii() == 'rook takes 👑'":         6,
			},
		},
		{name: "escaped strings",
			obj:    objs("l1\nl2", "l1\nl2"),
			schema: schemas(stringType, stringType),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", "'l1\\nl2'"): 11,
				"self.val1 == '''l1\nl2'''": 3,
			},
		},
		{name: "bytes",
			obj:    objs("QUI=", "QUI="),
			schema: schemas(byteType, byteType),
			expectCost: map[string]int64{
				"self.val1 == self.val2":   5,
				"self.val1 == b'AB'":       3,
				"type(self.val1) == bytes": 4,
				"size(self.val1) == 2":     4,
			},
		},
		{name: "booleans",
			obj:    objs(true, true, false, false),
			schema: schemas(booleanType, booleanType, booleanType, booleanType),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", "true"): 11,
				"self.val1 != self.val4":  5,
				"type(self.val1) == bool": 4,
			},
		},
		{name: "duration format",
			obj:    objs("1h2m3s4ms", "1h2m3s4ms"),
			schema: schemas(durationFormat, durationFormat),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", "duration('1h2m3s4ms')"): 11,
				"self.val1 == duration('1h2m') + duration('3s4ms')":                                  4,
				"self.val1.getHours() == 1":                                                          4,
				"type(self.val1) == google.protobuf.Duration":                                        4,
			},
		},
		{name: "date format",
			obj:    objs("1997-07-16", "1997-07-16"),
			schema: schemas(dateFormat, dateFormat),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", "timestamp('1997-07-16T00:00:00.000Z')"): 11,
				"self.val1.getDate() == 16":                    4,
				"type(self.val1) == google.protobuf.Timestamp": 4,
			},
		},
		{name: "date-time format",
			obj:    objs("2011-08-18T19:03:37.010000000+01:00", "2011-08-18T19:03:37.010000000+01:00"),
			schema: schemas(dateTimeFormat, dateTimeFormat),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", "timestamp('2011-08-18T19:03:37.010+01:00')"): 11,
				"self.val1 == timestamp('2011-08-18T00:00:00.000+01:00') + duration('19h3m37s10ms')":                      4,
				"self.val1.getDate('01:00') == 18":             4,
				"type(self.val1) == google.protobuf.Timestamp": 4,
			},
		},
		{name: "enums",
			obj: map[string]interface{}{"enumStr": "Pending"},
			schema: objectTypePtr(map[string]schema.Structural{"enumStr": {
				Generic: schema.Generic{
					Type: "string",
				},
				ValueValidation: &schema.ValueValidation{
					Enum: []schema.JSON{
						{Object: "Pending"},
						{Object: "Available"},
						{Object: "Bound"},
						{Object: "Released"},
						{Object: "Failed"},
					},
				},
			}}),
			expectCost: map[string]int64{
				"self.enumStr == 'Pending'":                3,
				"self.enumStr in ['Pending', 'Available']": 2,
			},
		},
		{name: "conversions",
			obj:    objs(int64(10), 10.0, 10.49, 10.5, true, "10", "MTA=", "3723.004s", "1h2m3s4ms", "2011-08-18T19:03:37.01+01:00", "2011-08-18T19:03:37.01+01:00", "2011-08-18T00:00:00Z", "2011-08-18"),
			schema: schemas(integerType, numberType, numberType, numberType, booleanType, stringType, byteType, stringType, durationFormat, stringType, dateTimeFormat, stringType, dateFormat),
			expectCost: map[string]int64{
				"int(self.val2) == self.val1":         6,
				"double(self.val1) == self.val2":      6,
				"bytes(self.val6) == self.val7":       6,
				"string(self.val1) == self.val6":      6,
				"string(self.val4) == '10.5'":         4,
				"string(self.val7) == self.val6":      6,
				"duration(self.val8) == self.val9":    6,
				"timestamp(self.val10) == self.val11": 6,
				"string(self.val11) == self.val10":    8,
				"timestamp(self.val12) == self.val13": 6,
				"string(self.val13) == self.val12":    7,
			},
		},
		{name: "lists",
			obj:    objs([]interface{}{1, 2, 3}, []interface{}{1, 2, 3}),
			schema: schemas(listType(&integerType), listType(&integerType)),
			expectCost: map[string]int64{
				ValsEqualThemselvesAndDataLiteral("self.val1", "self.val2", "[1, 2, 3]"): 11,
				"1 in self.val1":                              5,
				"self.val2[0] in self.val1":                   8,
				"!(0 in self.val1)":                           6,
				"self.val1 + self.val2 == [1, 2, 3, 1, 2, 3]": 6,
				"self.val1 + [4, 5] == [1, 2, 3, 4, 5]":       4,
			},
		},
		{name: "listSets",
			obj:    objs([]interface{}{"a", "b", "c"}, []interface{}{"a", "c", "b"}),
			schema: schemas(listSetType(&stringType), listSetType(&stringType)),
			expectCost: map[string]int64{
				// equal even though order is different
				"self.val1 == ['c', 'b', 'a']":                   3,
				"self.val1 == self.val2":                         5,
				"'a' in self.val1":                               5,
				"self.val2[0] in self.val1":                      8,
				"!('x' in self.val1)":                            6,
				"self.val1 + self.val2 == ['a', 'b', 'c']":       6,
				"self.val1 + ['c', 'd'] == ['a', 'b', 'c', 'd']": 4,
			},
		},
		{name: "listMaps",
			obj: map[string]interface{}{
				"objs": []interface{}{
					[]interface{}{
						map[string]interface{}{"k": "a", "v": "1"},
						map[string]interface{}{"k": "b", "v": "2"},
					},
					[]interface{}{
						map[string]interface{}{"k": "b", "v": "2"},
						map[string]interface{}{"k": "a", "v": "1"},
					},
					[]interface{}{
						map[string]interface{}{"k": "b", "v": "3"},
						map[string]interface{}{"k": "a", "v": "1"},
					},
					[]interface{}{
						map[string]interface{}{"k": "c", "v": "4"},
					},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"objs": listType(listMapTypePtr([]string{"k"}, objectTypePtr(map[string]schema.Structural{
					"k": stringType,
					"v": stringType,
				}))),
			}),
			expectCost: map[string]int64{
				"self.objs[0] == self.objs[1]":                7,  // equal even though order is different
				"self.objs[0] + self.objs[2] == self.objs[2]": 11, // rhs overwrites lhs values
				"self.objs[2] + self.objs[0] == self.objs[0]": 11,

				"self.objs[0] == [self.objs[0][0], self.objs[0][1]]": 22, // equal against a declared list
				"self.objs[0] == [self.objs[0][1], self.objs[0][0]]": 22,

				"self.objs[2] + [self.objs[0][0], self.objs[0][1]] == self.objs[0]": 26, // concat against a declared list
				"size(self.objs[0] + [self.objs[3][0]]) == 3":                       20,
			},
		},
		{name: "maps",
			obj:    objs(map[string]interface{}{"k1": "a", "k2": "b"}, map[string]interface{}{"k2": "b", "k1": "a"}),
			schema: schemas(mapType(&stringType), mapType(&stringType)),
			expectCost: map[string]int64{
				"self.val1 == self.val2":              5, // equal even though order is different
				"'k1' in self.val1":                   3,
				"!('k3' in self.val1)":                4,
				"self.val1 == {'k1': 'a', 'k2': 'b'}": 3,
			},
		},
		{name: "objects",
			obj: map[string]interface{}{
				"objs": []interface{}{
					map[string]interface{}{"f1": "a", "f2": "b"},
					map[string]interface{}{"f1": "a", "f2": "b"},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"objs": listType(objectTypePtr(map[string]schema.Structural{
					"f1": stringType,
					"f2": stringType,
				})),
			}),
			expectCost: map[string]int64{
				"self.objs[0] == self.objs[1]": 7,
			},
		},
		{name: "object access",
			obj: map[string]interface{}{
				"a": map[string]interface{}{
					"b": 1,
					"d": nil,
				},
				"a1": map[string]interface{}{
					"b1": map[string]interface{}{
						"c1": 4,
					},
				},
				"a3": map[string]interface{}{},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"a": objectType(map[string]schema.Structural{
					"b": integerType,
					"c": integerType,
					"d": withNullable(true, integerType),
				}),
				"a1": objectType(map[string]schema.Structural{
					"b1": objectType(map[string]schema.Structural{
						"c1": integerType,
					}),
					"d2": objectType(map[string]schema.Structural{
						"e2": integerType,
					}),
				}),
			}),
			// https://github.com/google/cel-spec/blob/master/doc/langdef.md#field-selection
			expectCost: map[string]int64{
				"has(self.a.b)":                            2,
				"has(self.a1.b1.c1)":                       3,
				"!(has(self.a1.d2) && has(self.a1.d2.e2))": 3, // must check intermediate optional fields (see below no such key error for d2)
				"!has(self.a1.d2)":                         3,
			},
		},
		{name: "map access",
			obj: map[string]interface{}{
				"val": map[string]interface{}{
					"b": 1,
					"d": 2,
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"val": mapType(&integerType),
			}),
			expectCost: map[string]int64{
				// idiomatic map access
				"!('a' in self.val)": 4,
				"'b' in self.val":    3,
				"!('c' in self.val)": 4,
				"'d' in self.val":    3,
				// field selection also possible if map key is a valid CEL identifier
				"!has(self.val.a)":                               3,
				"has(self.val.b)":                                2,
				"!has(self.val.c)":                               3,
				"has(self.val.d)":                                2,
				"self.val.all(k, self.val[k] > 0)":               17,
				"self.val.exists_one(k, self.val[k] == 2)":       14,
				"!self.val.exists_one(k, self.val[k] > 0)":       17,
				"size(self.val) == 2":                            4,
				"size(self.val.filter(k, self.val[k] > 1)) == 1": 26,
			},
		},
		{name: "listMap access",
			obj: map[string]interface{}{
				"listMap": []interface{}{
					map[string]interface{}{"k": "a1", "v": "b1"},
					map[string]interface{}{"k": "a2", "v": "b2"},
					map[string]interface{}{"k": "a3", "v": "b3", "v2": "z"},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"listMap": listMapType([]string{"k"}, objectTypePtr(map[string]schema.Structural{
					"k":  stringType,
					"v":  stringType,
					"v2": stringType,
				})),
			}),
			expectCost: map[string]int64{
				"has(self.listMap[0].v)":                             3,
				"self.listMap.all(m, m.k.startsWith('a'))":           21,
				"self.listMap.all(m, !has(m.v2) || m.v2 == 'z')":     21,
				"self.listMap.exists(m, m.k.endsWith('1'))":          13,
				"self.listMap.exists_one(m, m.k == 'a3')":            15,
				"!self.listMap.all(m, m.k.endsWith('1'))":            18,
				"!self.listMap.exists(m, m.v == 'x')":                25,
				"!self.listMap.exists_one(m, m.k.startsWith('a'))":   20,
				"size(self.listMap.filter(m, m.k == 'a1')) == 1":     27,
				"self.listMap.exists(m, m.k == 'a1' && m.v == 'b1')": 16,
				"self.listMap.map(m, m.v).exists(v, v == 'b1')":      55,

				// test comprehensions where the field used in predicates is unset on all but one of the elements:
				// - with has checks:

				"self.listMap.exists(m, has(m.v2) && m.v2 == 'z')":             21,
				"!self.listMap.all(m, has(m.v2) && m.v2 != 'z')":               10,
				"self.listMap.exists_one(m, has(m.v2) && m.v2 == 'z')":         12,
				"self.listMap.filter(m, has(m.v2) && m.v2 == 'z').size() == 1": 24,
				// undocumented overload of map that takes a filter argument. This is the same as .filter().map()
				"self.listMap.map(m, has(m.v2) && m.v2 == 'z', m.v2).size() == 1":           25,
				"self.listMap.filter(m, has(m.v2) && m.v2 == 'z').map(m, m.v2).size() == 1": 39,
				// - without has checks:

				// all() and exists() macros ignore errors from predicates so long as the condition holds for at least one element
				"self.listMap.exists(m, m.v2 == 'z')": 24,
				"!self.listMap.all(m, m.v2 != 'z')":   22,
			},
		},
		{name: "list access",
			obj: map[string]interface{}{
				"array": []interface{}{1, 1, 2, 2, 3, 3, 4, 5},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"array": listType(&integerType),
			}),
			expectCost: map[string]int64{
				"2 in self.array":                                                10,
				"self.array.all(e, e > 0)":                                       43,
				"self.array.exists(e, e > 2)":                                    36,
				"self.array.exists_one(e, e > 4)":                                22,
				"!self.array.all(e, e < 2)":                                      21,
				"!self.array.exists(e, e < 0)":                                   52,
				"!self.array.exists_one(e, e == 2)":                              25,
				"self.array.all(e, e < 100)":                                     43,
				"size(self.array.filter(e, e%2 == 0)) == 3":                      68,
				"self.array.map(e, e * 20).filter(e, e > 50).exists(e, e == 60)": 194,
				"size(self.array) == 8":                                          4,
			},
		},
		{name: "listSet access",
			obj: map[string]interface{}{
				"set": []interface{}{1, 2, 3, 4, 5},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"set": listType(&integerType),
			}),
			expectCost: map[string]int64{
				"3 in self.set":                                                    7,
				"self.set.all(e, e > 0)":                                           28,
				"self.set.exists(e, e > 3)":                                        30,
				"self.set.exists_one(e, e == 3)":                                   16,
				"!self.set.all(e, e < 3)":                                          21,
				"!self.set.exists(e, e < 0)":                                       34,
				"!self.set.exists_one(e, e > 3)":                                   19,
				"self.set.all(e, e < 10)":                                          28,
				"size(self.set.filter(e, e%2 == 0)) == 2":                          46,
				"self.set.map(e, e * 20).filter(e, e > 50).exists_one(e, e == 60)": 133,
				"size(self.set) == 5":                                              4,
			},
		},
		{name: "typemeta and objectmeta access specified",
			obj: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":         "foo",
					"generateName": "pickItForMe",
					"namespace":    "xyz",
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"kind":       stringType,
				"apiVersion": stringType,
				"metadata": objectType(map[string]schema.Structural{
					"name":         stringType,
					"generateName": stringType,
				}),
			}),
			expectCost: map[string]int64{
				"self.kind == 'Pod'":                          3,
				"self.apiVersion == 'v1'":                     3,
				"self.metadata.name == 'foo'":                 4,
				"self.metadata.generateName == 'pickItForMe'": 5,
			},
		},
		{name: "typemeta and objectmeta access not specified",
			obj: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"name":         "foo",
					"generateName": "pickItForMe",
					"namespace":    "xyz",
				},
				"spec": map[string]interface{}{
					"field1": "a",
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"spec": objectType(map[string]schema.Structural{
					"field1": stringType,
				}),
			}),
			expectCost: map[string]int64{
				"self.kind == 'Pod'":                          3,
				"self.apiVersion == 'v1'":                     3,
				"self.metadata.name == 'foo'":                 4,
				"self.metadata.generateName == 'pickItForMe'": 5,
				"self.spec.field1 == 'a'":                     4,
			},
		},

		// Kubernetes special types
		{name: "embedded object",
			obj: map[string]interface{}{
				"embedded": map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Pod",
					"metadata": map[string]interface{}{
						"name":         "foo",
						"generateName": "pickItForMe",
						"namespace":    "xyz",
					},
					"spec": map[string]interface{}{
						"field1": "a",
					},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"embedded": {
					Generic: schema.Generic{Type: "object"},
					Extensions: schema.Extensions{
						XEmbeddedResource: true,
					},
				},
			}),
			expectCost: map[string]int64{
				// 'kind', 'apiVersion', 'metadata.name' and 'metadata.generateName' are always accessible
				// even if not specified in the schema.
				"self.embedded.kind == 'Pod'":                          4,
				"self.embedded.apiVersion == 'v1'":                     4,
				"self.embedded.metadata.name == 'foo'":                 5,
				"self.embedded.metadata.generateName == 'pickItForMe'": 6,
			},
		},
		{name: "embedded object with properties",
			obj: map[string]interface{}{
				"embedded": map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Pod",
					"metadata": map[string]interface{}{
						"name":         "foo",
						"generateName": "pickItForMe",
						"namespace":    "xyz",
					},
					"spec": map[string]interface{}{
						"field1": "a",
					},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"embedded": {
					Generic: schema.Generic{Type: "object"},
					Extensions: schema.Extensions{
						XEmbeddedResource: true,
					},
					Properties: map[string]schema.Structural{
						"kind":       stringType,
						"apiVersion": stringType,
						"metadata": objectType(map[string]schema.Structural{
							"name":         stringType,
							"generateName": stringType,
						}),
						"spec": objectType(map[string]schema.Structural{
							"field1": stringType,
						}),
					},
				},
			}),
			expectCost: map[string]int64{
				// in this case 'kind', 'apiVersion', 'metadata.name' and 'metadata.generateName' are specified in the
				// schema, but they would be accessible even if they were not
				"self.embedded.kind == 'Pod'":                          4,
				"self.embedded.apiVersion == 'v1'":                     4,
				"self.embedded.metadata.name == 'foo'":                 5,
				"self.embedded.metadata.generateName == 'pickItForMe'": 6,
				// the specified embedded fields are accessible
				"self.embedded.spec.field1 == 'a'": 5,
			},
		},
		{name: "embedded object with preserve unknown",
			obj: map[string]interface{}{
				"embedded": map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Pod",
					"metadata": map[string]interface{}{
						"name":         "foo",
						"generateName": "pickItForMe",
						"namespace":    "xyz",
					},
					"spec": map[string]interface{}{
						"field1": "a",
					},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"embedded": {
					Generic: schema.Generic{Type: "object"},
					Extensions: schema.Extensions{
						XPreserveUnknownFields: true,
						XEmbeddedResource:      true,
					},
				},
			}),
			expectCost: map[string]int64{
				// 'kind', 'apiVersion', 'metadata.name' and 'metadata.generateName' are always accessible
				// even if not specified in the schema, regardless of if x-preserve-unknown-fields is set.
				"self.embedded.kind == 'Pod'":                          4,
				"self.embedded.apiVersion == 'v1'":                     4,
				"self.embedded.metadata.name == 'foo'":                 5,
				"self.embedded.metadata.generateName == 'pickItForMe'": 6,

				// the object exists
				"has(self.embedded)": 1,
			},
		},
		{name: "string in intOrString",
			obj: map[string]interface{}{
				"something": "25%",
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"something": intOrStringType(),
			}),
			expectCost: map[string]int64{
				// typical int-or-string usage would be to check both types
				"type(self.something) == int ? self.something == 1 : self.something == '25%'": 7,
				// to require the value be a particular type, guard it with a runtime type check
				"type(self.something) == string && self.something == '25%'": 7,

				// In Kubernetes 1.24 and later, the CEL type returns false for an int-or-string comparison against the
				// other type, making it safe to write validation rules like:
				"self.something == '25%'":                        3,
				"self.something != 1":                            3,
				"self.something == 1 || self.something == '25%'": 6,
				"self.something == '25%' || self.something == 1": 3,

				// Because the type is dynamic it receives no type checking, and evaluates to false when compared to
				// other types at runtime.
				"self.something != ['anything']": 3,
			},
		},
		{name: "int in intOrString",
			obj: map[string]interface{}{
				"something": int64(1),
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"something": intOrStringType(),
			}),
			expectCost: map[string]int64{
				// typical int-or-string usage would be to check both types
				"type(self.something) == int ? self.something == 1 : self.something == '25%'": 7,
				// to require the value be a particular type, guard it with a runtime type check
				"type(self.something) == int && self.something == 1": 7,

				// In Kubernetes 1.24 and later, the CEL type returns false for an int-or-string comparison against the
				// other type, making it safe to write validation rules like:
				"self.something == 1":                            3,
				"self.something != 'some string'":                3,
				"self.something == 1 || self.something == '25%'": 3,
				"self.something == '25%' || self.something == 1": 6,

				// Because the type is dynamic it receives no type checking, and evaluates to false when compared to
				// other types at runtime.
				"self.something != ['anything']": 3,
			},
		},
		{name: "null in intOrString",
			obj: map[string]interface{}{
				"something": nil,
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"something": withNullable(true, intOrStringType()),
			}),
			expectCost: map[string]int64{
				"!has(self.something)": 2,
			},
		},
		{name: "percent comparison using intOrString",
			obj: map[string]interface{}{
				"min":       "50%",
				"current":   5,
				"available": 10,
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"min":       intOrStringType(),
				"current":   integerType,
				"available": integerType,
			}),
			expectCost: map[string]int64{
				// validate that if 'min' is a string that it is a percentage
				`type(self.min) == string && self.min.matches(r'(\d+(\.\d+)?%)')`: 10,
				// validate that 'min' can be either a exact value minimum, or a minimum as a percentage of 'available'
				"type(self.min) == int ? self.current <= self.min : double(self.current) / double(self.available) >= double(self.min.replace('%', '')) / 100.0": 17,
			},
		},
		{name: "preserve unknown fields",
			obj: map[string]interface{}{
				"withUnknown": map[string]interface{}{
					"field1": "a",
					"field2": "b",
				},
				"withUnknownList": []interface{}{
					map[string]interface{}{
						"field1": "a",
						"field2": "b",
					},
					map[string]interface{}{
						"field1": "x",
						"field2": "y",
					},
					map[string]interface{}{
						"field1": "x",
						"field2": "y",
					},
					map[string]interface{}{},
					map[string]interface{}{},
				},
				"withUnknownFieldList": []interface{}{
					map[string]interface{}{
						"fieldOfUnknownType": "a",
					},
					map[string]interface{}{
						"fieldOfUnknownType": 1,
					},
					map[string]interface{}{
						"fieldOfUnknownType": 1,
					},
				},
				"anyvalList":   []interface{}{"a", 2},
				"anyvalMap":    map[string]interface{}{"k": "1"},
				"anyvalField1": 1,
				"anyvalField2": "a",
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"withUnknown": {
					Generic: schema.Generic{Type: "object"},
					Extensions: schema.Extensions{
						XPreserveUnknownFields: true,
					},
				},
				"withUnknownList": listType(&schema.Structural{
					Generic: schema.Generic{Type: "object"},
					Extensions: schema.Extensions{
						XPreserveUnknownFields: true,
					},
				}),
				"withUnknownFieldList": listType(&schema.Structural{
					Generic: schema.Generic{Type: "object"},
					Properties: map[string]schema.Structural{
						"fieldOfUnknownType": {
							Extensions: schema.Extensions{
								XPreserveUnknownFields: true,
							},
						},
					},
				}),
				"anyvalList": listType(&schema.Structural{
					Extensions: schema.Extensions{
						XPreserveUnknownFields: true,
					},
				}),
				"anyvalMap": mapType(&schema.Structural{
					Extensions: schema.Extensions{
						XPreserveUnknownFields: true,
					},
				}),
				"anyvalField1": {
					Extensions: schema.Extensions{
						XPreserveUnknownFields: true,
					},
				},
				"anyvalField2": {
					Extensions: schema.Extensions{
						XPreserveUnknownFields: true,
					},
				},
			}),
			expectCost: map[string]int64{
				"has(self.withUnknown)":            1,
				"self.withUnknownList.size() == 5": 4,
				// fields that are unknown because they were not specified on the object schema are included in equality checks
				"self.withUnknownList[0] != self.withUnknownList[1]": 7,
				"self.withUnknownList[1] == self.withUnknownList[2]": 7,
				"self.withUnknownList[3] == self.withUnknownList[4]": 6,

				// fields specified on the object schema that are unknown because the field's schema is unknown are also included equality checks
				"self.withUnknownFieldList[0] != self.withUnknownFieldList[1]": 7,
				"self.withUnknownFieldList[1] == self.withUnknownFieldList[2]": 7,
			},
		},
		{name: "known and unknown fields",
			obj: map[string]interface{}{
				"withUnknown": map[string]interface{}{
					"known":   1,
					"unknown": "a",
				},
				"withUnknownList": []interface{}{
					map[string]interface{}{
						"known":   1,
						"unknown": "a",
					},
					map[string]interface{}{
						"known":   1,
						"unknown": "b",
					},
					map[string]interface{}{
						"known":   1,
						"unknown": "b",
					},
					map[string]interface{}{
						"known": 1,
					},
					map[string]interface{}{
						"known": 1,
					},
					map[string]interface{}{
						"known": 2,
					},
				},
			},
			schema: &schema.Structural{
				Generic: schema.Generic{
					Type: "object",
				},
				Properties: map[string]schema.Structural{
					"withUnknown": {
						Generic: schema.Generic{Type: "object"},
						Extensions: schema.Extensions{
							XPreserveUnknownFields: true,
						},
						Properties: map[string]schema.Structural{
							"known": integerType,
						},
					},
					"withUnknownList": listType(&schema.Structural{
						Generic: schema.Generic{Type: "object"},
						Extensions: schema.Extensions{
							XPreserveUnknownFields: true,
						},
						Properties: map[string]schema.Structural{
							"known": integerType,
						},
					}),
				},
			},
			expectCost: map[string]int64{
				"self.withUnknown.known == 1": 4,
				// if the unknown fields are the same, they are equal
				"self.withUnknownList[1] == self.withUnknownList[2]": 7,

				// if unknown fields are different, they are not equal
				"self.withUnknownList[0] != self.withUnknownList[1]": 7,
				"self.withUnknownList[0] != self.withUnknownList[3]": 7,
				"self.withUnknownList[0] != self.withUnknownList[5]": 7,

				// if all fields are known, equality works as usual
				"self.withUnknownList[3] == self.withUnknownList[4]": 7,
				"self.withUnknownList[4] != self.withUnknownList[5]": 7,
			},
		},
		{name: "field nullability",
			obj: map[string]interface{}{
				"setPlainStr":          "v1",
				"setDefaultedStr":      "v2",
				"setNullableStr":       "v3",
				"setToNullNullableStr": nil,

				// we don't run the defaulter in this test suite (depending on it would introduce a cycle)
				// so we fake it :(
				"unsetDefaultedStr": "default value",
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"unsetPlainStr":     stringType,
				"unsetDefaultedStr": withDefault("default value", stringType),
				"unsetNullableStr":  withNullable(true, stringType),

				"setPlainStr":          stringType,
				"setDefaultedStr":      withDefault("default value", stringType),
				"setNullableStr":       withNullable(true, stringType),
				"setToNullNullableStr": withNullable(true, stringType),
			}),
			expectCost: map[string]int64{
				"!has(self.unsetPlainStr)": 2,
				"has(self.unsetDefaultedStr) && self.unsetDefaultedStr == 'default value'": 5,
				"!has(self.unsetNullableStr)": 2,

				"has(self.setPlainStr) && self.setPlainStr == 'v1'":         4,
				"has(self.setDefaultedStr) && self.setDefaultedStr == 'v2'": 4,
				"has(self.setNullableStr) && self.setNullableStr == 'v3'":   4,
				// We treat null fields as absent fields, not as null valued fields.
				// Note that this is different than how we treat nullable list items or map values.
				"type(self.setNullableStr) != null_type": 4,

				// a field that is set to null is treated the same as an absent field in validation rules
				"!has(self.setToNullNullableStr)": 2,
			},
		},
		{name: "null values in container types",
			obj: map[string]interface{}{
				"m": map[string]interface{}{
					"a": nil,
					"b": "not-nil",
				},
				"l": []interface{}{
					nil, "not-nil",
				},
				"s": []interface{}{
					nil, "not-nil",
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"m": mapType(withNullablePtr(true, stringType)),
				"l": listType(withNullablePtr(true, stringType)),
				"s": listSetType(withNullablePtr(true, stringType)),
			}),
			expectCost: map[string]int64{
				"self.m.size() == 2":             4,
				"'a' in self.m":                  3,
				"type(self.m['a']) == null_type": 5, // null check using runtime type checking
				//"self.m['a'] == null",
			},
		},
		{name: "object types are not accessible",
			obj: map[string]interface{}{
				"nestedInMap": map[string]interface{}{
					"k1": map[string]interface{}{
						"inMapField": 1,
					},
					"k2": map[string]interface{}{
						"inMapField": 2,
					},
				},
				"nestedInList": []interface{}{
					map[string]interface{}{
						"inListField": 1,
					},
					map[string]interface{}{
						"inListField": 2,
					},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"nestedInMap": mapType(objectTypePtr(map[string]schema.Structural{
					"inMapField": integerType,
				})),
				"nestedInList": listType(objectTypePtr(map[string]schema.Structural{
					"inListField": integerType,
				})),
			}),
			expectCost: map[string]int64{
				// we do not expose a stable type for the self variable, even when it is an object that CEL
				// considers a named type. The only operation developers should be able to perform on the type is
				// equality checking.
				"type(self) == type(self)":                                     5,
				"type(self.nestedInMap['k1']) == type(self.nestedInMap['k2'])": 9,
			},
		},
		{name: "listMaps with unsupported identity characters in property names",
			obj: map[string]interface{}{
				"objs": []interface{}{
					[]interface{}{
						map[string]interface{}{"k!": "a", "k.": "1"},
						map[string]interface{}{"k!": "b", "k.": "2"},
					},
					[]interface{}{
						map[string]interface{}{"k!": "b", "k.": "2"},
						map[string]interface{}{"k!": "a", "k.": "1"},
					},
					[]interface{}{
						map[string]interface{}{"k!": "b", "k.": "2"},
						map[string]interface{}{"k!": "c", "k.": "1"},
					},
					[]interface{}{
						map[string]interface{}{"k!": "b", "k.": "2"},
						map[string]interface{}{"k!": "a", "k.": "3"},
					},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"objs": listType(listMapTypePtr([]string{"k!", "k."}, objectTypePtr(map[string]schema.Structural{
					"k!": stringType,
					"k.": stringType,
				}))),
			}),
			expectCost: map[string]int64{
				"self.objs[0] == self.objs[1]":    7, // equal even though order is different
				"self.objs[0][0].k__dot__ == '1'": 6, // '.' is a supported character in identifiers, but it is escaped
			},
		},
		{name: "container type composition",
			obj: map[string]interface{}{
				"obj": map[string]interface{}{
					"field": "a",
				},
				"mapOfMap": map[string]interface{}{
					"x": map[string]interface{}{
						"y": "b",
					},
				},
				"mapOfObj": map[string]interface{}{
					"k": map[string]interface{}{
						"field2": "c",
					},
				},
				"mapOfListMap": map[string]interface{}{
					"o": []interface{}{
						map[string]interface{}{
							"k": "1",
							"v": "d",
						},
					},
				},
				"mapOfList": map[string]interface{}{
					"l": []interface{}{"e"},
				},
				"listMapOfObj": []interface{}{
					map[string]interface{}{
						"k2": "2",
						"v2": "f",
					},
				},
				"listOfMap": []interface{}{
					map[string]interface{}{
						"z": "g",
					},
				},
				"listOfObj": []interface{}{
					map[string]interface{}{
						"field3": "h",
					},
				},
				"listOfListMap": []interface{}{
					[]interface{}{
						map[string]interface{}{
							"k3": "3",
							"v3": "i",
						},
					},
				},
			},
			schema: objectTypePtr(map[string]schema.Structural{
				"obj": objectType(map[string]schema.Structural{
					"field": stringType,
				}),
				"mapOfMap": mapType(mapTypePtr(&stringType)),
				"mapOfObj": mapType(objectTypePtr(map[string]schema.Structural{
					"field2": stringType,
				})),
				"mapOfListMap": mapType(listMapTypePtr([]string{"k"}, objectTypePtr(map[string]schema.Structural{
					"k": stringType,
					"v": stringType,
				}))),
				"mapOfList": mapType(listTypePtr(&stringType)),
				"listMapOfObj": listMapType([]string{"k2"}, objectTypePtr(map[string]schema.Structural{
					"k2": stringType,
					"v2": stringType,
				})),
				"listOfMap": listType(mapTypePtr(&stringType)),
				"listOfObj": listType(objectTypePtr(map[string]schema.Structural{
					"field3": stringType,
				})),
				"listOfListMap": listType(listMapTypePtr([]string{"k3"}, objectTypePtr(map[string]schema.Structural{
					"k3": stringType,
					"v3": stringType,
				}))),
			}),
			expectCost: map[string]int64{
				"self.obj.field == 'a'":                                       4,
				"self.mapOfMap['x']['y'] == 'b'":                              5,
				"self.mapOfObj['k'].field2 == 'c'":                            5,
				"self.mapOfListMap['o'].exists(e, e.k == '1' && e.v == 'd')":  14,
				"self.mapOfList['l'][0] == 'e'":                               5,
				"self.listMapOfObj.exists(e, e.k2 == '2' && e.v2 == 'f')":     13,
				"self.listOfMap[0]['z'] == 'g'":                               5,
				"self.listOfObj[0].field3 == 'h'":                             5,
				"self.listOfListMap[0].exists(e, e.k3 == '3' && e.v3 == 'i')": 14,
			},
		},
	}

	for _, tt := range cases {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			for validRule, expectedCost := range tt.expectCost {
				testName := validRule
				if len(testName) > 127 {
					testName = testName[:127]
				}
				t.Run(testName, func(t *testing.T) {
					t.Parallel()
					s := withRule(*tt.schema, validRule)
					celValidator := NewValidator(&s, PerCallLimit)
					if celValidator == nil {
						t.Fatal("expected non nil validator")
					}
					ctx := context.TODO()
					errs, remainingBudegt := celValidator.Validate(ctx, field.NewPath("root"), &s, tt.obj, nil, RuntimeCELCostBudget)
					for _, err := range errs {
						t.Errorf("unexpected error: %v", err)
					}
					rtCost := RuntimeCELCostBudget - remainingBudegt
					if rtCost != expectedCost {
						t.Fatalf("runtime cost %d does not match expected runtime cost %d", rtCost, expectedCost)
					}
				})
			}
		})
	}
}
