/*
Copyright 2019 The Kubernetes Authors.

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

package value

import (
	"testing"
)

func TestValueLess(t *testing.T) {
	table := []struct {
		name string
		// we expect a < b and !(b < a) unless eq is true, in which
		// case we expect less to return false in both orders.
		a, b Value
		eq   bool
	}{
		{
			name: "Invalid-1",
			a:    Value{},
			b:    Value{},
			eq:   true,
		}, {
			name: "Invalid-2",
			a:    FloatValue(1),
			b:    Value{},
		}, {
			name: "Invalid-3",
			a:    IntValue(1),
			b:    Value{},
		}, {
			name: "Invalid-4",
			a:    StringValue("aoeu"),
			b:    Value{},
		}, {
			name: "Invalid-5",
			a:    BooleanValue(true),
			b:    Value{},
		}, {
			name: "Invalid-6",
			a:    Value{ListValue: &List{}},
			b:    Value{},
		}, {
			name: "Invalid-7",
			a:    Value{MapValue: &Map{}},
			b:    Value{},
		}, {
			name: "Invalid-8",
			a:    Value{Null: true},
			b:    Value{},
		}, {
			name: "Float-1",
			a:    FloatValue(1.14),
			b:    FloatValue(3.14),
		}, {
			name: "Float-2",
			a:    FloatValue(1),
			b:    FloatValue(1),
			eq:   true,
		}, {
			name: "Float-3",
			a:    FloatValue(1),
			b:    IntValue(1),
			eq:   true,
		}, {
			name: "Float-4",
			a:    FloatValue(1),
			b:    IntValue(2),
		}, {
			name: "Float-5",
			a:    FloatValue(1),
			b:    StringValue("aoeu"),
		}, {
			name: "Float-6",
			a:    FloatValue(1),
			b:    BooleanValue(true),
		}, {
			name: "Float-7",
			a:    FloatValue(1),
			b:    Value{ListValue: &List{}},
		}, {
			name: "Float-8",
			a:    FloatValue(1),
			b:    Value{MapValue: &Map{}},
		}, {
			name: "Float-9",
			a:    FloatValue(1),
			b:    Value{Null: true},
		}, {
			name: "Int-1",
			a:    IntValue(1),
			b:    IntValue(2),
		}, {
			name: "Int-2",
			a:    IntValue(1),
			b:    IntValue(1),
			eq:   true,
		}, {
			name: "Int-3",
			a:    IntValue(1),
			b:    FloatValue(1),
			eq:   true,
		}, {
			name: "Int-4",
			a:    IntValue(1),
			b:    FloatValue(2),
		}, {
			name: "Int-5",
			a:    IntValue(1),
			b:    StringValue("aoeu"),
		}, {
			name: "Int-6",
			a:    IntValue(1),
			b:    BooleanValue(true),
		}, {
			name: "Int-7",
			a:    IntValue(1),
			b:    Value{ListValue: &List{}},
		}, {
			name: "Int-8",
			a:    IntValue(1),
			b:    Value{MapValue: &Map{}},
		}, {
			name: "Int-9",
			a:    IntValue(1),
			b:    Value{Null: true},
		}, {
			name: "String-1",
			a:    StringValue("b-12"),
			b:    StringValue("b-9"),
		}, {
			name: "String-2",
			a:    StringValue("folate"),
			b:    StringValue("folate"),
			eq:   true,
		}, {
			name: "String-3",
			a:    StringValue("folate"),
			b:    BooleanValue(true),
		}, {
			name: "String-4",
			a:    StringValue("folate"),
			b:    Value{ListValue: &List{}},
		}, {
			name: "String-5",
			a:    StringValue("folate"),
			b:    Value{MapValue: &Map{}},
		}, {
			name: "String-6",
			a:    StringValue("folate"),
			b:    Value{Null: true},
		}, {
			name: "Bool-1",
			a:    BooleanValue(false),
			b:    BooleanValue(true),
		}, {
			name: "Bool-2",
			a:    BooleanValue(false),
			b:    BooleanValue(false),
			eq:   true,
		}, {
			name: "Bool-3",
			a:    BooleanValue(true),
			b:    BooleanValue(true),
			eq:   true,
		}, {
			name: "Bool-4",
			a:    BooleanValue(false),
			b:    Value{ListValue: &List{}},
		}, {
			name: "Bool-5",
			a:    BooleanValue(false),
			b:    Value{MapValue: &Map{}},
		}, {
			name: "Bool-6",
			a:    BooleanValue(false),
			b:    Value{Null: true},
		}, {
			name: "List-1",
			a:    Value{ListValue: &List{}},
			b:    Value{ListValue: &List{}},
			eq:   true,
		}, {
			name: "List-2",
			a:    Value{ListValue: &List{Items: []Value{IntValue(1)}}},
			b:    Value{ListValue: &List{Items: []Value{IntValue(1)}}},
			eq:   true,
		}, {
			name: "List-3",
			a:    Value{ListValue: &List{Items: []Value{IntValue(1)}}},
			b:    Value{ListValue: &List{Items: []Value{IntValue(2)}}},
		}, {
			name: "List-4",
			a:    Value{ListValue: &List{Items: []Value{IntValue(1)}}},
			b:    Value{ListValue: &List{Items: []Value{IntValue(1), IntValue(1)}}},
		}, {
			name: "List-5",
			a:    Value{ListValue: &List{Items: []Value{IntValue(1), IntValue(1)}}},
			b:    Value{ListValue: &List{Items: []Value{IntValue(2)}}},
		}, {
			name: "List-6",
			a:    Value{ListValue: &List{}},
			b:    Value{MapValue: &Map{}},
		}, {
			name: "List-7",
			a:    Value{ListValue: &List{}},
			b:    Value{Null: true},
		}, {
			name: "Map-1",
			a:    Value{MapValue: &Map{}},
			b:    Value{MapValue: &Map{}},
			eq:   true,
		}, {
			name: "Map-2",
			a:    Value{MapValue: &Map{Items: []Field{{Name: "carotine", Value: IntValue(1)}}}},
			b:    Value{MapValue: &Map{Items: []Field{{Name: "carotine", Value: IntValue(1)}}}},
			eq:   true,
		}, {
			name: "Map-3",
			a:    Value{MapValue: &Map{Items: []Field{{Name: "carotine", Value: IntValue(1)}}}},
			b:    Value{MapValue: &Map{Items: []Field{{Name: "carotine", Value: IntValue(2)}}}},
		}, {
			name: "Map-4",
			a:    Value{MapValue: &Map{Items: []Field{{Name: "carotine", Value: IntValue(1)}}}},
			b:    Value{MapValue: &Map{Items: []Field{{Name: "ethanol", Value: IntValue(1)}}}},
		}, {
			name: "Map-5",
			a: Value{MapValue: &Map{Items: []Field{
				{Name: "carotine", Value: IntValue(1)},
				{Name: "ethanol", Value: IntValue(1)},
			}}},
			b: Value{MapValue: &Map{Items: []Field{
				{Name: "ethanol", Value: IntValue(1)},
				{Name: "carotine", Value: IntValue(1)},
			}}},
			eq: true,
		}, {
			name: "Map-6",
			a: Value{MapValue: &Map{Items: []Field{
				{Name: "carotine", Value: IntValue(1)},
				{Name: "ethanol", Value: IntValue(1)},
			}}},
			b: Value{MapValue: &Map{Items: []Field{
				{Name: "ethanol", Value: IntValue(1)},
				{Name: "carotine", Value: IntValue(2)},
			}}},
		}, {
			name: "Map-7",
			a: Value{MapValue: &Map{Items: []Field{
				{Name: "carotine", Value: IntValue(1)},
			}}},
			b: Value{MapValue: &Map{Items: []Field{
				{Name: "ethanol", Value: IntValue(1)},
				{Name: "carotine", Value: IntValue(2)},
			}}},
		}, {
			name: "Map-8",
			a: Value{MapValue: &Map{Items: []Field{
				{Name: "carotine", Value: IntValue(1)},
			}}},
			b: Value{MapValue: &Map{Items: []Field{
				{Name: "ethanol", Value: IntValue(1)},
				{Name: "carotine", Value: IntValue(1)},
			}}},
		}, {
			name: "Map-9",
			a: Value{MapValue: &Map{Items: []Field{
				{Name: "carotine", Value: IntValue(1)},
				{Name: "ethanol", Value: IntValue(1)},
			}}},
			b: Value{MapValue: &Map{Items: []Field{
				{Name: "carotine", Value: IntValue(2)},
			}}},
		}, {
			name: "Map-8",
			a:    Value{MapValue: &Map{}},
			b:    Value{Null: true},
		},
	}

	for i := range table {
		i := i
		t.Run(table[i].name, func(t *testing.T) {
			tt := table[i]
			if tt.eq {
				if !tt.a.Equals(tt.b) {
					t.Errorf("oops, a != b: %#v, %#v", tt.a, tt.b)
				}
				if tt.a.Less(tt.b) {
					t.Errorf("oops, a < b: %#v, %#v", tt.a, tt.b)
				}
			} else {
				if !tt.a.Less(tt.b) {
					t.Errorf("oops, a >= b: %#v, %#v", tt.a, tt.b)
				}
			}
			if tt.b.Less(tt.a) {
				t.Errorf("oops, b < a: %#v, %#v", tt.b, tt.a)
			}

			if tt.eq {
				if tt.a.Compare(tt.b) != 0 || tt.b.Compare(tt.b) != 0 {
					t.Errorf("oops, a != b: %#v, %#v", tt.a, tt.b)
				}
			} else {
				if !(tt.a.Compare(tt.b) < 0) {
					t.Errorf("oops, a is not less than b: %#v, %#v", tt.a, tt.b)
				}
				if !(tt.b.Compare(tt.a) > 0) {
					t.Errorf("oops, b is not more than a: %#v, %#v", tt.a, tt.b)
				}
			}
		})
	}

}
