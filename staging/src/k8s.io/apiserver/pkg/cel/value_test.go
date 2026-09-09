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
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

func TestConvertToType(t *testing.T) {
	objType := NewObjectType("TestObject", map[string]*DeclField{})
	tests := []struct {
		val interface{}
		typ ref.Type
	}{
		{true, types.BoolType},
		{float64(1.2), types.DoubleType},
		{int64(-42), types.IntType},
		{uint64(63), types.UintType},
		{time.Duration(300), types.DurationType},
		{time.Now().UTC(), types.TimestampType},
		{types.NullValue, types.NullType},
		{NewListValue(), types.ListType},
		{NewMapValue(), types.MapType},
		{[]byte("bytes"), types.BytesType},
		{NewObjectValue(objType), objType},
	}
	for i, tc := range tests {
		idx := i
		tst := tc
		t.Run(fmt.Sprintf("[%d]", i), func(t *testing.T) {
			dv := testValue(t, int64(idx), tst.val)
			ev := dv.ExprValue()
			if ev.ConvertToType(types.TypeType).(ref.Type).TypeName() != tst.typ.TypeName() {
				t.Errorf("got %v, wanted %v type", ev.ConvertToType(types.TypeType), tst.typ)
			}
			if ev.ConvertToType(tst.typ).Equal(ev) != types.True {
				t.Errorf("got %v, wanted input value %v", ev.ConvertToType(tst.typ), ev)
			}
		})
	}
}

func TestEqual(t *testing.T) {
	vals := []interface{}{
		true, []byte("bytes"), float64(1.2), int64(-42), uint64(63), time.Duration(300),
		time.Now().UTC(), types.NullValue, NewListValue(), NewMapValue(),
		NewObjectValue(NewObjectType("TestObject", map[string]*DeclField{})),
	}
	for i, v := range vals {
		dv := testValue(t, int64(i), v)
		if dv.Equal(dv.ExprValue()) != types.True {
			t.Errorf("got %v, wanted dyn value %v equal to itself", dv.Equal(dv.ExprValue()), dv.ExprValue())
		}
	}
}

func TestListValueAdd(t *testing.T) {
	lv := NewListValue()
	lv.Append(testValue(t, 1, "first"))
	ov := NewListValue()
	ov.Append(testValue(t, 2, "second"))
	ov.Append(testValue(t, 3, "third"))
	llv := NewListValue()
	llv.Append(testValue(t, 4, lv))
	lov := NewListValue()
	lov.Append(testValue(t, 5, ov))
	var v traits.Lister = llv.Add(lov).(traits.Lister)
	if v.Size() != types.Int(2) {
		t.Errorf("got list size %d, wanted 2", v.Size())
	}
	complex, err := v.ConvertToNative(reflect.TypeOf([][]string{}))
	complexList := complex.([][]string)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(complexList, [][]string{{"first"}, {"second", "third"}}) {
		t.Errorf("got %v, wanted [['first'], ['second', 'third']]", complexList)
	}
}

func TestListValueContains(t *testing.T) {
	lv := NewListValue()
	lv.Append(testValue(t, 1, "first"))
	lv.Append(testValue(t, 2, "second"))
	lv.Append(testValue(t, 3, "third"))
	for i := types.Int(0); i < lv.Size().(types.Int); i++ {
		e := lv.Get(i)
		contained := lv.Contains(e)
		if contained != types.True {
			t.Errorf("got %v, wanted list contains elem[%v] %v == true", contained, i, e)
		}
	}
	if lv.Contains(types.String("fourth")) != types.False {
		t.Errorf("got %v, wanted false 'fourth'", lv.Contains(types.String("fourth")))
	}
	if !types.IsError(lv.Contains(types.Int(-1))) {
		t.Errorf("got %v, wanted error for invalid type", lv.Contains(types.Int(-1)))
	}
}

func TestListValueContainsNestedList(t *testing.T) {
	lvA := NewListValue()
	lvA.Append(testValue(t, 1, int64(1)))
	lvA.Append(testValue(t, 2, int64(2)))

	lvB := NewListValue()
	lvB.Append(testValue(t, 3, int64(3)))

	elemA, elemB := testValue(t, 4, lvA), testValue(t, 5, lvB)
	lv := NewListValue()
	lv.Append(elemA)
	lv.Append(elemB)

	contained := lv.Contains(elemA.ExprValue())
	if contained != types.True {
		t.Errorf("got %v, wanted elemA contained in list value", contained)
	}
	contained = lv.Contains(elemB.ExprValue())
	if contained != types.True {
		t.Errorf("got %v, wanted elemB contained in list value", contained)
	}
	contained = lv.Contains(types.DefaultTypeAdapter.NativeToValue([]int32{4}))
	if contained != types.False {
		t.Errorf("got %v, wanted empty list not contained", contained)
	}
}

func TestListValueConvertToNative(t *testing.T) {
	lv := NewListValue()
	none, err := lv.ConvertToNative(reflect.TypeOf([]interface{}{}))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(none, []interface{}{}) {
		t.Errorf("got %v, wanted empty list", none)
	}
	lv.Append(testValue(t, 1, "first"))
	one, err := lv.ConvertToNative(reflect.TypeOf([]string{}))
	oneList := one.([]string)
	if err != nil {
		t.Fatal(err)
	}
	if len(oneList) != 1 {
		t.Errorf("got len(one) == %d, wanted 1", len(oneList))
	}
	if !reflect.DeepEqual(oneList, []string{"first"}) {
		t.Errorf("got %v, wanted string list", oneList)
	}
	ov := NewListValue()
	ov.Append(testValue(t, 2, "second"))
	ov.Append(testValue(t, 3, "third"))
	if ov.Size() != types.Int(2) {
		t.Errorf("got list size %d, wanted 2", ov.Size())
	}
	llv := NewListValue()
	llv.Append(testValue(t, 4, lv))
	llv.Append(testValue(t, 5, ov))
	if llv.Size() != types.Int(2) {
		t.Errorf("got list size %d, wanted 2", llv.Size())
	}
	complex, err := llv.ConvertToNative(reflect.TypeOf([][]string{}))
	complexList := complex.([][]string)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(complexList, [][]string{{"first"}, {"second", "third"}}) {
		t.Errorf("got %v, wanted [['first'], ['second', 'third']]", complexList)
	}
}

func TestListValueIterator(t *testing.T) {
	lv := NewListValue()
	lv.Append(testValue(t, 1, "first"))
	lv.Append(testValue(t, 2, "second"))
	lv.Append(testValue(t, 3, "third"))
	it := lv.Iterator()
	if it.Type() != types.IteratorType {
		t.Errorf("got type %v for iterator, wanted IteratorType", it.Type())
	}
	i := types.Int(0)
	for it.HasNext() == types.True {
		v := it.Next()
		if v.Equal(lv.Get(i)) != types.True {
			t.Errorf("iterator value %v and value %v at index %d not equal", v, lv.Get(i), i)
		}
		i++
	}
}

func TestMapValueConvertToNative(t *testing.T) {
	mv := NewMapValue()
	none, err := mv.ConvertToNative(reflect.TypeOf(map[string]interface{}{}))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(none, map[string]interface{}{}) {
		t.Errorf("got %v, wanted empty map", none)
	}
	none, err = mv.ConvertToNative(reflect.TypeOf(map[interface{}]interface{}{}))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(none, map[interface{}]interface{}{}) {
		t.Errorf("got %v, wanted empty map", none)
	}
	mv.AddField(NewField(1, "Test"))
	tst, _ := mv.GetField("Test")
	tst.Ref = testValue(t, 2, uint64(12))
	mv.AddField(NewField(3, "Check"))
	chk, _ := mv.GetField("Check")
	chk.Ref = testValue(t, 4, uint64(34))
	if mv.Size() != types.Int(2) {
		t.Errorf("got size %d, wanted 2", mv.Size())
	}
	if mv.Contains(types.String("Test")) != types.True {
		t.Error("key 'Test' not found")
	}
	if mv.Contains(types.String("Check")) != types.True {
		t.Error("key 'Check' not found")
	}
	if mv.Contains(types.String("Checked")) != types.False {
		t.Error("key 'Checked' found, wanted not found")
	}
	it := mv.Iterator()
	for it.HasNext() == types.True {
		k := it.Next()
		v := mv.Get(k)
		if k == types.String("Test") && v != types.Uint(12) {
			t.Errorf("key 'Test' not equal to 12u")
		}
		if k == types.String("Check") && v != types.Uint(34) {
			t.Errorf("key 'Check' not equal to 34u")
		}
	}
	mpStrUint, err := mv.ConvertToNative(reflect.TypeOf(map[string]uint64{}))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mpStrUint, map[string]uint64{
		"Test":  uint64(12),
		"Check": uint64(34),
	}) {
		t.Errorf("got %v, wanted {'Test': 12u, 'Check': 34u}", mpStrUint)
	}
	tstStr, err := mv.ConvertToNative(reflect.TypeOf(&tstStruct{}))
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(tstStr, &tstStruct{
		Test:  uint64(12),
		Check: uint64(34),
	}) {
		t.Errorf("got %v, wanted tstStruct{Test: 12u, Check: 34u}", tstStr)
	}
}

func TestMapValueEqual(t *testing.T) {
	mv := NewMapValue()
	name := NewField(1, "name")
	name.Ref = testValue(t, 2, "alert")
	priority := NewField(3, "priority")
	priority.Ref = testValue(t, 4, int64(4))
	mv.AddField(name)
	mv.AddField(priority)
	if mv.Equal(mv) != types.True {
		t.Fatalf("map.Equal(map) failed: %v", mv.Equal(mv))
	}
}

func TestMapValueNotEqual(t *testing.T) {
	mv := NewMapValue()
	name := NewField(1, "name")
	name.Ref = testValue(t, 2, "alert")
	priority := NewField(3, "priority")
	priority.Ref = testValue(t, 4, int64(4))
	mv.AddField(name)
	mv.AddField(priority)

	mv2 := NewMapValue()
	mv2.AddField(name)
	if mv.Equal(mv2) != types.False {
		t.Fatalf("mv.Equal(mv2) failed: %v", mv.Equal(mv2))
	}

	priority2 := NewField(5, "priority")
	priority2.Ref = testValue(t, 6, int64(3))
	mv2.AddField(priority2)
	if mv.Equal(mv2) != types.False {
		t.Fatalf("mv.Equal(mv2) failed: %v", mv.Equal(mv2))
	}
}

func TestMapValueIsSet(t *testing.T) {
	mv := NewMapValue()
	if mv.IsSet(types.String("name")) != types.False {
		t.Error("map.IsSet('name') returned true for unset key")
	}
	mv.AddField(NewField(1, "name"))
	if mv.IsSet(types.String("name")) != types.True {
		t.Error("map.IsSet('name') returned false for a set key")
	}
}

func TestObjectValueEqual(t *testing.T) {
	objType := NewObjectType("Notice", map[string]*DeclField{
		"name":     {Name: "name", Type: StringType},
		"priority": {Name: "priority", Type: IntType},
		"message":  {Name: "message", Type: StringType, defaultValue: "<eom>"},
	})
	name := NewField(1, "name")
	name.Ref = testValue(t, 2, "alert")
	priority := NewField(3, "priority")
	priority.Ref = testValue(t, 4, int64(4))
	message := NewField(5, "message")
	message.Ref = testValue(t, 6, "call immediately")

	mv1 := NewMapValue()
	mv1.AddField(name)
	mv1.AddField(priority)
	obj1 := mv1.ConvertToObject(objType)
	if obj1.Equal(obj1) != types.True {
		t.Errorf("obj1.Equal(obj1) failed, got: %v", obj1.Equal(obj1))
	}

	mv2 := NewMapValue()
	mv2.AddField(name)
	mv2.AddField(priority)
	mv2.AddField(message)
	obj2 := mv2.ConvertToObject(objType)
	if obj1.Equal(obj2) == types.True {
		t.Error("obj1.Equal(obj2) returned true, wanted false")
	}
	if obj2.Equal(obj1) == types.True {
		t.Error("obj2.Equal(obj1) returned true, wanted false")
	}
}

type tstStruct struct {
	Test  uint64
	Check uint64
}
