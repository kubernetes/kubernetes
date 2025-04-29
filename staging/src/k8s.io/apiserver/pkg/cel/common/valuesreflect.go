/*
Copyright 2025 The Kubernetes Authors.

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

package common

import (
	"encoding/json"
	"fmt"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"reflect"
	"sigs.k8s.io/structured-merge-diff/v4/value"
	"sync"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
)

// TypedToVal wraps "typed" Go value as CEL ref.Val types using reflection.
// "typed" values must be values declared by Kubernetes API types.go definitions.
func TypedToVal(val interface{}, schema Schema) ref.Val {
	if val == nil {
		return types.NullValue
	}
	v := reflect.ValueOf(val)
	if !v.IsValid() {
		return types.NewErr("invalid data, got invalid reflect value: %v", v)
	}
	for v.Kind() == reflect.Pointer {
		if v.IsNil() {
			return types.NullValue
		}
		v = v.Elem()
	}
	val = v.Interface()

	switch typedVal := val.(type) {
	case bool:
		return types.Bool(typedVal)
	case int:
		return types.Int(typedVal)
	case int32:
		return types.Int(typedVal)
	case int64:
		return types.Int(typedVal)
	case float32:
		return types.Double(typedVal)
	case float64:
		return types.Double(typedVal)
	case string:
		return types.String(typedVal)
	case []byte:
		if typedVal == nil {
			return types.NullValue
		}
		return types.Bytes(typedVal)
	case metav1.Time:
		return types.Timestamp{Time: typedVal.Time}
	case metav1.MicroTime:
		return types.Timestamp{Time: typedVal.Time}
	case metav1.Duration:
		return types.Duration{Duration: typedVal.Duration}
	case intstr.IntOrString:
		switch typedVal.Type {
		case intstr.Int:
			return types.Int(typedVal.IntVal)
		case intstr.String:
			return types.String(typedVal.StrVal)
		}
	case resource.Quantity:
		// For compatibility with CRD Validation rules, represent quantity as a plain string.
		return types.String(typedVal.String())
	case json.Marshaler:
		// All JSON marshaled types must be mapped to a CEL type in the above switch.
		// This ensures that all types are purposefully mapped to CEL types.
		return types.NewErr("unsupported Go type for CEL: %T", typedVal)
	default:
		// continue on to the next switch
	}

	switch v.Kind() {
	case reflect.Slice:
		if schema.Items() == nil {
			return types.NewErr("invalid schema for slice type: %v", schema)
		}
		typedList := typedList{value: v, itemsSchema: schema.Items()}
		listType := schema.XListType()
		if listType != "" {
			switch listType {
			case "map":
				mapKeys := schema.XListMapKeys()
				return &typedMapList{typedList: typedList, escapedKeyProps: escapeKeyProps(mapKeys)}
			case "set":
				return &typedSetList{typedList: typedList}
			case "atomic":
				return &typedList
			default:
				return types.NewErr("invalid x-kubernetes-list-type, expected 'map', 'set' or 'atomic' but got %s", listType)
			}
		}
		return &typedList
	case reflect.Map:
		if schema.AdditionalProperties() == nil || schema.AdditionalProperties().Schema() == nil {
			return types.NewErr("invalid schema for map type: %v", schema)
		}
		return &typedMap{value: v, valuesSchema: schema.AdditionalProperties().Schema()}
	case reflect.Struct:
		if schema.Properties() == nil {
			return types.NewErr("invalid schema for struct type: %v", schema)
		}
		return &typedStruct{
			value: v,
			propSchema: func(key string) (Schema, bool) {
				if schema, ok := schema.Properties()[key]; ok {
					return schema, true
				}
				return nil, false
			},
		}
	// Match type aliases to primitives by kind
	case reflect.Bool:
		return types.Bool(v.Bool())
	case reflect.String:
		return types.String(v.String())
	case reflect.Int, reflect.Int32, reflect.Int64:
		return types.Int(v.Int())
	case reflect.Float32, reflect.Float64:
		return types.Double(v.Float())
	default:
		return types.NewErr("unsupported Go type for CEL: %v", v.Type())
	}
}

// typedStruct wraps a struct as a CEL ref.Val and provides lazy access to fields via reflection.
type typedStruct struct {
	value reflect.Value // Kind is required to be: reflect.Struct

	// propSchema finds the schema to use for a particular map key.
	propSchema func(key string) (Schema, bool)
}

func (s *typedStruct) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	if s.value.Type().AssignableTo(typeDesc) {
		return s.value.Interface(), nil
	}
	return nil, fmt.Errorf("type conversion error from struct type %v to %v", s.value.Type(), typeDesc)
}

func (s *typedStruct) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case s.Type():
		return s
	case types.MapType:
		return s
	case types.TypeType:
		return s.objType()
	}
	return types.NewErr("type conversion error from struct %s to %s", s.Type().TypeName(), typeValue.TypeName())
}

func (s *typedStruct) Equal(other ref.Val) ref.Val {
	otherStruct, ok := other.(*typedStruct)
	if ok {
		return types.Bool(apiequality.Semantic.DeepEqual(s.value.Interface(), otherStruct.value.Interface()))
	}
	return types.MaybeNoSuchOverloadErr(other)
}

func (s *typedStruct) Type() ref.Type {
	return s.objType()
}

func (s *typedStruct) objType() *types.Type {
	typeName := s.value.Type().Name()
	if pkgPath := s.value.Type().PkgPath(); pkgPath != "" {
		typeName = pkgPath + "." + typeName
	}
	return types.NewObjectType(typeName)
}

func (s *typedStruct) Value() interface{} {
	return s.value.Interface()
}

func (s *typedStruct) IsSet(field ref.Val) ref.Val {
	v, found := s.lookupField(field)
	if v != nil && types.IsUnknownOrError(v) {
		return v
	}
	return types.Bool(found)
}

func (s *typedStruct) Get(key ref.Val) ref.Val {
	v, found := s.lookupField(key)
	if !found {
		return types.NewErr("no such key: %v", key)
	}
	return v
}

func (s *typedStruct) lookupField(key ref.Val) (ref.Val, bool) {
	keyStr, ok := key.(types.String)
	if !ok {
		return types.MaybeNoSuchOverloadErr(key), true
	}
	fieldName := keyStr.Value().(string)

	cacheEntry := value.TypeReflectEntryOf(s.value.Type())
	fieldCache, ok := cacheEntry.Fields()[fieldName]
	if !ok {
		return nil, false
	}

	if e := fieldCache.GetFrom(s.value); !fieldCache.CanOmit(e) {
		if propSchema, ok := s.propSchema(fieldName); ok {
			v := TypedToVal(e.Interface(), propSchema)
			if v == types.NullValue {
				return nil, false
			}
			return v, true
		}
	}
	return nil, false
}

type typedList struct {
	value reflect.Value // Kind is required to be: reflect.Slice

	itemsSchema Schema
}

func (t *typedList) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	switch typeDesc.Kind() {
	case reflect.Slice:
		return t.value.Interface(), nil
	default:
		return nil, fmt.Errorf("type conversion error from '%s' to '%s'", t.Type(), typeDesc)
	}
}

func (t *typedList) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.ListType:
		return t
	case types.TypeType:
		return types.ListType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", t.Type(), typeValue.TypeName())
}

func (t *typedList) Equal(other ref.Val) ref.Val {
	oList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	sz := types.Int(t.value.Len())
	if sz != oList.Size() {
		return types.False
	}
	for i := types.Int(0); i < sz; i++ {
		eq := t.Get(i).Equal(oList.Get(i))
		if eq != types.True {
			return eq // either false or error
		}
	}
	return types.True
}

func (t *typedList) Type() ref.Type {
	return types.ListType
}

func (t *typedList) Value() interface{} {
	return t.value
}

func (t *typedList) Add(other ref.Val) ref.Val {
	oList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	resultValue := t.value
	for it := oList.Iterator(); it.HasNext() == types.True; {
		next := it.Next().Value()
		resultValue = reflect.Append(resultValue, reflect.ValueOf(next))
	}

	return &typedList{value: resultValue, itemsSchema: t.itemsSchema}
}

func (t *typedList) Contains(val ref.Val) ref.Val {
	if types.IsUnknownOrError(val) {
		return val
	}
	var err ref.Val
	sz := t.value.Len()
	for i := 0; i < sz; i++ {
		elem := TypedToVal(t.value.Index(i).Interface(), t.itemsSchema)
		cmp := elem.Equal(val)
		b, ok := cmp.(types.Bool)
		if !ok && err == nil {
			err = types.MaybeNoSuchOverloadErr(cmp)
		}
		if b == types.True {
			return types.True
		}
	}
	if err != nil {
		return err
	}
	return types.False
}

func (t *typedList) Get(idx ref.Val) ref.Val {
	iv, isInt := idx.(types.Int)
	if !isInt {
		return types.ValOrErr(idx, "unsupported index: %v", idx)
	}
	i := int(iv)
	if i < 0 || i >= t.value.Len() {
		return types.NewErr("index out of bounds: %v", idx)
	}
	return TypedToVal(t.value.Index(i).Interface(), t.itemsSchema)
}

func (t *typedList) Iterator() traits.Iterator {
	elements := make([]ref.Val, t.value.Len())
	sz := t.value.Len()
	for i := 0; i < sz; i++ {
		elements[i] = TypedToVal(t.value.Index(i).Interface(), t.itemsSchema)
	}
	return &sliceIter{typedList: t, elements: elements}
}

func (t *typedList) Size() ref.Val {
	return types.Int(t.value.Len())
}

type sliceIter struct {
	*typedList
	elements []ref.Val
	idx      int
}

func (it *sliceIter) HasNext() ref.Val {
	return types.Bool(it.idx < len(it.elements))
}

func (it *sliceIter) Next() ref.Val {
	if it.idx >= len(it.elements) {
		return types.NewErr("iterator exhausted")
	}
	elem := it.elements[it.idx]
	it.idx++
	return elem
}

type typedMapList struct {
	typedList
	escapedKeyProps []string

	sync.Once // for lazy load of mapOfList since it is only needed if Equals is called
	mapOfList map[interface{}]interface{}
}

func (t *typedMapList) getMap() map[interface{}]interface{} {
	t.Do(func() {
		sz := t.value.Len()
		t.mapOfList = make(map[interface{}]interface{}, sz)
		for i := types.Int(0); i < types.Int(sz); i++ {
			v := t.Get(i)
			e := reflect.ValueOf(v.Value())
			t.mapOfList[t.toMapKey(e)] = e.Interface()
		}
	})
	return t.mapOfList
}

// toMapKey returns a valid golang map key for the given element of the map list.
// element must be a valid map list entry where all map key props are scalar types (which are comparable in go
// and valid for use in a golang map key).
func (t *typedMapList) toMapKey(element reflect.Value) interface{} {
	if element.Kind() != reflect.Struct {
		return types.NewErr("unexpected data format for element of array with x-kubernetes-list-type=map: %T", element)
	}
	cacheEntry := value.TypeReflectEntryOf(element.Type())
	var fieldEntries []*value.FieldCacheEntry
	for i := 0; i < len(t.escapedKeyProps); i++ {
		if ce, ok := cacheEntry.Fields()[t.escapedKeyProps[i]]; !ok {
			return types.NewErr("unexpected data format for element of array with x-kubernetes-list-type=map: %T", element)
		} else {
			fieldEntries = append(fieldEntries, ce)
		}
	}

	// Arrays are comparable in go and may be used as map keys, but maps and slices are not.
	// So we can special case small numbers of key props as arrays and fall back to serialization
	// for larger numbers of key props
	if len(fieldEntries) == 1 {
		return fieldEntries[0].GetFrom(element).Interface()
	}
	if len(fieldEntries) == 2 {
		return [2]interface{}{fieldEntries[0].GetFrom(element).Interface(), fieldEntries[1].GetFrom(element).Interface()}
	}
	if len(fieldEntries) == 3 {
		return [3]interface{}{fieldEntries[0].GetFrom(element).Interface(), fieldEntries[1].GetFrom(element).Interface(), fieldEntries[3].GetFrom(element).Interface()}
	}

	key := make([]interface{}, len(fieldEntries))
	for i := range fieldEntries {
		key[i] = fieldEntries[i].GetFrom(element).Interface()
	}
	return fmt.Sprintf("%v", key)
}

// Equal on a map list ignores list element order.
func (t *typedMapList) Equal(other ref.Val) ref.Val {
	oMapList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	sz := types.Int(t.value.Len())
	if sz != oMapList.Size() {
		return types.False
	}
	tMap := t.getMap()
	for it := oMapList.Iterator(); it.HasNext() == types.True; {
		v := it.Next()
		k := t.toMapKey(reflect.ValueOf(v.Value()))
		tVal, ok := tMap[k]
		if !ok {
			return types.False
		}
		eq := TypedToVal(tVal, t.itemsSchema).Equal(v)
		if eq != types.True {
			return eq // either false or error
		}
	}
	return types.True
}

// Add for a map list `X + Y` performs a merge where the array positions of all keys in `X` are preserved but the values
// are overwritten by values in `Y` when the key sets of `X` and `Y` intersect. Elements in `Y` with
// non-intersecting keys are appended, retaining their partial order.
func (t *typedMapList) Add(other ref.Val) ref.Val {
	sliceType := t.value.Type()
	elementType := sliceType.Elem()
	oMapList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	sz := t.value.Len()
	elements := reflect.MakeSlice(sliceType, sz, sz)
	keyToIdx := map[interface{}]int{}
	for i := 0; i < sz; i++ {
		e := t.Get(types.Int(i)).Value()
		re := reflect.ValueOf(e)
		k := t.toMapKey(re)
		keyToIdx[k] = i
		elements.Index(i).Set(re.Convert(elementType))
	}
	for it := oMapList.Iterator(); it.HasNext() == types.True; {
		e := it.Next()
		re := reflect.ValueOf(e.Value())
		k := t.toMapKey(re)
		if overwritePosition, ok := keyToIdx[k]; ok {
			elements.Index(overwritePosition).Set(re)
		} else {
			elements = reflect.Append(elements, re.Convert(elementType))
		}
	}
	return &typedMapList{
		typedList:       typedList{value: elements, itemsSchema: t.itemsSchema},
		escapedKeyProps: t.escapedKeyProps,
	}
}

type typedSetList struct {
	typedList

	sync.Once // for lazy load of setOfList since it is only needed if Equals is called
	set       map[interface{}]struct{}
}

func (t *typedSetList) getSet() map[interface{}]struct{} {
	// sets are only allowed to contain scalar elements, which are comparable in go, and can safely be used as
	// golang map keys
	t.Do(func() {
		sz := t.value.Len()
		t.set = make(map[interface{}]struct{}, sz)
		for i := types.Int(0); i < types.Int(sz); i++ {
			e := t.Get(i).Value()
			t.set[e] = struct{}{}
		}
	})
	return t.set
}

// Equal on a map list ignores list element order.
func (t *typedSetList) Equal(other ref.Val) ref.Val {
	oSetList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	sz := types.Int(t.value.Len())
	if sz != oSetList.Size() {
		return types.False
	}
	tSet := t.getSet()
	for it := oSetList.Iterator(); it.HasNext() == types.True; {
		next := it.Next().Value()
		_, ok := tSet[next]
		if !ok {
			return types.False
		}
	}
	return types.True
}

// Add for a set list `X + Y` performs a union where the array positions of all elements in `X` are preserved and
// non-intersecting elements in `Y` are appended, retaining their partial order.
func (t *typedSetList) Add(other ref.Val) ref.Val {
	setType := t.value.Type()
	elementType := setType.Elem()
	oSetList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	sz := t.value.Len()
	elements := reflect.MakeSlice(setType, sz, sz)
	for i := 0; i < sz; i++ {
		e := t.Get(types.Int(i)).Value()
		re := reflect.ValueOf(e)
		elements.Index(i).Set(re.Convert(elementType))
	}
	set := t.getSet()
	for it := oSetList.Iterator(); it.HasNext() == types.True; {
		e := it.Next().Value()
		re := reflect.ValueOf(e)
		if _, ok := set[e]; !ok {
			set[e] = struct{}{}
			elements = reflect.Append(elements, re.Convert(elementType))
		}
	}
	return &typedSetList{
		typedList: typedList{value: elements, itemsSchema: t.itemsSchema},
	}
}

type typedMap struct {
	value reflect.Value // Kind is required to be: reflect.Map

	valuesSchema Schema
}

func (t *typedMap) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	switch typeDesc.Kind() {
	case reflect.Map:
		return t.value, nil
	default:
		return nil, fmt.Errorf("type conversion error from '%s' to '%s'", t.Type(), typeDesc)
	}
}

func (t *typedMap) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.MapType:
		return t
	case types.TypeType:
		return types.MapType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", t.Type(), typeValue.TypeName())
}

func (t *typedMap) Equal(other ref.Val) ref.Val {
	oMap, isMap := other.(traits.Mapper)
	if !isMap {
		return types.MaybeNoSuchOverloadErr(other)
	}
	if types.Int(t.value.Len()) != oMap.Size() {
		return types.False
	}
	for it := t.value.MapRange(); it.Next(); {
		key := it.Key()
		value := it.Value()
		ov, found := oMap.Find(types.String(key.String()))
		if !found {
			return types.False
		}
		v := TypedToVal(value.Interface(), t.valuesSchema)
		vEq := v.Equal(ov)
		if vEq != types.True {
			return vEq // either false or error
		}
	}
	return types.True
}

func (t *typedMap) Type() ref.Type {
	return types.MapType
}

func (t *typedMap) Value() interface{} {
	return t.value
}

func (t *typedMap) Contains(key ref.Val) ref.Val {
	v, found := t.Find(key)
	if v != nil && types.IsUnknownOrError(v) {
		return v
	}

	return types.Bool(found)
}

func (t *typedMap) Get(key ref.Val) ref.Val {
	v, found := t.Find(key)
	if found {
		return v
	}
	return types.ValOrErr(key, "no such key: %v", key)
}

func (t *typedMap) Size() ref.Val {
	return types.Int(t.value.Len())
}

func (t *typedMap) Find(key ref.Val) (ref.Val, bool) {
	keyStr, ok := key.(types.String)
	if !ok {
		return types.MaybeNoSuchOverloadErr(key), true
	}
	k := keyStr.Value().(string)
	if v := t.value.MapIndex(reflect.ValueOf(k)); v.IsValid() {
		return TypedToVal(v.Interface(), t.valuesSchema), true
	}
	return nil, false
}

func (t *typedMap) Iterator() traits.Iterator {
	keys := make([]ref.Val, t.value.Len())
	for i, k := range t.value.MapKeys() {
		keys[i] = types.String(k.String())
	}
	return &mapIter{typedMap: t, keys: keys}
}

type mapIter struct {
	*typedMap
	keys []ref.Val
	idx  int
}

func (it *mapIter) HasNext() ref.Val {
	return types.Bool(it.idx < len(it.keys))
}

func (it *mapIter) Next() ref.Val {
	key := it.keys[it.idx]
	it.idx++
	return key
}
