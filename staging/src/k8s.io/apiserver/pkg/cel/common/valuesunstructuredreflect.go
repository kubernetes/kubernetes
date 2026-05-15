/*
Copyright The Kubernetes Authors.

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
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/structured-merge-diff/v6/value"
)

type unstructuredWrapper interface {
	ToUnstructured() interface{}
}

// UnstructuredReflectToVal wraps a Go value as a CEL ref.Val.
// It mimics how the value would look if it were converted to an unstructured
// map using JSON serialization, but avoids the full allocation by lazily
// traversing the value via reflection and structured-merge-diff value package.
func UnstructuredReflectToVal(val interface{}) ref.Val {
	if val == nil {
		return types.NullValue
	}

	// Fast-path: immediately handle primitive types to completely bypass reflect.ValueOf overhead.
	switch typedVal := val.(type) {
	case bool:
		return types.Bool(typedVal)
	case string:
		return types.String(typedVal)
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
	}

	v := reflect.ValueOf(val)
	if !v.IsValid() {
		return types.NewErr("invalid data, got invalid reflect value: %v", v)
	}

	switch v.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Slice:
		if v.IsNil() {
			return types.NullValue
		}
	}

	// Handle special wrapper types like wrappedParam that provide a custom ToUnstructured() method.
	if unstrObj, ok := val.(unstructuredWrapper); ok {
		return UnstructuredReflectToVal(unstrObj.ToUnstructured())
	}
	// Custom JSON marshaling logic must be run to ensure parity with standard unstructured conversion
	// for types implementing custom marshaling (e.g., runtime.RawExtension).
	// Well-known Kubernetes type intstr.IntOrString is skipped here because its JSON
	// marshaling would produce a numeric JSON value that gets unmarshaled to float64,
	// incorrectly turning an integer IntOrString into a CEL Double instead of a CEL Int.
	// We skip it here to handle it explicitly downstream and preserve correct types.
	if marshaler, ok := val.(json.Marshaler); ok {
		switch val.(type) {
		case intstr.IntOrString:
		default:
			return marshalToVal(marshaler)
		}
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return types.NullValue
		}
		return UnstructuredReflectToVal(v.Elem().Interface())
	case reflect.Slice:
		if v.Type().Elem().Kind() == reflect.Uint8 {
			if v.IsNil() {
				return types.NullValue
			}
			// In unstructured conversion, []byte is base64 encoded. We directly use base64.StdEncoding
			// to avoid the high overhead of json serialization/deserialization.
			return types.String(base64.StdEncoding.EncodeToString(v.Bytes()))
		}
		return &reflectUnstructuredList{value: v}
	case reflect.Map:
		return &reflectUnstructuredMap{value: v}
	case reflect.Struct:
		if typedVal, ok := val.(intstr.IntOrString); ok {
			switch typedVal.Type {
			case intstr.Int:
				return types.Int(typedVal.IntVal)
			case intstr.String:
				return types.String(typedVal.StrVal)
			}
		}
		// Check if pointer to struct implements ToUnstructured
		ptrType := reflect.PointerTo(v.Type())
		if ptrType.Implements(reflect.TypeFor[unstructuredWrapper]()) {
			ptr := reflect.New(v.Type())
			ptr.Elem().Set(v)
			return UnstructuredReflectToVal(ptr.Interface().(unstructuredWrapper).ToUnstructured())
		}
		// Check if pointer to struct implements json.Marshaler
		if ptrType.Implements(reflect.TypeFor[json.Marshaler]()) {
			ptr := reflect.New(v.Type())
			ptr.Elem().Set(v)
			return marshalToVal(ptr.Interface().(json.Marshaler))
		}
		return &reflectUnstructuredStruct{value: v}
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

func marshalToVal(marshaler json.Marshaler) ref.Val {
	b, err := json.Marshal(marshaler)
	if err != nil {
		return types.NewErr("failed to marshal type %T: %v", marshaler, err)
	}
	var unmarshaled interface{}
	if err := json.Unmarshal(b, &unmarshaled); err != nil {
		return types.NewErr("failed to unmarshal type %T: %v", marshaler, err)
	}
	return types.DefaultTypeAdapter.NativeToValue(unmarshaled)
}

type reflectUnstructuredList struct {
	value reflect.Value
}

func (l *reflectUnstructuredList) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	switch typeDesc.Kind() {
	case reflect.Slice:
		return l.value.Interface(), nil
	default:
		return nil, fmt.Errorf("type conversion error from '%s' to '%s'", l.Type(), typeDesc)
	}
}

func (l *reflectUnstructuredList) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.ListType:
		return l
	case types.TypeType:
		return types.ListType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", l.Type(), typeValue.TypeName())
}

func (l *reflectUnstructuredList) Equal(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	sz := l.Size().(types.Int)
	otherSz := otherList.Size().(types.Int)
	if sz != otherSz {
		return types.False
	}
	for i := range sz {
		v1 := l.Get(i)
		v2 := otherList.Get(i)
		eq := v1.Equal(v2)
		if eq != types.True {
			return types.False
		}
	}
	return types.True
}

func (l *reflectUnstructuredList) Type() ref.Type {
	return types.ListType
}

func (l *reflectUnstructuredList) Value() interface{} {
	return l.value.Interface()
}

func (l *reflectUnstructuredList) Contains(val ref.Val) ref.Val {
	sz := l.Size().(types.Int)
	for i := range sz {
		v := l.Get(i)
		eq := v.Equal(val)
		if eq == types.True {
			return types.True
		}
	}
	return types.False
}

func (l *reflectUnstructuredList) Get(index ref.Val) ref.Val {
	idx, ok := index.(types.Int)
	if !ok {
		return types.ValOrErr(index, "unsupported index type: %T", index)
	}
	if idx < 0 || int(idx) >= l.value.Len() {
		return types.NewErr("index out of bounds: %d", idx)
	}
	return UnstructuredReflectToVal(l.value.Index(int(idx)).Interface())
}

func (l *reflectUnstructuredList) Iterator() traits.Iterator {
	return &reflectUnstructuredListIterator{reflectUnstructuredList: l, idx: 0}
}

func (l *reflectUnstructuredList) Size() ref.Val {
	return types.Int(l.value.Len())
}

type reflectUnstructuredListIterator struct {
	*reflectUnstructuredList
	idx int
}

func (it *reflectUnstructuredListIterator) HasNext() ref.Val {
	return types.Bool(it.idx < it.reflectUnstructuredList.value.Len())
}

func (it *reflectUnstructuredListIterator) Next() ref.Val {
	if it.idx >= it.reflectUnstructuredList.value.Len() {
		return types.NewErr("no more elements")
	}
	val := it.reflectUnstructuredList.Get(types.Int(it.idx))
	it.idx++
	return val
}

type reflectUnstructuredMap struct {
	value reflect.Value
}

func (m *reflectUnstructuredMap) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	if m.value.Type().AssignableTo(typeDesc) {
		return m.value.Interface(), nil
	}
	return nil, fmt.Errorf("type conversion error from '%s' to '%s'", m.Type(), typeDesc)
}

func (m *reflectUnstructuredMap) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.MapType:
		return m
	case types.TypeType:
		return types.MapType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", m.Type(), typeValue.TypeName())
}

func (m *reflectUnstructuredMap) Equal(other ref.Val) ref.Val {
	otherMap, ok := other.(traits.Mapper)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	if m.Size() != otherMap.Size() {
		return types.False
	}
	it := m.Iterator()
	for it.HasNext() == types.True {
		key := it.Next()
		v1 := m.Get(key)
		v2 := otherMap.Get(key)
		eq := v1.Equal(v2)
		if eq != types.True {
			return types.False
		}
	}
	return types.True
}

func (m *reflectUnstructuredMap) Type() ref.Type {
	return types.MapType
}

func (m *reflectUnstructuredMap) Value() interface{} {
	return m.value.Interface()
}

func (m *reflectUnstructuredMap) Contains(key ref.Val) ref.Val {
	keyVal, ok := key.(types.String)
	if !ok {
		return types.ValOrErr(key, "unsupported map key type: %T", key)
	}
	mapKeys := m.value.MapKeys()
	for _, mk := range mapKeys {
		if mk.String() == string(keyVal) {
			return types.True
		}
	}
	return types.False
}

func (m *reflectUnstructuredMap) Find(key ref.Val) (ref.Val, bool) {
	keyVal, ok := key.(types.String)
	if !ok {
		return types.ValOrErr(key, "unsupported map key type: %T", key), false
	}
	keyType := m.value.Type().Key()
	if keyType.Kind() != reflect.String {
		return nil, false
	}
	reflectKey := reflect.ValueOf(string(keyVal)).Convert(keyType)
	val := m.value.MapIndex(reflectKey)
	if !val.IsValid() {
		return nil, false
	}
	return UnstructuredReflectToVal(val.Interface()), true
}

func (m *reflectUnstructuredMap) Get(key ref.Val) ref.Val {
	keyVal, ok := key.(types.String)
	if !ok {
		return types.ValOrErr(key, "unsupported map key type: %T", key)
	}
	keyType := m.value.Type().Key()
	if keyType.Kind() != reflect.String {
		return types.ValOrErr(key, "no such key: %v", keyVal)
	}
	reflectKey := reflect.ValueOf(string(keyVal)).Convert(keyType)
	val := m.value.MapIndex(reflectKey)
	if !val.IsValid() {
		return types.ValOrErr(key, "no such key: %v", keyVal)
	}
	return UnstructuredReflectToVal(val.Interface())
}

func (m *reflectUnstructuredMap) Iterator() traits.Iterator {
	return &reflectUnstructuredMapIterator{reflectUnstructuredMap: m, keys: m.value.MapKeys(), idx: 0}
}

func (m *reflectUnstructuredMap) Size() ref.Val {
	return types.Int(m.value.Len())
}

type reflectUnstructuredMapIterator struct {
	*reflectUnstructuredMap
	keys []reflect.Value
	idx  int
}

func (it *reflectUnstructuredMapIterator) HasNext() ref.Val {
	return types.Bool(it.idx < len(it.keys))
}

func (it *reflectUnstructuredMapIterator) Next() ref.Val {
	if it.idx >= len(it.keys) {
		return types.NewErr("no more elements")
	}
	val := types.String(it.keys[it.idx].String())
	it.idx++
	return val
}

type reflectUnstructuredStruct struct {
	value reflect.Value
}

func (s *reflectUnstructuredStruct) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	if s.value.Type().AssignableTo(typeDesc) {
		return s.value.Interface(), nil
	}
	return nil, fmt.Errorf("type conversion error from struct type %v to %v", s.value.Type(), typeDesc)
}

func (s *reflectUnstructuredStruct) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.MapType:
		return s
	case types.TypeType:
		return types.MapType
	}
	return types.NewErr("type conversion error from struct to %s", typeValue.TypeName())
}

func (s *reflectUnstructuredStruct) Equal(other ref.Val) ref.Val {
	otherStruct, ok := other.(*reflectUnstructuredStruct)
	if ok {
		return types.Bool(apiequality.Semantic.DeepEqual(s.value.Interface(), otherStruct.value.Interface()))
	}
	otherMap, ok := other.(traits.Mapper)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	if s.Size() != otherMap.Size() {
		return types.False
	}
	it := s.Iterator()
	for it.HasNext() == types.True {
		key := it.Next()
		v1 := s.Get(key)
		v2 := otherMap.Get(key)
		eq := v1.Equal(v2)
		if eq != types.True {
			return types.False
		}
	}
	return types.True
}

func (s *reflectUnstructuredStruct) Type() ref.Type {
	return types.MapType // Unstructured behaves exactly as a Map
}

func (s *reflectUnstructuredStruct) Value() interface{} {
	return s.value.Interface()
}

func (s *reflectUnstructuredStruct) IsSet(field ref.Val) ref.Val {
	v, found := s.lookupField(field)
	if v != nil && types.IsUnknownOrError(v) {
		return v
	}
	return types.Bool(found)
}

func (s *reflectUnstructuredStruct) Find(key ref.Val) (ref.Val, bool) {
	return s.lookupField(key)
}

func (s *reflectUnstructuredStruct) Get(key ref.Val) ref.Val {
	v, found := s.lookupField(key)
	if !found {
		return types.ValOrErr(key, "no such key: %v", key)
	}
	return v
}

func (s *reflectUnstructuredStruct) Contains(key ref.Val) ref.Val {
	_, found := s.lookupField(key)
	return types.Bool(found)
}

func (s *reflectUnstructuredStruct) lookupField(key ref.Val) (ref.Val, bool) {
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
		v := UnstructuredReflectToVal(e.Interface())
		return v, true
	}
	return nil, false
}

func (s *reflectUnstructuredStruct) Size() ref.Val {
	cacheEntry := value.TypeReflectEntryOf(s.value.Type())
	count := 0
	for _, fieldCache := range cacheEntry.Fields() {
		if e := fieldCache.GetFrom(s.value); !fieldCache.CanOmit(e) {
			count++
		}
	}
	return types.Int(count)
}

func (s *reflectUnstructuredStruct) Iterator() traits.Iterator {
	cacheEntry := value.TypeReflectEntryOf(s.value.Type())
	var keys []string
	for fieldName, fieldCache := range cacheEntry.Fields() {
		if e := fieldCache.GetFrom(s.value); !fieldCache.CanOmit(e) {
			keys = append(keys, string(fieldName))
		}
	}
	return &reflectUnstructuredStructIterator{reflectUnstructuredStruct: s, keys: keys, idx: 0}
}

type reflectUnstructuredStructIterator struct {
	*reflectUnstructuredStruct
	keys []string
	idx  int
}

func (it *reflectUnstructuredStructIterator) HasNext() ref.Val {
	return types.Bool(it.idx < len(it.keys))
}

func (it *reflectUnstructuredStructIterator) Next() ref.Val {
	if it.idx >= len(it.keys) {
		return types.NewErr("no more elements")
	}
	val := types.String(it.keys[it.idx])
	it.idx++
	return val
}
