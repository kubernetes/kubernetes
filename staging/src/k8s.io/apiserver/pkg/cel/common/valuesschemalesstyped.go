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
	"fmt"
	"reflect"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/intstr"
	"sigs.k8s.io/structured-merge-diff/v6/value"
)

// SchemalessTypedToVal wraps a Go value as a CEL ref.Val.
// It mimics how the value would look if it were converted to an unstructured
// map using JSON serialization, but avoids the full allocation by lazily
// traversing the value via reflection and structured-merge-diff value package.
//
// It provides full functional equivalence to runtime.DefaultUnstructuredConverter.ToUnstructured.
// Any behavioral difference is a bug. The returned ref.Val utilizes internal caching
// for lazy reflection, which is not thread-safe and must only be used synchronously within a
// single CEL evaluation.
func SchemalessTypedToVal(val interface{}) ref.Val {
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
	case []byte:
		if typedVal == nil {
			return types.NullValue
		}
		return types.String(base64.StdEncoding.EncodeToString(typedVal))
	case intstr.IntOrString:
		switch typedVal.Type {
		case intstr.Int:
			return types.Int(typedVal.IntVal)
		case intstr.String:
			return types.String(typedVal.StrVal)
		}
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

	cacheEntry := value.TypeReflectEntryOf(v.Type())
	if cacheEntry.CanConvertToUnstructured() {
		if !v.CanAddr() {
			ptr := reflect.New(v.Type())
			ptr.Elem().Set(v)
			v = ptr.Elem()
		}
		unstr, err := cacheEntry.ToUnstructured(v)
		if err != nil {
			return types.NewErr("failed to convert to unstructured: %v", err)
		}
		if unstr == nil {
			return types.NullValue
		}
		return types.DefaultTypeAdapter.NativeToValue(unstr)
	}
	switch v.Kind() {
	case reflect.Pointer:
		if v.IsNil() {
			return types.NullValue
		}
		return SchemalessTypedToVal(v.Elem().Interface())
	case reflect.Slice:
		if v.Type().Elem().Kind() == reflect.Uint8 { // byte, relection return slice of Uint8 for bytes.
			return types.String(base64.StdEncoding.EncodeToString(v.Bytes()))
		}
		return &reflectSchemalessTypedList{value: v}
	case reflect.Map:
		return &reflectSchemalessTypedMap{value: v}
	case reflect.Struct:
		return &reflectSchemalessTypedStruct{value: v}
	// Match type aliases to primitives by kind
	case reflect.Bool:
		return types.Bool(v.Bool())
	case reflect.String:
		return types.String(v.String())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return types.Int(v.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		uVal := v.Uint()
		if uVal > 9223372036854775807 { // math.MaxInt64
			return types.NewErr("unsigned value %d does not fit into int64 (overflow)", uVal)
		}
		return types.Int(uVal)
	case reflect.Float32, reflect.Float64:
		return types.Double(v.Float())
	default:
		return types.NewErr("unsupported Go type for CEL: %v", v.Type())
	}
}

var _ traits.Lister = &reflectSchemalessTypedList{}

// reflectSchemalessTypedList wraps a Go slice/array as a lazy CEL Lister.
// fieldCache is NOT thread-safe, as CEL-Go's synchronous value evaluation contract
// guarantees that a single wrapped value is only ever accessed by a single goroutine.
type reflectSchemalessTypedList struct {
	value      reflect.Value
	fieldCache []ref.Val
}

func (l *reflectSchemalessTypedList) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	switch typeDesc.Kind() {
	case reflect.Slice:
		return l.value.Interface(), nil
	default:
		return nil, fmt.Errorf("type conversion error from '%s' to '%s'", l.Type(), typeDesc)
	}
}

func (l *reflectSchemalessTypedList) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.ListType:
		return l
	case types.TypeType:
		return types.ListType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", l.Type(), typeValue.TypeName())
}

func (l *reflectSchemalessTypedList) Equal(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	sz := l.Size().(types.Int)
	if sz != otherList.Size().(types.Int) {
		return types.False
	}
	for i := range sz {
		v1 := l.Get(i)
		v2 := otherList.Get(i)
		if v1.Equal(v2) != types.True {
			return types.False
		}
	}
	return types.True
}

func (l *reflectSchemalessTypedList) Type() ref.Type {
	return types.ListType
}

func (l *reflectSchemalessTypedList) Value() interface{} {
	return l.value.Interface()
}

func (l *reflectSchemalessTypedList) Contains(val ref.Val) ref.Val {
	sz := l.Size().(types.Int)
	for i := range sz {
		v := l.Get(i)
		if v.Equal(val) == types.True {
			return types.True
		}
	}
	return types.False
}

func (l *reflectSchemalessTypedList) Add(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return types.MaybeNoSuchOverloadErr(other)
	}
	sz := l.value.Len()
	elements := make([]interface{}, 0, sz)
	for i := range sz {
		elements = append(elements, l.value.Index(i).Interface())
	}
	it := otherList.Iterator()
	for it.HasNext() == types.True {
		elements = append(elements, it.Next().Value())
	}
	return &reflectSchemalessTypedList{value: reflect.ValueOf(elements)}
}

func (l *reflectSchemalessTypedList) Get(index ref.Val) ref.Val {
	idx, ok := index.(types.Int)
	if !ok {
		return types.ValOrErr(index, "unsupported index type: %T", index)
	}
	if idx < 0 || int(idx) >= l.value.Len() {
		return types.NewErr("index out of bounds: %d", idx)
	}
	i := int(idx)
	if l.fieldCache == nil {
		l.fieldCache = make([]ref.Val, l.value.Len())
	}
	if cached := l.fieldCache[i]; cached != nil {
		return cached
	}
	v := SchemalessTypedToVal(l.value.Index(i).Interface())
	l.fieldCache[i] = v
	return v
}

func (l *reflectSchemalessTypedList) Iterator() traits.Iterator {
	return &reflectSchemalessTypedListIterator{reflectSchemalessTypedList: l, idx: 0}
}

func (l *reflectSchemalessTypedList) Size() ref.Val {
	return types.Int(l.value.Len())
}

type reflectSchemalessTypedListIterator struct {
	*reflectSchemalessTypedList
	idx int
}

func (it *reflectSchemalessTypedListIterator) HasNext() ref.Val {
	return types.Bool(it.idx < it.reflectSchemalessTypedList.value.Len())
}

func (it *reflectSchemalessTypedListIterator) Next() ref.Val {
	if it.idx >= it.reflectSchemalessTypedList.value.Len() {
		return types.NewErr("no more elements")
	}
	val := it.reflectSchemalessTypedList.Get(types.Int(it.idx))
	it.idx++
	return val
}

var _ traits.Mapper = &reflectSchemalessTypedMap{}

// reflectSchemalessTypedMap wraps a Go map as a lazy CEL Mapper.
// fieldCache is NOT thread-safe, as CEL-Go's synchronous value evaluation contract
// guarantees that a single wrapped value is only ever accessed by a single goroutine.
type reflectSchemalessTypedMap struct {
	value      reflect.Value
	fieldCache map[string]ref.Val
}

func (m *reflectSchemalessTypedMap) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	if m.value.Type().AssignableTo(typeDesc) {
		return m.value.Interface(), nil
	}
	return nil, fmt.Errorf("type conversion error from '%s' to '%s'", m.Type(), typeDesc)
}

func (m *reflectSchemalessTypedMap) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.MapType:
		return m
	case types.TypeType:
		return types.MapType
	}
	return types.NewErr("type conversion error from '%s' to '%s'", m.Type(), typeValue.TypeName())
}

func (m *reflectSchemalessTypedMap) Equal(other ref.Val) ref.Val {
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
		v1, _ := m.Find(key)
		v2, found := otherMap.Find(key)
		if !found || v1.Equal(v2) != types.True {
			return types.False
		}
	}
	return types.True
}

func (m *reflectSchemalessTypedMap) Type() ref.Type {
	return types.MapType
}

func (m *reflectSchemalessTypedMap) Value() interface{} {
	return m.value.Interface()
}

func (m *reflectSchemalessTypedMap) Contains(key ref.Val) ref.Val {
	v, found := m.Find(key)
	if v != nil && types.IsError(v) {
		return v
	}
	return types.Bool(found)
}

func (m *reflectSchemalessTypedMap) Find(key ref.Val) (ref.Val, bool) {
	keyVal, ok := key.(types.String)
	if !ok {
		return types.ValOrErr(key, "unsupported map key type: %T", key), false
	}
	fieldName := string(keyVal)
	if m.fieldCache != nil {
		if cached, ok := m.fieldCache[fieldName]; ok {
			return cached, cached != nil
		}
	}

	var result ref.Val
	found := false

	keyType := m.value.Type().Key()
	if keyType.Kind() == reflect.String {
		reflectKey := reflect.ValueOf(fieldName).Convert(keyType)
		val := m.value.MapIndex(reflectKey)
		if val.IsValid() {
			result = SchemalessTypedToVal(val.Interface())
			found = true
		}
	}

	if m.fieldCache == nil {
		m.fieldCache = make(map[string]ref.Val)
	}
	m.fieldCache[fieldName] = result
	return result, found
}

func (m *reflectSchemalessTypedMap) Get(key ref.Val) ref.Val {
	v, found := m.Find(key)
	if !found {
		if v != nil && types.IsError(v) {
			return v
		}
		return types.ValOrErr(key, "no such key: %v", key)
	}
	return v
}

func (m *reflectSchemalessTypedMap) Iterator() traits.Iterator {
	return &reflectSchemalessTypedMapIterator{reflectSchemalessTypedMap: m, keys: m.value.MapKeys(), idx: 0}
}

func (m *reflectSchemalessTypedMap) Size() ref.Val {
	return types.Int(m.value.Len())
}

type reflectSchemalessTypedMapIterator struct {
	*reflectSchemalessTypedMap
	keys []reflect.Value
	idx  int
}

func (it *reflectSchemalessTypedMapIterator) HasNext() ref.Val {
	return types.Bool(it.idx < len(it.keys))
}

func (it *reflectSchemalessTypedMapIterator) Next() ref.Val {
	if it.idx >= len(it.keys) {
		return types.NewErr("no more elements")
	}
	val := types.String(it.keys[it.idx].String())
	it.idx++
	return val
}

var _ traits.Mapper = &reflectSchemalessTypedStruct{}

// reflectSchemalessTypedStruct wraps a Go struct as a lazy CEL Mapper.
// fieldCache is NOT thread-safe, as CEL-Go's synchronous value evaluation contract
// guarantees that a single wrapped value is only ever accessed by a single goroutine.
type reflectSchemalessTypedStruct struct {
	value      reflect.Value
	fieldCache map[string]ref.Val
}

func (s *reflectSchemalessTypedStruct) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	if s.value.Type().AssignableTo(typeDesc) {
		return s.value.Interface(), nil
	}
	return nil, fmt.Errorf("type conversion error from struct type %v to %v", s.value.Type(), typeDesc)
}

func (s *reflectSchemalessTypedStruct) ConvertToType(typeValue ref.Type) ref.Val {
	switch typeValue {
	case types.MapType:
		return s
	case types.TypeType:
		return types.MapType
	}
	return types.NewErr("type conversion error from struct to %s", typeValue.TypeName())
}

func (s *reflectSchemalessTypedStruct) Equal(other ref.Val) ref.Val {
	otherStruct, ok := other.(*reflectSchemalessTypedStruct)
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
		v1, _ := s.Find(key)
		v2, found := otherMap.Find(key)
		if !found {
			return types.False
		}
		eq := v1.Equal(v2)
		if eq != types.True {
			return eq
		}
	}
	return types.True
}

func (s *reflectSchemalessTypedStruct) Type() ref.Type {
	return types.MapType // Unstructured behaves exactly as a Map
}

func (s *reflectSchemalessTypedStruct) Value() interface{} {
	return s.value.Interface()
}

func (s *reflectSchemalessTypedStruct) IsSet(field ref.Val) ref.Val {
	v, found := s.lookupField(field)
	if v != nil && types.IsUnknownOrError(v) {
		return v
	}
	return types.Bool(found)
}

func (s *reflectSchemalessTypedStruct) Find(key ref.Val) (ref.Val, bool) {
	return s.lookupField(key)
}

func (s *reflectSchemalessTypedStruct) Get(key ref.Val) ref.Val {
	v, found := s.lookupField(key)
	if !found {
		return types.ValOrErr(key, "no such key: %v", key)
	}
	return v
}

func (s *reflectSchemalessTypedStruct) Contains(key ref.Val) ref.Val {
	v, found := s.Find(key)
	if v != nil && types.IsError(v) {
		return v
	}
	return types.Bool(found)
}

func (s *reflectSchemalessTypedStruct) lookupField(key ref.Val) (ref.Val, bool) {
	keyStr, ok := key.(types.String)
	if !ok {
		return types.ValOrErr(key, "unsupported map key type: %T", key), true
	}
	fieldName := keyStr.Value().(string)

	if s.fieldCache != nil {
		if cached, ok := s.fieldCache[fieldName]; ok {
			return cached, cached != nil
		}
	}

	var result ref.Val
	found := false

	cacheEntry := value.TypeReflectEntryOf(s.value.Type())
	fieldCache, ok := cacheEntry.Fields()[fieldName]
	if ok {
		if e := fieldCache.GetFrom(s.value); !fieldCache.CanOmit(e) {
			result = SchemalessTypedToVal(e.Interface())
			found = true
		}
	}

	if s.fieldCache == nil {
		s.fieldCache = make(map[string]ref.Val)
	}
	s.fieldCache[fieldName] = result
	return result, found
}

func (s *reflectSchemalessTypedStruct) Size() ref.Val {
	cacheEntry := value.TypeReflectEntryOf(s.value.Type())
	count := 0
	for _, fieldCache := range cacheEntry.Fields() {
		if e := fieldCache.GetFrom(s.value); !fieldCache.CanOmit(e) {
			count++
		}
	}
	return types.Int(count)
}

func (s *reflectSchemalessTypedStruct) Iterator() traits.Iterator {
	cacheEntry := value.TypeReflectEntryOf(s.value.Type())
	var keys []string
	for fieldName, fieldCache := range cacheEntry.Fields() {
		if e := fieldCache.GetFrom(s.value); !fieldCache.CanOmit(e) {
			keys = append(keys, string(fieldName))
		}
	}
	return &reflectSchemalessTypedStructIterator{reflectSchemalessTypedStruct: s, keys: keys, idx: 0}
}

type reflectSchemalessTypedStructIterator struct {
	*reflectSchemalessTypedStruct
	keys []string
	idx  int
}

func (it *reflectSchemalessTypedStructIterator) HasNext() ref.Val {
	return types.Bool(it.idx < len(it.keys))
}

func (it *reflectSchemalessTypedStructIterator) Next() ref.Val {
	if it.idx >= len(it.keys) {
		return types.NewErr("no more elements")
	}
	val := types.String(it.keys[it.idx])
	it.idx++
	return val
}
