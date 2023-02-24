// Copyright 2018 Google LLC
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

package types

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/stoewer/go-strcase"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/google/cel-go/common/types/pb"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

// NewDynamicMap returns a traits.Mapper value with dynamic key, value pairs.
func NewDynamicMap(adapter ref.TypeAdapter, value any) traits.Mapper {
	refValue := reflect.ValueOf(value)
	return &baseMap{
		TypeAdapter: adapter,
		mapAccessor: newReflectMapAccessor(adapter, refValue),
		value:       value,
		size:        refValue.Len(),
	}
}

// NewJSONStruct creates a traits.Mapper implementation backed by a JSON struct that has been
// encoded in protocol buffer form.
//
// The `adapter` argument provides type adaptation capabilities from proto to CEL.
func NewJSONStruct(adapter ref.TypeAdapter, value *structpb.Struct) traits.Mapper {
	fields := value.GetFields()
	return &baseMap{
		TypeAdapter: adapter,
		mapAccessor: newJSONStructAccessor(adapter, fields),
		value:       value,
		size:        len(fields),
	}
}

// NewRefValMap returns a specialized traits.Mapper with CEL valued keys and values.
func NewRefValMap(adapter ref.TypeAdapter, value map[ref.Val]ref.Val) traits.Mapper {
	return &baseMap{
		TypeAdapter: adapter,
		mapAccessor: newRefValMapAccessor(value),
		value:       value,
		size:        len(value),
	}
}

// NewStringInterfaceMap returns a specialized traits.Mapper with string keys and interface values.
func NewStringInterfaceMap(adapter ref.TypeAdapter, value map[string]any) traits.Mapper {
	return &baseMap{
		TypeAdapter: adapter,
		mapAccessor: newStringIfaceMapAccessor(adapter, value),
		value:       value,
		size:        len(value),
	}
}

// NewStringStringMap returns a specialized traits.Mapper with string keys and values.
func NewStringStringMap(adapter ref.TypeAdapter, value map[string]string) traits.Mapper {
	return &baseMap{
		TypeAdapter: adapter,
		mapAccessor: newStringMapAccessor(value),
		value:       value,
		size:        len(value),
	}
}

// NewProtoMap returns a specialized traits.Mapper for handling protobuf map values.
func NewProtoMap(adapter ref.TypeAdapter, value *pb.Map) traits.Mapper {
	return &protoMap{
		TypeAdapter: adapter,
		value:       value,
	}
}

var (
	// MapType singleton.
	MapType = NewTypeValue("map",
		traits.ContainerType,
		traits.IndexerType,
		traits.IterableType,
		traits.SizerType)
)

// mapAccessor is a private interface for finding values within a map and iterating over the keys.
// This interface implements portions of the API surface area required by the traits.Mapper
// interface.
type mapAccessor interface {
	// Find returns a value, if one exists, for the input key.
	//
	// If the key is not found the function returns (nil, false).
	Find(ref.Val) (ref.Val, bool)

	// Iterator returns an Iterator over the map key set.
	Iterator() traits.Iterator
}

// baseMap is a reflection based map implementation designed to handle a variety of map-like types.
//
// Since CEL is side-effect free, the base map represents an immutable object.
type baseMap struct {
	// TypeAdapter used to convert keys and values accessed within the map.
	ref.TypeAdapter

	// mapAccessor interface implementation used to find and iterate over map keys.
	mapAccessor

	// value is the native Go value upon which the map type operators.
	value any

	// size is the number of entries in the map.
	size int
}

// Contains implements the traits.Container interface method.
func (m *baseMap) Contains(index ref.Val) ref.Val {
	_, found := m.Find(index)
	return Bool(found)
}

// ConvertToNative implements the ref.Val interface method.
func (m *baseMap) ConvertToNative(typeDesc reflect.Type) (any, error) {
	// If the map is already assignable to the desired type return it, e.g. interfaces and
	// maps with the same key value types.
	if reflect.TypeOf(m.value).AssignableTo(typeDesc) {
		return m.value, nil
	}
	if reflect.TypeOf(m).AssignableTo(typeDesc) {
		return m, nil
	}
	switch typeDesc {
	case anyValueType:
		json, err := m.ConvertToNative(jsonStructType)
		if err != nil {
			return nil, err
		}
		return anypb.New(json.(proto.Message))
	case jsonValueType, jsonStructType:
		jsonEntries, err :=
			m.ConvertToNative(reflect.TypeOf(map[string]*structpb.Value{}))
		if err != nil {
			return nil, err
		}
		jsonMap := &structpb.Struct{Fields: jsonEntries.(map[string]*structpb.Value)}
		if typeDesc == jsonStructType {
			return jsonMap, nil
		}
		return structpb.NewStructValue(jsonMap), nil
	}

	// Unwrap pointers, but track their use.
	isPtr := false
	if typeDesc.Kind() == reflect.Ptr {
		tk := typeDesc
		typeDesc = typeDesc.Elem()
		if typeDesc.Kind() == reflect.Ptr {
			return nil, fmt.Errorf("unsupported type conversion to '%v'", tk)
		}
		isPtr = true
	}
	switch typeDesc.Kind() {
	// Map conversion.
	case reflect.Map:
		otherKey := typeDesc.Key()
		otherElem := typeDesc.Elem()
		nativeMap := reflect.MakeMapWithSize(typeDesc, m.size)
		it := m.Iterator()
		for it.HasNext() == True {
			key := it.Next()
			refKeyValue, err := key.ConvertToNative(otherKey)
			if err != nil {
				return nil, err
			}
			refElemValue, err := m.Get(key).ConvertToNative(otherElem)
			if err != nil {
				return nil, err
			}
			nativeMap.SetMapIndex(reflect.ValueOf(refKeyValue), reflect.ValueOf(refElemValue))
		}
		return nativeMap.Interface(), nil
	case reflect.Struct:
		nativeStructPtr := reflect.New(typeDesc)
		nativeStruct := nativeStructPtr.Elem()
		it := m.Iterator()
		for it.HasNext() == True {
			key := it.Next()
			// Ensure the field name being referenced is exported.
			// Only exported (public) field names can be set by reflection, where the name
			// must be at least one character in length and start with an upper-case letter.
			fieldName := key.ConvertToType(StringType)
			if IsError(fieldName) {
				return nil, fieldName.(*Err)
			}
			name := string(fieldName.(String))
			name = strcase.UpperCamelCase(name)
			fieldRef := nativeStruct.FieldByName(name)
			if !fieldRef.IsValid() {
				return nil, fmt.Errorf("type conversion error, no such field '%s' in type '%v'", name, typeDesc)
			}
			fieldValue, err := m.Get(key).ConvertToNative(fieldRef.Type())
			if err != nil {
				return nil, err
			}
			fieldRef.Set(reflect.ValueOf(fieldValue))
		}
		if isPtr {
			return nativeStructPtr.Interface(), nil
		}
		return nativeStruct.Interface(), nil
	}
	return nil, fmt.Errorf("type conversion error from map to '%v'", typeDesc)
}

// ConvertToType implements the ref.Val interface method.
func (m *baseMap) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case MapType:
		return m
	case TypeType:
		return MapType
	}
	return NewErr("type conversion error from '%s' to '%s'", MapType, typeVal)
}

// Equal implements the ref.Val interface method.
func (m *baseMap) Equal(other ref.Val) ref.Val {
	otherMap, ok := other.(traits.Mapper)
	if !ok {
		return False
	}
	if m.Size() != otherMap.Size() {
		return False
	}
	it := m.Iterator()
	for it.HasNext() == True {
		key := it.Next()
		thisVal, _ := m.Find(key)
		otherVal, found := otherMap.Find(key)
		if !found {
			return False
		}
		valEq := Equal(thisVal, otherVal)
		if valEq == False {
			return False
		}
	}
	return True
}

// Get implements the traits.Indexer interface method.
func (m *baseMap) Get(key ref.Val) ref.Val {
	v, found := m.Find(key)
	if !found {
		return ValOrErr(v, "no such key: %v", key)
	}
	return v
}

// IsZeroValue returns true if the map is empty.
func (m *baseMap) IsZeroValue() bool {
	return m.size == 0
}

// Size implements the traits.Sizer interface method.
func (m *baseMap) Size() ref.Val {
	return Int(m.size)
}

// String converts the map into a human-readable string.
func (m *baseMap) String() string {
	var sb strings.Builder
	sb.WriteString("{")
	it := m.Iterator()
	i := 0
	for it.HasNext() == True {
		k := it.Next()
		v, _ := m.Find(k)
		sb.WriteString(fmt.Sprintf("%v: %v", k, v))
		if i != m.size-1 {
			sb.WriteString(", ")
		}
		i++
	}
	sb.WriteString("}")
	return sb.String()
}

// Type implements the ref.Val interface method.
func (m *baseMap) Type() ref.Type {
	return MapType
}

// Value implements the ref.Val interface method.
func (m *baseMap) Value() any {
	return m.value
}

func newJSONStructAccessor(adapter ref.TypeAdapter, st map[string]*structpb.Value) mapAccessor {
	return &jsonStructAccessor{
		TypeAdapter: adapter,
		st:          st,
	}
}

type jsonStructAccessor struct {
	ref.TypeAdapter
	st map[string]*structpb.Value
}

// Find searches the json struct field map for the input key value and returns (value, true) if
// found.
//
// If the key is not found the function returns (nil, false).
func (a *jsonStructAccessor) Find(key ref.Val) (ref.Val, bool) {
	strKey, ok := key.(String)
	if !ok {
		return nil, false
	}
	keyVal, found := a.st[string(strKey)]
	if !found {
		return nil, false
	}
	return a.NativeToValue(keyVal), true
}

// Iterator creates a new traits.Iterator from the set of JSON struct field names.
func (a *jsonStructAccessor) Iterator() traits.Iterator {
	// Copy the keys to make their order stable.
	mapKeys := make([]string, len(a.st))
	i := 0
	for k := range a.st {
		mapKeys[i] = k
		i++
	}
	return &stringKeyIterator{
		mapKeys: mapKeys,
		len:     len(mapKeys),
	}
}

func newReflectMapAccessor(adapter ref.TypeAdapter, value reflect.Value) mapAccessor {
	keyType := value.Type().Key()
	return &reflectMapAccessor{
		TypeAdapter: adapter,
		refValue:    value,
		keyType:     keyType,
	}
}

type reflectMapAccessor struct {
	ref.TypeAdapter
	refValue reflect.Value
	keyType  reflect.Type
}

// Find converts the input key to a native Golang type and then uses reflection to find the key,
// returning (value, true) if present.
//
// If the key is not found the function returns (nil, false).
func (m *reflectMapAccessor) Find(key ref.Val) (ref.Val, bool) {
	if m.refValue.Len() == 0 {
		return nil, false
	}
	if keyVal, found := m.findInternal(key); found {
		return keyVal, true
	}
	switch k := key.(type) {
	// Double is not a valid proto map key type, so check for the key as an int or uint.
	case Double:
		if ik, ok := doubleToInt64Lossless(float64(k)); ok {
			if keyVal, found := m.findInternal(Int(ik)); found {
				return keyVal, true
			}
		}
		if uk, ok := doubleToUint64Lossless(float64(k)); ok {
			return m.findInternal(Uint(uk))
		}
	// map keys of type double are not supported.
	case Int:
		if uk, ok := int64ToUint64Lossless(int64(k)); ok {
			return m.findInternal(Uint(uk))
		}
	case Uint:
		if ik, ok := uint64ToInt64Lossless(uint64(k)); ok {
			return m.findInternal(Int(ik))
		}
	}
	return nil, false
}

// findInternal attempts to convert the incoming key to the map's internal native type
// and then returns the value, if found.
func (m *reflectMapAccessor) findInternal(key ref.Val) (ref.Val, bool) {
	k, err := key.ConvertToNative(m.keyType)
	if err != nil {
		return nil, false
	}
	refKey := reflect.ValueOf(k)
	val := m.refValue.MapIndex(refKey)
	if val.IsValid() {
		return m.NativeToValue(val.Interface()), true
	}
	return nil, false
}

// Iterator creates a Golang reflection based traits.Iterator.
func (m *reflectMapAccessor) Iterator() traits.Iterator {
	return &mapIterator{
		TypeAdapter: m.TypeAdapter,
		mapKeys:     m.refValue.MapRange(),
		len:         m.refValue.Len(),
	}
}

func newRefValMapAccessor(mapVal map[ref.Val]ref.Val) mapAccessor {
	return &refValMapAccessor{mapVal: mapVal}
}

type refValMapAccessor struct {
	mapVal map[ref.Val]ref.Val
}

// Find uses native map accesses to find the key, returning (value, true) if present.
//
// If the key is not found the function returns (nil, false).
func (a *refValMapAccessor) Find(key ref.Val) (ref.Val, bool) {
	if len(a.mapVal) == 0 {
		return nil, false
	}
	if keyVal, found := a.mapVal[key]; found {
		return keyVal, true
	}
	switch k := key.(type) {
	case Double:
		if ik, ok := doubleToInt64Lossless(float64(k)); ok {
			if keyVal, found := a.mapVal[Int(ik)]; found {
				return keyVal, found
			}
		}
		if uk, ok := doubleToUint64Lossless(float64(k)); ok {
			keyVal, found := a.mapVal[Uint(uk)]
			return keyVal, found
		}
	// map keys of type double are not supported.
	case Int:
		if uk, ok := int64ToUint64Lossless(int64(k)); ok {
			keyVal, found := a.mapVal[Uint(uk)]
			return keyVal, found
		}
	case Uint:
		if ik, ok := uint64ToInt64Lossless(uint64(k)); ok {
			keyVal, found := a.mapVal[Int(ik)]
			return keyVal, found
		}
	}
	return nil, false
}

// Iterator produces a new traits.Iterator which iterates over the map keys via Golang reflection.
func (a *refValMapAccessor) Iterator() traits.Iterator {
	return &mapIterator{
		TypeAdapter: DefaultTypeAdapter,
		mapKeys:     reflect.ValueOf(a.mapVal).MapRange(),
		len:         len(a.mapVal),
	}
}

func newStringMapAccessor(strMap map[string]string) mapAccessor {
	return &stringMapAccessor{mapVal: strMap}
}

type stringMapAccessor struct {
	mapVal map[string]string
}

// Find uses native map accesses to find the key, returning (value, true) if present.
//
// If the key is not found the function returns (nil, false).
func (a *stringMapAccessor) Find(key ref.Val) (ref.Val, bool) {
	strKey, ok := key.(String)
	if !ok {
		return nil, false
	}
	keyVal, found := a.mapVal[string(strKey)]
	if !found {
		return nil, false
	}
	return String(keyVal), true
}

// Iterator creates a new traits.Iterator from the string key set of the map.
func (a *stringMapAccessor) Iterator() traits.Iterator {
	// Copy the keys to make their order stable.
	mapKeys := make([]string, len(a.mapVal))
	i := 0
	for k := range a.mapVal {
		mapKeys[i] = k
		i++
	}
	return &stringKeyIterator{
		mapKeys: mapKeys,
		len:     len(mapKeys),
	}
}

func newStringIfaceMapAccessor(adapter ref.TypeAdapter, mapVal map[string]any) mapAccessor {
	return &stringIfaceMapAccessor{
		TypeAdapter: adapter,
		mapVal:      mapVal,
	}
}

type stringIfaceMapAccessor struct {
	ref.TypeAdapter
	mapVal map[string]any
}

// Find uses native map accesses to find the key, returning (value, true) if present.
//
// If the key is not found the function returns (nil, false).
func (a *stringIfaceMapAccessor) Find(key ref.Val) (ref.Val, bool) {
	strKey, ok := key.(String)
	if !ok {
		return nil, false
	}
	keyVal, found := a.mapVal[string(strKey)]
	if !found {
		return nil, false
	}
	return a.NativeToValue(keyVal), true
}

// Iterator creates a new traits.Iterator from the string key set of the map.
func (a *stringIfaceMapAccessor) Iterator() traits.Iterator {
	// Copy the keys to make their order stable.
	mapKeys := make([]string, len(a.mapVal))
	i := 0
	for k := range a.mapVal {
		mapKeys[i] = k
		i++
	}
	return &stringKeyIterator{
		mapKeys: mapKeys,
		len:     len(mapKeys),
	}
}

// protoMap is a specialized, separate implementation of the traits.Mapper interfaces tailored to
// accessing protoreflect.Map values.
type protoMap struct {
	ref.TypeAdapter
	value *pb.Map
}

// Contains returns whether the map contains the given key.
func (m *protoMap) Contains(key ref.Val) ref.Val {
	_, found := m.Find(key)
	return Bool(found)
}

// ConvertToNative implements the ref.Val interface method.
//
// Note, assignment to Golang struct types is not yet supported.
func (m *protoMap) ConvertToNative(typeDesc reflect.Type) (any, error) {
	// If the map is already assignable to the desired type return it, e.g. interfaces and
	// maps with the same key value types.
	switch typeDesc {
	case anyValueType:
		json, err := m.ConvertToNative(jsonStructType)
		if err != nil {
			return nil, err
		}
		return anypb.New(json.(proto.Message))
	case jsonValueType, jsonStructType:
		jsonEntries, err :=
			m.ConvertToNative(reflect.TypeOf(map[string]*structpb.Value{}))
		if err != nil {
			return nil, err
		}
		jsonMap := &structpb.Struct{
			Fields: jsonEntries.(map[string]*structpb.Value)}
		if typeDesc == jsonStructType {
			return jsonMap, nil
		}
		return structpb.NewStructValue(jsonMap), nil
	}
	switch typeDesc.Kind() {
	case reflect.Struct, reflect.Ptr:
		if reflect.TypeOf(m.value).AssignableTo(typeDesc) {
			return m.value, nil
		}
		if reflect.TypeOf(m).AssignableTo(typeDesc) {
			return m, nil
		}
	}
	if typeDesc.Kind() != reflect.Map {
		return nil, fmt.Errorf("unsupported type conversion: %v to map", typeDesc)
	}

	keyType := m.value.KeyType.ReflectType()
	valType := m.value.ValueType.ReflectType()
	otherKeyType := typeDesc.Key()
	otherValType := typeDesc.Elem()
	mapVal := reflect.MakeMapWithSize(typeDesc, m.value.Len())
	var err error
	m.value.Range(func(key protoreflect.MapKey, val protoreflect.Value) bool {
		ntvKey := key.Interface()
		ntvVal := val.Interface()
		switch pv := ntvVal.(type) {
		case protoreflect.Message:
			ntvVal = pv.Interface()
		}
		if keyType == otherKeyType && valType == otherValType {
			mapVal.SetMapIndex(reflect.ValueOf(ntvKey), reflect.ValueOf(ntvVal))
			return true
		}
		celKey := m.NativeToValue(ntvKey)
		celVal := m.NativeToValue(ntvVal)
		ntvKey, err = celKey.ConvertToNative(otherKeyType)
		if err != nil {
			// early terminate the range loop.
			return false
		}
		ntvVal, err = celVal.ConvertToNative(otherValType)
		if err != nil {
			// early terminate the range loop.
			return false
		}
		mapVal.SetMapIndex(reflect.ValueOf(ntvKey), reflect.ValueOf(ntvVal))
		return true
	})
	if err != nil {
		return nil, err
	}
	return mapVal.Interface(), nil
}

// ConvertToType implements the ref.Val interface method.
func (m *protoMap) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case MapType:
		return m
	case TypeType:
		return MapType
	}
	return NewErr("type conversion error from '%s' to '%s'", MapType, typeVal)
}

// Equal implements the ref.Val interface method.
func (m *protoMap) Equal(other ref.Val) ref.Val {
	otherMap, ok := other.(traits.Mapper)
	if !ok {
		return False
	}
	if m.value.Map.Len() != int(otherMap.Size().(Int)) {
		return False
	}
	var retVal ref.Val = True
	m.value.Range(func(key protoreflect.MapKey, val protoreflect.Value) bool {
		keyVal := m.NativeToValue(key.Interface())
		valVal := m.NativeToValue(val)
		otherVal, found := otherMap.Find(keyVal)
		if !found {
			retVal = False
			return false
		}
		valEq := Equal(valVal, otherVal)
		if valEq != True {
			retVal = valEq
			return false
		}
		return true
	})
	return retVal
}

// Find returns whether the protoreflect.Map contains the input key.
//
// If the key is not found the function returns (nil, false).
func (m *protoMap) Find(key ref.Val) (ref.Val, bool) {
	if keyVal, found := m.findInternal(key); found {
		return keyVal, true
	}
	switch k := key.(type) {
	// Double is not a valid proto map key type, so check for the key as an int or uint.
	case Double:
		if ik, ok := doubleToInt64Lossless(float64(k)); ok {
			if keyVal, found := m.findInternal(Int(ik)); found {
				return keyVal, true
			}
		}
		if uk, ok := doubleToUint64Lossless(float64(k)); ok {
			return m.findInternal(Uint(uk))
		}
	// map keys of type double are not supported.
	case Int:
		if uk, ok := int64ToUint64Lossless(int64(k)); ok {
			return m.findInternal(Uint(uk))
		}
	case Uint:
		if ik, ok := uint64ToInt64Lossless(uint64(k)); ok {
			return m.findInternal(Int(ik))
		}
	}
	return nil, false
}

// findInternal attempts to convert the incoming key to the map's internal native type
// and then returns the value, if found.
func (m *protoMap) findInternal(key ref.Val) (ref.Val, bool) {
	// Convert the input key to the expected protobuf key type.
	ntvKey, err := key.ConvertToNative(m.value.KeyType.ReflectType())
	if err != nil {
		return nil, false
	}
	// Use protoreflection to get the key value.
	val := m.value.Get(protoreflect.ValueOf(ntvKey).MapKey())
	if !val.IsValid() {
		return nil, false
	}
	// Perform nominal type unwrapping from the input value.
	switch v := val.Interface().(type) {
	case protoreflect.List, protoreflect.Map:
		// Maps do not support list or map values
		return nil, false
	default:
		return m.NativeToValue(v), true
	}
}

// Get implements the traits.Indexer interface method.
func (m *protoMap) Get(key ref.Val) ref.Val {
	v, found := m.Find(key)
	if !found {
		return ValOrErr(v, "no such key: %v", key)
	}
	return v
}

// IsZeroValue returns true if the map is empty.
func (m *protoMap) IsZeroValue() bool {
	return m.value.Len() == 0
}

// Iterator implements the traits.Iterable interface method.
func (m *protoMap) Iterator() traits.Iterator {
	// Copy the keys to make their order stable.
	mapKeys := make([]protoreflect.MapKey, 0, m.value.Len())
	m.value.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
		mapKeys = append(mapKeys, k)
		return true
	})
	return &protoMapIterator{
		TypeAdapter: m.TypeAdapter,
		mapKeys:     mapKeys,
		len:         m.value.Len(),
	}
}

// Size returns the number of entries in the protoreflect.Map.
func (m *protoMap) Size() ref.Val {
	return Int(m.value.Len())
}

// Type implements the ref.Val interface method.
func (m *protoMap) Type() ref.Type {
	return MapType
}

// Value implements the ref.Val interface method.
func (m *protoMap) Value() any {
	return m.value
}

type mapIterator struct {
	*baseIterator
	ref.TypeAdapter
	mapKeys *reflect.MapIter
	cursor  int
	len     int
}

// HasNext implements the traits.Iterator interface method.
func (it *mapIterator) HasNext() ref.Val {
	return Bool(it.cursor < it.len)
}

// Next implements the traits.Iterator interface method.
func (it *mapIterator) Next() ref.Val {
	if it.HasNext() == True && it.mapKeys.Next() {
		it.cursor++
		refKey := it.mapKeys.Key()
		return it.NativeToValue(refKey.Interface())
	}
	return nil
}

type protoMapIterator struct {
	*baseIterator
	ref.TypeAdapter
	mapKeys []protoreflect.MapKey
	cursor  int
	len     int
}

// HasNext implements the traits.Iterator interface method.
func (it *protoMapIterator) HasNext() ref.Val {
	return Bool(it.cursor < it.len)
}

// Next implements the traits.Iterator interface method.
func (it *protoMapIterator) Next() ref.Val {
	if it.HasNext() == True {
		index := it.cursor
		it.cursor++
		refKey := it.mapKeys[index]
		return it.NativeToValue(refKey.Interface())
	}
	return nil
}

type stringKeyIterator struct {
	*baseIterator
	mapKeys []string
	cursor  int
	len     int
}

// HasNext implements the traits.Iterator interface method.
func (it *stringKeyIterator) HasNext() ref.Val {
	return Bool(it.cursor < it.len)
}

// Next implements the traits.Iterator interface method.
func (it *stringKeyIterator) Next() ref.Val {
	if it.HasNext() == True {
		index := it.cursor
		it.cursor++
		return String(it.mapKeys[index])
	}
	return nil
}
