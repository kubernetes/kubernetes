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

	"github.com/google/cel-go/common/types/pb"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"github.com/stoewer/go-strcase"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

// NewDynamicMap returns a traits.Mapper value with dynamic key, value pairs.
func NewDynamicMap(adapter ref.TypeAdapter, value interface{}) traits.Mapper {
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
func NewStringInterfaceMap(adapter ref.TypeAdapter, value map[string]interface{}) traits.Mapper {
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
	// Find returns a value, if one exists, for the inpput key.
	//
	// If the key is not found the function returns (nil, false).
	// If the input key is not valid for the map, or is Err or Unknown the function returns
	// (Unknown|Err, false).
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
	value interface{}

	// size is the number of entries in the map.
	size int
}

// Contains implements the traits.Container interface method.
func (m *baseMap) Contains(index ref.Val) ref.Val {
	val, found := m.Find(index)
	// When the index is not found and val is non-nil, this is an error or unknown value.
	if !found && val != nil && IsUnknownOrError(val) {
		return val
	}
	return Bool(found)
}

// ConvertToNative implements the ref.Val interface method.
func (m *baseMap) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
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
		return MaybeNoSuchOverloadErr(other)
	}
	if m.Size() != otherMap.Size() {
		return False
	}
	it := m.Iterator()
	var maybeErr ref.Val
	for it.HasNext() == True {
		key := it.Next()
		thisVal, _ := m.Find(key)
		otherVal, found := otherMap.Find(key)
		if !found {
			if otherVal == nil {
				return False
			}
			if maybeErr == nil {
				maybeErr = MaybeNoSuchOverloadErr(otherVal)
			}
			continue
		}
		valEq := thisVal.Equal(otherVal)
		if valEq == False {
			return False
		}
		if maybeErr == nil && IsUnknownOrError(valEq) {
			maybeErr = valEq
		}
	}
	if maybeErr != nil {
		return maybeErr
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

// Size implements the traits.Sizer interface method.
func (m *baseMap) Size() ref.Val {
	return Int(m.size)
}

// Type implements the ref.Val interface method.
func (m *baseMap) Type() ref.Type {
	return MapType
}

// Value implements the ref.Val interface method.
func (m *baseMap) Value() interface{} {
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
// If the input key is not a String, or is an  Err or Unknown, the function returns
// (Unknown|Err, false).
func (a *jsonStructAccessor) Find(key ref.Val) (ref.Val, bool) {
	strKey, ok := key.(String)
	if !ok {
		return ValOrErr(key, "unsupported key type: %v", key.Type()), false
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
// If the input key is not a String, or is an  Err or Unknown, the function returns
// (Unknown|Err, false).
func (a *reflectMapAccessor) Find(key ref.Val) (ref.Val, bool) {
	if IsUnknownOrError(key) {
		return MaybeNoSuchOverloadErr(key), false
	}
	if a.refValue.Len() == 0 {
		return nil, false
	}
	k, err := key.ConvertToNative(a.keyType)
	if err != nil {
		return NewErr("unsupported key type: %v", key.Type()), false
	}
	refKey := reflect.ValueOf(k)
	val := a.refValue.MapIndex(refKey)
	if val.IsValid() {
		return a.NativeToValue(val.Interface()), true
	}
	mapIt := a.refValue.MapRange()
	for mapIt.Next() {
		if refKey.Kind() == mapIt.Key().Kind() {
			return nil, false
		}
	}
	return NewErr("unsupported key type: %v", key.Type()), false
}

// Iterator creates a Golang reflection based traits.Iterator.
func (a *reflectMapAccessor) Iterator() traits.Iterator {
	return &mapIterator{
		TypeAdapter: a.TypeAdapter,
		mapKeys:     a.refValue.MapRange(),
		len:         a.refValue.Len(),
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
// If the input key is an Err or Unknown, the function returns (Unknown|Err, false).
func (a *refValMapAccessor) Find(key ref.Val) (ref.Val, bool) {
	if IsUnknownOrError(key) {
		return key, false
	}
	if len(a.mapVal) == 0 {
		return nil, false
	}
	keyVal, found := a.mapVal[key]
	if found {
		return keyVal, true
	}
	for k := range a.mapVal {
		if k.Type().TypeName() == key.Type().TypeName() {
			return nil, false
		}
	}
	return NewErr("unsupported key type: %v", key.Type()), found
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
// If the input key is not a String, or is an Err or Unknown, the function returns
// (Unknown|Err, false).
func (a *stringMapAccessor) Find(key ref.Val) (ref.Val, bool) {
	strKey, ok := key.(String)
	if !ok {
		return ValOrErr(key, "unsupported key type: %v", key.Type()), false
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

func newStringIfaceMapAccessor(adapter ref.TypeAdapter, mapVal map[string]interface{}) mapAccessor {
	return &stringIfaceMapAccessor{
		TypeAdapter: adapter,
		mapVal:      mapVal,
	}
}

type stringIfaceMapAccessor struct {
	ref.TypeAdapter
	mapVal map[string]interface{}
}

// Find uses native map accesses to find the key, returning (value, true) if present.
//
// If the key is not found the function returns (nil, false).
// If the input key is not a String, or is an Err or Unknown, the function returns
// (Unknown|Err, false).
func (a *stringIfaceMapAccessor) Find(key ref.Val) (ref.Val, bool) {
	strKey, ok := key.(String)
	if !ok {
		return ValOrErr(key, "unsupported key type: %v", key.Type()), false
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
	val, found := m.Find(key)
	// When the index is not found and val is non-nil, this is an error or unknown value.
	if !found && val != nil && IsUnknownOrError(val) {
		return val
	}
	return Bool(found)
}

// ConvertToNative implements the ref.Val interface method.
//
// Note, assignment to Golang struct types is not yet supported.
func (m *protoMap) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
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
		switch ntvVal.(type) {
		case protoreflect.Message:
			ntvVal = ntvVal.(protoreflect.Message).Interface()
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
		return MaybeNoSuchOverloadErr(other)
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
			if otherVal == nil {
				retVal = False
				return false
			}
			retVal = MaybeNoSuchOverloadErr(otherVal)
			return false
		}
		valEq := valVal.Equal(otherVal)
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
// If the input key is not a supported proto map key type, or is an Err or Unknown,
// the function returns
// (Unknown|Err, false).
func (m *protoMap) Find(key ref.Val) (ref.Val, bool) {
	if IsUnknownOrError(key) {
		return key, false
	}
	// Convert the input key to the expected protobuf key type.
	ntvKey, err := key.ConvertToNative(m.value.KeyType.ReflectType())
	if err != nil {
		return NewErr("unsupported key type: %v", key.Type()), false
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
		return NewErr("unsupported map element type: (%T)%v", v, v), false
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
func (m *protoMap) Value() interface{} {
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
