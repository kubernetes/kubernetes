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
	"sort"
	"strings"
	"unicode"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"github.com/google/cel-go/common/types/pb"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

// NewDynamicMap returns a traits.Mapper value with dynamic key, value pairs.
func NewDynamicMap(adapter Adapter, value any) traits.Mapper {
	refValue := reflect.ValueOf(value)
	return &baseMap{
		Adapter:     adapter,
		mapAccessor: newReflectMapAccessor(adapter, refValue),
		value:       value,
		size:        refValue.Len(),
	}
}

// NewJSONStruct creates a traits.Mapper implementation backed by a JSON struct that has been
// encoded in protocol buffer form.
//
// The `adapter` argument provides type adaptation capabilities from proto to CEL.
func NewJSONStruct(adapter Adapter, value *structpb.Struct) traits.Mapper {
	fields := value.GetFields()
	return &baseMap{
		Adapter:     adapter,
		mapAccessor: newJSONStructAccessor(adapter, fields),
		value:       value,
		size:        len(fields),
	}
}

// NewRefValMap returns a specialized traits.Mapper with CEL valued keys and values.
func NewRefValMap(adapter Adapter, value map[ref.Val]ref.Val) traits.Mapper {
	return &baseMap{
		Adapter:     adapter,
		mapAccessor: newRefValMapAccessor(value),
		value:       value,
		size:        len(value),
	}
}

// NewStringInterfaceMap returns a specialized traits.Mapper with string keys and interface values.
func NewStringInterfaceMap(adapter Adapter, value map[string]any) traits.Mapper {
	return &baseMap{
		Adapter:     adapter,
		mapAccessor: newStringIfaceMapAccessor(adapter, value),
		value:       value,
		size:        len(value),
	}
}

// NewStringStringMap returns a specialized traits.Mapper with string keys and values.
func NewStringStringMap(adapter Adapter, value map[string]string) traits.Mapper {
	return &baseMap{
		Adapter:     adapter,
		mapAccessor: newStringMapAccessor(value),
		value:       value,
		size:        len(value),
	}
}

// NewProtoMap returns a specialized traits.Mapper for handling protobuf map values.
func NewProtoMap(adapter Adapter, value *pb.Map) traits.Mapper {
	return &protoMap{
		Adapter: adapter,
		value:   value,
	}
}

// NewMutableMap constructs a mutable map from an adapter and a set of map values.
func NewMutableMap(adapter Adapter, mutableValues map[ref.Val]ref.Val) traits.MutableMapper {
	mutableCopy := make(map[ref.Val]ref.Val, len(mutableValues))
	for k, v := range mutableValues {
		mutableCopy[k] = v
	}
	m := &mutableMap{
		baseMap: &baseMap{
			Adapter:     adapter,
			mapAccessor: newRefValMapAccessor(mutableCopy),
			value:       mutableCopy,
			size:        len(mutableCopy),
		},
		mutableValues: mutableCopy,
	}
	return m
}

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

	// Fold calls the FoldEntry method for each (key, value) pair in the map.
	Fold(traits.Folder)
}

// baseMap is a reflection based map implementation designed to handle a variety of map-like types.
//
// Since CEL is side-effect free, the base map represents an immutable object.
type baseMap struct {
	// TypeAdapter used to convert keys and values accessed within the map.
	Adapter

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
	if typeDesc == reflect.TypeFor[any]() {
		typeDesc = reflect.TypeFor[map[any]any]()
	}
	if reflect.TypeOf(m.value).AssignableTo(typeDesc) {
		return m.value, nil
	}
	if reflect.TypeOf(m).AssignableTo(typeDesc) {
		return m, nil
	}
	switch typeDesc {
	case anyValueType:
		json, err := m.ConvertToNative(JSONStructType)
		if err != nil {
			return nil, err
		}
		return anypb.New(json.(proto.Message))
	case JSONValueType, JSONStructType:
		jsonEntries, err :=
			m.ConvertToNative(reflect.TypeOf(map[string]*structpb.Value{}))
		if err != nil {
			return nil, err
		}
		jsonMap := &structpb.Struct{Fields: jsonEntries.(map[string]*structpb.Value)}
		if typeDesc == JSONStructType {
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
			name = upperCamelCase(name)
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

type baseMapEntry struct {
	key string
	val string
}

func formatMap(m traits.Mapper, sb *strings.Builder) {
	it := m.Iterator()
	var ents []baseMapEntry
	if s, ok := m.Size().(Int); ok {
		ents = make([]baseMapEntry, 0, int(s))
	}
	for it.HasNext() == True {
		k := it.Next()
		v, _ := m.Find(k)
		ents = append(ents, baseMapEntry{Format(k), Format(v)})
	}
	sort.SliceStable(ents, func(i, j int) bool {
		return ents[i].key < ents[j].key
	})
	sb.WriteString("{")
	for i, ent := range ents {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(ent.key)
		sb.WriteString(": ")
		sb.WriteString(ent.val)
	}
	sb.WriteString("}")
}

func (m *baseMap) format(sb *strings.Builder) {
	formatMap(m, sb)
}

// Type implements the ref.Val interface method.
func (m *baseMap) Type() ref.Type {
	return MapType
}

// Value implements the ref.Val interface method.
func (m *baseMap) Value() any {
	return m.value
}

// mutableMap holds onto a set of mutable values which are used for intermediate computations.
type mutableMap struct {
	*baseMap
	mutableValues map[ref.Val]ref.Val
}

// Insert implements the traits.MutableMapper interface method, returning true if the key insertion
// succeeds.
func (m *mutableMap) Insert(k, v ref.Val) ref.Val {
	if _, found := m.Find(k); found {
		return NewErr("insert failed: key %v already exists", k)
	}
	m.mutableValues[k] = v
	return m
}

// ToImmutableMap implements the traits.MutableMapper interface method, converting a mutable map
// an immutable map implementation.
func (m *mutableMap) ToImmutableMap() traits.Mapper {
	return NewRefValMap(m.Adapter, m.mutableValues)
}

func newJSONStructAccessor(adapter Adapter, st map[string]*structpb.Value) mapAccessor {
	return &jsonStructAccessor{
		Adapter: adapter,
		st:      st,
	}
}

type jsonStructAccessor struct {
	Adapter
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

// Fold calls the FoldEntry method for each (key, value) pair in the map.
func (a *jsonStructAccessor) Fold(f traits.Folder) {
	for k, v := range a.st {
		if !f.FoldEntry(k, v) {
			break
		}
	}
}

func newReflectMapAccessor(adapter Adapter, value reflect.Value) mapAccessor {
	keyType := value.Type().Key()
	return &reflectMapAccessor{
		Adapter:  adapter,
		refValue: value,
		keyType:  keyType,
	}
}

type reflectMapAccessor struct {
	Adapter
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
		Adapter: m.Adapter,
		mapKeys: m.refValue.MapRange(),
		len:     m.refValue.Len(),
	}
}

// Fold calls the FoldEntry method for each (key, value) pair in the map.
func (m *reflectMapAccessor) Fold(f traits.Folder) {
	mapRange := m.refValue.MapRange()
	for mapRange.Next() {
		if !f.FoldEntry(mapRange.Key().Interface(), mapRange.Value().Interface()) {
			break
		}
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
		Adapter: DefaultTypeAdapter,
		mapKeys: reflect.ValueOf(a.mapVal).MapRange(),
		len:     len(a.mapVal),
	}
}

// Fold calls the FoldEntry method for each (key, value) pair in the map.
func (a *refValMapAccessor) Fold(f traits.Folder) {
	for k, v := range a.mapVal {
		if !f.FoldEntry(k, v) {
			break
		}
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

// Fold calls the FoldEntry method for each (key, value) pair in the map.
func (a *stringMapAccessor) Fold(f traits.Folder) {
	for k, v := range a.mapVal {
		if !f.FoldEntry(k, v) {
			break
		}
	}
}

func newStringIfaceMapAccessor(adapter Adapter, mapVal map[string]any) mapAccessor {
	return &stringIfaceMapAccessor{
		Adapter: adapter,
		mapVal:  mapVal,
	}
}

type stringIfaceMapAccessor struct {
	Adapter
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

// Fold calls the FoldEntry method for each (key, value) pair in the map.
func (a *stringIfaceMapAccessor) Fold(f traits.Folder) {
	for k, v := range a.mapVal {
		if !f.FoldEntry(k, v) {
			break
		}
	}
}

// protoMap is a specialized, separate implementation of the traits.Mapper interfaces tailored to
// accessing protoreflect.Map values.
type protoMap struct {
	Adapter
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
		json, err := m.ConvertToNative(JSONStructType)
		if err != nil {
			return nil, err
		}
		return anypb.New(json.(proto.Message))
	case JSONValueType, JSONStructType:
		jsonEntries, err :=
			m.ConvertToNative(reflect.TypeOf(map[string]*structpb.Value{}))
		if err != nil {
			return nil, err
		}
		jsonMap := &structpb.Struct{
			Fields: jsonEntries.(map[string]*structpb.Value)}
		if typeDesc == JSONStructType {
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
		Adapter: m.Adapter,
		mapKeys: mapKeys,
		len:     m.value.Len(),
	}
}

// Fold calls the FoldEntry method for each (key, value) pair in the map.
func (m *protoMap) Fold(f traits.Folder) {
	m.value.Range(func(k protoreflect.MapKey, v protoreflect.Value) bool {
		return f.FoldEntry(k.Interface(), v.Interface())
	})
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
	Adapter
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
	Adapter
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

// ToFoldableMap will create a Foldable version of a map suitable for key-value pair iteration.
//
// For values which are already Foldable, this call is a no-op. For all other values, the fold
// is driven via the Iterator HasNext() and Next() calls as well as the map's Get() method
// which means that the folding will function, but take a performance hit.
func ToFoldableMap(m traits.Mapper) traits.Foldable {
	if f, ok := m.(traits.Foldable); ok {
		return f
	}
	return interopFoldableMap{Mapper: m}
}

type interopFoldableMap struct {
	traits.Mapper
}

func (m interopFoldableMap) Fold(f traits.Folder) {
	it := m.Iterator()
	for it.HasNext() == True {
		k := it.Next()
		if !f.FoldEntry(k, m.Get(k)) {
			break
		}
	}
}

// InsertMapKeyValue inserts a key, value pair into the target map if the target map does not
// already contain the given key.
//
// If the map is mutable, it is modified in-place per the MutableMapper contract.
// If the map is not mutable, a copy containing the new key, value pair is made.
func InsertMapKeyValue(m traits.Mapper, k, v ref.Val) ref.Val {
	if mutable, ok := m.(traits.MutableMapper); ok {
		return mutable.Insert(k, v)
	}

	// Otherwise perform the slow version of the insertion which makes a copy of the incoming map.
	if _, found := m.Find(k); !found {
		size := m.Size().(Int)
		copy := make(map[ref.Val]ref.Val, size+1)
		copy[k] = v
		it := m.Iterator()
		for it.HasNext() == True {
			nextK := it.Next()
			nextV := m.Get(nextK)
			copy[nextK] = nextV
		}
		return DefaultTypeAdapter.NativeToValue(copy)
	}
	return NewErr("insert failed: key %v already exists", k)
}

func upperCamelCase(s string) string {
	var newStr strings.Builder
	s = strings.TrimSpace(s)
	var prev rune
	for _, curr := range s {
		if prev == 0 || isDelim(prev) {
			if !isDelim(curr) {
				newStr.WriteRune(unicode.ToUpper(curr))
			}
		} else if !isDelim(curr) {
			if isLower(prev) {
				newStr.WriteRune(curr)
			} else {
				newStr.WriteRune(unicode.ToLower(curr))
			}
		}
		prev = curr
	}
	return newStr.String()
}

func isDelim(r rune) bool {
	return r == '_' || r == '-'
}

func isLower(r rune) bool {
	return r >= 'a' && r <= 'z'
}
