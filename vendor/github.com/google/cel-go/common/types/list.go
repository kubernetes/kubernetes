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

	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/common/types/traits"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

var (
	// ListType singleton.
	ListType = NewTypeValue("list",
		traits.AdderType,
		traits.ContainerType,
		traits.IndexerType,
		traits.IterableType,
		traits.SizerType)
)

// NewDynamicList returns a traits.Lister with heterogenous elements.
// value should be an array of "native" types, i.e. any type that
// NativeToValue() can convert to a ref.Val.
func NewDynamicList(adapter ref.TypeAdapter, value interface{}) traits.Lister {
	refValue := reflect.ValueOf(value)
	return &baseList{
		TypeAdapter: adapter,
		value:       value,
		size:        refValue.Len(),
		get: func(i int) interface{} {
			return refValue.Index(i).Interface()
		},
	}
}

// NewStringList returns a traits.Lister containing only strings.
func NewStringList(adapter ref.TypeAdapter, elems []string) traits.Lister {
	return &baseList{
		TypeAdapter: adapter,
		value:       elems,
		size:        len(elems),
		get:         func(i int) interface{} { return elems[i] },
	}
}

// NewRefValList returns a traits.Lister with ref.Val elements.
//
// This type specialization is used with list literals within CEL expressions.
func NewRefValList(adapter ref.TypeAdapter, elems []ref.Val) traits.Lister {
	return &baseList{
		TypeAdapter: adapter,
		value:       elems,
		size:        len(elems),
		get:         func(i int) interface{} { return elems[i] },
	}
}

// NewProtoList returns a traits.Lister based on a pb.List instance.
func NewProtoList(adapter ref.TypeAdapter, list protoreflect.List) traits.Lister {
	return &baseList{
		TypeAdapter: adapter,
		value:       list,
		size:        list.Len(),
		get:         func(i int) interface{} { return list.Get(i).Interface() },
	}
}

// NewJSONList returns a traits.Lister based on structpb.ListValue instance.
func NewJSONList(adapter ref.TypeAdapter, l *structpb.ListValue) traits.Lister {
	vals := l.GetValues()
	return &baseList{
		TypeAdapter: adapter,
		value:       l,
		size:        len(vals),
		get:         func(i int) interface{} { return vals[i] },
	}
}

// NewMutableList creates a new mutable list whose internal state can be modified.
func NewMutableList(adapter ref.TypeAdapter) traits.MutableLister {
	var mutableValues []ref.Val
	return &mutableList{
		baseList: &baseList{
			TypeAdapter: adapter,
			value:       mutableValues,
			size:        0,
			get:         func(i int) interface{} { return mutableValues[i] },
		},
		mutableValues: mutableValues,
	}
}

// baseList points to a list containing elements of any type.
// The `value` is an array of native values, and refValue is its reflection object.
// The `ref.TypeAdapter` enables native type to CEL type conversions.
type baseList struct {
	ref.TypeAdapter
	value interface{}

	// size indicates the number of elements within the list.
	// Since objects are immutable the size of a list is static.
	size int

	// get returns a value at the specified integer index.
	// The index is guaranteed to be checked against the list index range.
	get func(int) interface{}
}

// Add implements the traits.Adder interface method.
func (l *baseList) Add(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	if l.Size() == IntZero {
		return other
	}
	if otherList.Size() == IntZero {
		return l
	}
	return &concatList{
		TypeAdapter: l.TypeAdapter,
		prevList:    l,
		nextList:    otherList}
}

// Contains implements the traits.Container interface method.
func (l *baseList) Contains(elem ref.Val) ref.Val {
	for i := 0; i < l.size; i++ {
		val := l.NativeToValue(l.get(i))
		cmp := elem.Equal(val)
		b, ok := cmp.(Bool)
		if ok && b == True {
			return True
		}
	}
	return False
}

// ConvertToNative implements the ref.Val interface method.
func (l *baseList) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	// If the underlying list value is assignable to the reflected type return it.
	if reflect.TypeOf(l.value).AssignableTo(typeDesc) {
		return l.value, nil
	}
	// If the list wrapper is assignable to the desired type return it.
	if reflect.TypeOf(l).AssignableTo(typeDesc) {
		return l, nil
	}
	// Attempt to convert the list to a set of well known protobuf types.
	switch typeDesc {
	case anyValueType:
		json, err := l.ConvertToNative(jsonListValueType)
		if err != nil {
			return nil, err
		}
		return anypb.New(json.(proto.Message))
	case jsonValueType, jsonListValueType:
		jsonValues, err :=
			l.ConvertToNative(reflect.TypeOf([]*structpb.Value{}))
		if err != nil {
			return nil, err
		}
		jsonList := &structpb.ListValue{Values: jsonValues.([]*structpb.Value)}
		if typeDesc == jsonListValueType {
			return jsonList, nil
		}
		return structpb.NewListValue(jsonList), nil
	}
	// Non-list conversion.
	if typeDesc.Kind() != reflect.Slice && typeDesc.Kind() != reflect.Array {
		return nil, fmt.Errorf("type conversion error from list to '%v'", typeDesc)
	}

	// List conversion.
	// Allow the element ConvertToNative() function to determine whether conversion is possible.
	otherElemType := typeDesc.Elem()
	elemCount := l.size
	nativeList := reflect.MakeSlice(typeDesc, elemCount, elemCount)
	for i := 0; i < elemCount; i++ {
		elem := l.NativeToValue(l.get(i))
		nativeElemVal, err := elem.ConvertToNative(otherElemType)
		if err != nil {
			return nil, err
		}
		nativeList.Index(i).Set(reflect.ValueOf(nativeElemVal))
	}
	return nativeList.Interface(), nil
}

// ConvertToType implements the ref.Val interface method.
func (l *baseList) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case ListType:
		return l
	case TypeType:
		return ListType
	}
	return NewErr("type conversion error from '%s' to '%s'", ListType, typeVal)
}

// Equal implements the ref.Val interface method.
func (l *baseList) Equal(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return False
	}
	if l.Size() != otherList.Size() {
		return False
	}
	for i := IntZero; i < l.Size().(Int); i++ {
		thisElem := l.Get(i)
		otherElem := otherList.Get(i)
		elemEq := Equal(thisElem, otherElem)
		if elemEq == False {
			return False
		}
	}
	return True
}

// Get implements the traits.Indexer interface method.
func (l *baseList) Get(index ref.Val) ref.Val {
	ind, err := indexOrError(index)
	if err != nil {
		return ValOrErr(index, err.Error())
	}
	if ind < 0 || ind >= l.size {
		return NewErr("index '%d' out of range in list size '%d'", ind, l.Size())
	}
	return l.NativeToValue(l.get(ind))
}

// Iterator implements the traits.Iterable interface method.
func (l *baseList) Iterator() traits.Iterator {
	return newListIterator(l)
}

// Size implements the traits.Sizer interface method.
func (l *baseList) Size() ref.Val {
	return Int(l.size)
}

// Type implements the ref.Val interface method.
func (l *baseList) Type() ref.Type {
	return ListType
}

// Value implements the ref.Val interface method.
func (l *baseList) Value() interface{} {
	return l.value
}

// mutableList aggregates values into its internal storage. For use with internal CEL variables only.
type mutableList struct {
	*baseList
	mutableValues []ref.Val
}

// Add copies elements from the other list into the internal storage of the mutable list.
// The ref.Val returned by Add is the receiver.
func (l *mutableList) Add(other ref.Val) ref.Val {
	switch otherList := other.(type) {
	case *mutableList:
		l.mutableValues = append(l.mutableValues, otherList.mutableValues...)
		l.size += len(otherList.mutableValues)
	case traits.Lister:
		for i := IntZero; i < otherList.Size().(Int); i++ {
			l.size++
			l.mutableValues = append(l.mutableValues, otherList.Get(i))
		}
	default:
		return MaybeNoSuchOverloadErr(otherList)
	}
	return l
}

// ToImmutableList returns an immutable list based on the internal storage of the mutable list.
func (l *mutableList) ToImmutableList() traits.Lister {
	// The reference to internal state is guaranteed to be safe as this call is only performed
	// when mutations have been completed.
	return NewRefValList(l.TypeAdapter, l.mutableValues)
}

// concatList combines two list implementations together into a view.
// The `ref.TypeAdapter` enables native type to CEL type conversions.
type concatList struct {
	ref.TypeAdapter
	value    interface{}
	prevList traits.Lister
	nextList traits.Lister
}

// Add implements the traits.Adder interface method.
func (l *concatList) Add(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	if l.Size() == IntZero {
		return other
	}
	if otherList.Size() == IntZero {
		return l
	}
	return &concatList{
		TypeAdapter: l.TypeAdapter,
		prevList:    l,
		nextList:    otherList}
}

// Contains implements the traits.Container interface method.
func (l *concatList) Contains(elem ref.Val) ref.Val {
	// The concat list relies on the IsErrorOrUnknown checks against the input element to be
	// performed by the `prevList` and/or `nextList`.
	prev := l.prevList.Contains(elem)
	// Short-circuit the return if the elem was found in the prev list.
	if prev == True {
		return prev
	}
	// Return if the elem was found in the next list.
	next := l.nextList.Contains(elem)
	if next == True {
		return next
	}
	// Handle the case where an error or unknown was encountered before checking next.
	if IsUnknownOrError(prev) {
		return prev
	}
	// Otherwise, rely on the next value as the representative result.
	return next
}

// ConvertToNative implements the ref.Val interface method.
func (l *concatList) ConvertToNative(typeDesc reflect.Type) (interface{}, error) {
	combined := NewDynamicList(l.TypeAdapter, l.Value().([]interface{}))
	return combined.ConvertToNative(typeDesc)
}

// ConvertToType implements the ref.Val interface method.
func (l *concatList) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case ListType:
		return l
	case TypeType:
		return ListType
	}
	return NewErr("type conversion error from '%s' to '%s'", ListType, typeVal)
}

// Equal implements the ref.Val interface method.
func (l *concatList) Equal(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return False
	}
	if l.Size() != otherList.Size() {
		return False
	}
	var maybeErr ref.Val
	for i := IntZero; i < l.Size().(Int); i++ {
		thisElem := l.Get(i)
		otherElem := otherList.Get(i)
		elemEq := Equal(thisElem, otherElem)
		if elemEq == False {
			return False
		}
		if maybeErr == nil && IsUnknownOrError(elemEq) {
			maybeErr = elemEq
		}
	}
	if maybeErr != nil {
		return maybeErr
	}
	return True
}

// Get implements the traits.Indexer interface method.
func (l *concatList) Get(index ref.Val) ref.Val {
	ind, err := indexOrError(index)
	if err != nil {
		return ValOrErr(index, err.Error())
	}
	i := Int(ind)
	if i < l.prevList.Size().(Int) {
		return l.prevList.Get(i)
	}
	offset := i - l.prevList.Size().(Int)
	return l.nextList.Get(offset)
}

// Iterator implements the traits.Iterable interface method.
func (l *concatList) Iterator() traits.Iterator {
	return newListIterator(l)
}

// Size implements the traits.Sizer interface method.
func (l *concatList) Size() ref.Val {
	return l.prevList.Size().(Int).Add(l.nextList.Size())
}

// Type implements the ref.Val interface method.
func (l *concatList) Type() ref.Type {
	return ListType
}

// Value implements the ref.Val interface method.
func (l *concatList) Value() interface{} {
	if l.value == nil {
		merged := make([]interface{}, l.Size().(Int))
		prevLen := l.prevList.Size().(Int)
		for i := Int(0); i < prevLen; i++ {
			merged[i] = l.prevList.Get(i).Value()
		}
		nextLen := l.nextList.Size().(Int)
		for j := Int(0); j < nextLen; j++ {
			merged[prevLen+j] = l.nextList.Get(j).Value()
		}
		l.value = merged
	}
	return l.value
}

func newListIterator(listValue traits.Lister) traits.Iterator {
	return &listIterator{
		listValue: listValue,
		len:       listValue.Size().(Int),
	}
}

type listIterator struct {
	*baseIterator
	listValue traits.Lister
	cursor    Int
	len       Int
}

// HasNext implements the traits.Iterator interface method.
func (it *listIterator) HasNext() ref.Val {
	return Bool(it.cursor < it.len)
}

// Next implements the traits.Iterator interface method.
func (it *listIterator) Next() ref.Val {
	if it.HasNext() == True {
		index := it.cursor
		it.cursor++
		return it.listValue.Get(index)
	}
	return nil
}

func indexOrError(index ref.Val) (int, error) {
	switch iv := index.(type) {
	case Int:
		return int(iv), nil
	case Double:
		if ik, ok := doubleToInt64Lossless(float64(iv)); ok {
			return int(ik), nil
		}
		return -1, fmt.Errorf("unsupported index value %v in list", index)
	case Uint:
		if ik, ok := uint64ToInt64Lossless(uint64(iv)); ok {
			return int(ik), nil
		}
		return -1, fmt.Errorf("unsupported index value %v in list", index)
	default:
		return -1, fmt.Errorf("unsupported index type '%s' in list", index.Type())
	}
}
