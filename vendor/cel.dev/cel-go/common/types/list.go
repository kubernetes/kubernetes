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
	"bytes"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"unsafe"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"

	"cel.dev/cel-go/common/types/ref"
	"cel.dev/cel-go/common/types/traits"

	anypb "google.golang.org/protobuf/types/known/anypb"
	structpb "google.golang.org/protobuf/types/known/structpb"
)

// NewDynamicList returns a traits.Lister with heterogenous elements.
// value should be an array of "native" types, i.e. any type that
// NativeToValue() can convert to a ref.Val.
func NewDynamicList(adapter Adapter, value any) traits.Lister {
	if value == nil {
		return NewList[any](adapter, nil)
	}
	switch v := value.(type) {
	case []string:
		return NewList(adapter, v)
	case []ref.Val:
		return NewList(adapter, v)
	case []int:
		return NewList(adapter, v)
	case []int64:
		return NewList(adapter, v)
	case []int32:
		return NewList(adapter, v)
	case []uint:
		return NewList(adapter, v)
	case []uint64:
		return NewList(adapter, v)
	case []uint32:
		return NewList(adapter, v)
	case []float64:
		return NewList(adapter, v)
	case []float32:
		return NewList(adapter, v)
	case []bool:
		return NewList(adapter, v)
	case []any:
		return NewList(adapter, v)
	case [][]byte:
		return NewList(adapter, v)
	}
	rt := reflect.TypeOf(value)
	if meta := getDynamicSliceMeta(rt); meta.supported {
		sh := (*unsafeSlice)(anyDataPtr(value))
		isNil := sh == nil || sh.Data == nil
		qualifyRaw := canQualifyRawVal(adapter, meta.sliceType.Elem(), meta.qualifyRawVal)
		if meta.isPtrElem {
			var elems []unsafe.Pointer
			if sh != nil && sh.Len > 0 {
				elems = unsafe.Slice((*unsafe.Pointer)(sh.Data), sh.Len)
			}
			return &sliceList[unsafe.Pointer]{
				Adapter:       adapter,
				elems:         elems,
				elemTypePtr:   meta.elemTypePtr,
				meta:          meta,
				qualifyRawVal: qualifyRaw,
				isNilSlice:    isNil,
			}
		}
		var elems []byte
		if sh != nil && sh.Len > 0 {
			elems = unsafe.Slice((*byte)(sh.Data), sh.Len*int(meta.elemStride))
		}
		return &sliceList[byte]{
			Adapter:       adapter,
			elems:         elems,
			elemTypePtr:   meta.elemTypePtr,
			meta:          meta,
			qualifyRawVal: qualifyRaw,
			isNilSlice:    isNil,
		}
	}
	refValue := reflect.ValueOf(value)
	return &baseList{
		Adapter: adapter,
		value:   value,
		size:    refValue.Len(),
		get: func(i int) any {
			return refValue.Index(i).Interface()
		},
	}
}

// NewList returns a traits.Lister backed by a typed Go slice []T.
func NewList[T any](adapter Adapter, elems []T) traits.Lister {
	return &sliceList[T]{
		Adapter:       adapter,
		elems:         elems,
		elemTypePtr:   elemTypePtrFor[T](),
		qualifyRawVal: isQualifyRawStructFor[T](adapter),
	}
}

// NewStringList returns a traits.Lister containing only strings.
func NewStringList(adapter Adapter, elems []string) traits.Lister {
	return NewList(adapter, elems)
}

// NewRefValList returns a traits.Lister with ref.Val elements.
//
// This type specialization is used with list literals within CEL expressions.
func NewRefValList(adapter Adapter, elems []ref.Val) traits.Lister {
	return &sliceList[ref.Val]{
		Adapter: adapter,
		elems:   elems,
	}
}

// NewProtoList returns a traits.Lister based on a pb.List instance.
func NewProtoList(adapter Adapter, list protoreflect.List) traits.Lister {
	return &baseList{
		Adapter: adapter,
		value:   list,
		size:    list.Len(),
		get:     func(i int) any { return list.Get(i).Interface() },
	}
}

// NewJSONList returns a traits.Lister based on structpb.ListValue instance.
func NewJSONList(adapter Adapter, l *structpb.ListValue) traits.Lister {
	vals := l.GetValues()
	return &baseList{
		Adapter: adapter,
		value:   l,
		size:    len(vals),
		get:     func(i int) any { return vals[i] },
	}
}

// NewMutableList creates a new mutable list whose internal state can be modified.
func NewMutableList(adapter Adapter) traits.MutableLister {
	return &mutableList{
		sliceList: &sliceList[ref.Val]{
			Adapter: adapter,
		},
	}
}

// MaybeSliceList attempts to sub-slice a traits.Lister using its underlying native slice representation
// via reflection, preserving the original element type. If the underlying representation is not a slice
// or the indices are invalid, it returns (nil, false).
func MaybeSliceList(adapter Adapter, list traits.Lister, start, end int) (traits.Lister, bool) {
	if list == nil {
		return nil, false
	}
	if _, isConcat := list.(*concatList); isConcat {
		return nil, false
	}
	val := list.Value()
	if val == nil {
		return nil, false
	}
	rv := reflect.ValueOf(val)
	if rv.Kind() != reflect.Slice {
		return nil, false
	}
	if start < 0 || end < 0 || start > end || end > rv.Len() {
		return nil, false
	}
	adapter = maybeCustomAdapter(adapter, list)
	sub := rv.Slice(start, end).Interface()
	return NewDynamicList(adapter, sub), true
}

// MaybeReverseList attempts to reverse a traits.Lister using its underlying native slice representation,
// preserving the original element type. If the underlying representation is not a slice, it returns (nil, false).
func MaybeReverseList(adapter Adapter, list traits.Lister) (traits.Lister, bool) {
	if list == nil {
		return nil, false
	}
	if _, isConcat := list.(*concatList); isConcat {
		return nil, false
	}
	val := list.Value()
	if val == nil {
		return nil, false
	}
	rv := reflect.ValueOf(val)
	if rv.Kind() != reflect.Slice {
		return nil, false
	}
	n := rv.Len()
	if n <= 1 {
		return list, true
	}
	adapter = maybeCustomAdapter(adapter, list)
	out := reflect.MakeSlice(rv.Type(), n, n)
	for i := range n {
		out.Index(n - 1 - i).Set(rv.Index(i))
	}
	return NewDynamicList(adapter, out.Interface()), true
}

type listAdapter interface {
	listAdapter() Adapter
}

func maybeCustomAdapter(adapter Adapter, list traits.Lister) Adapter {
	if la, ok := list.(listAdapter); ok {
		if listAdapt := la.listAdapter(); listAdapt != nil && !isStandardAdapter(listAdapt) {
			return listAdapt
		}
	}
	return adapter
}

// baseList points to a list containing elements of any type.
// The `value` is an array of native values, and refValue is its reflection object.
// The `Adapter` enables native type to CEL type conversions.
type baseList struct {
	Adapter
	value any
	size  int
	// aggSize memoizes the aggregate size computed by the first completed sizing of this
	// list. Accessed atomically since immutable lists may be shared across concurrent
	// evaluations; zero means not yet computed. See the SizeCalculator documentation for
	// the memoization contract.
	aggSize uint32
	get     func(int) any
}

func (l *baseList) listAdapter() Adapter {
	return l.Adapter
}

// Add implements the traits.Adder interface method.
func (l *baseList) Add(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	return newConcatList(l.Adapter, l, otherList)
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
func (l *baseList) ConvertToNative(typeDesc reflect.Type) (any, error) {
	return convertListToNative(l, l.value, typeDesc)
}

func convertListToNative(l traits.Lister, val any, typeDesc reflect.Type) (any, error) {
	if typeDesc == reflect.TypeFor[any]() {
		typeDesc = reflect.TypeFor[[]any]()
	}
	// If the underlying list value is assignable to the reflected type return it.
	if val != nil && reflect.TypeOf(val).AssignableTo(typeDesc) {
		return val, nil
	}
	// If the list wrapper is assignable to the desired type return it.
	if reflect.TypeOf(l).AssignableTo(typeDesc) {
		return l, nil
	}
	// Attempt to convert the list to a set of well known protobuf types.
	switch typeDesc {
	case anyValueType:
		json, err := l.ConvertToNative(JSONListType)
		if err != nil {
			return nil, err
		}
		return anypb.New(json.(proto.Message))
	case JSONValueType, JSONListType:
		jsonValues, err :=
			l.ConvertToNative(reflect.TypeFor[[]*structpb.Value]())
		if err != nil {
			return nil, err
		}
		jsonList := &structpb.ListValue{Values: jsonValues.([]*structpb.Value)}
		if typeDesc == JSONListType {
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
	elemCount := int(l.Size().(Int))
	var nativeList reflect.Value
	if typeDesc.Kind() == reflect.Array {
		nativeList = reflect.New(reflect.ArrayOf(elemCount, typeDesc)).Elem().Index(0)
	} else {
		nativeList = reflect.MakeSlice(typeDesc, elemCount, elemCount)
	}
	for i := range elemCount {
		elem := l.Get(Int(i))
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
	ind, err := GetOrError(index)
	if err != nil {
		return ValOrErr(index, "%v", err)
	}
	if ind < 0 || ind >= l.size {
		return NewErr("index '%d' out of range in list size '%d'", ind, l.Size())
	}
	return l.NativeToValue(l.get(ind))
}

// IsZeroValue returns true if the list is empty.
func (l *baseList) IsZeroValue() bool {
	return l.size == 0
}

// Fold calls the FoldEntry method for each (index, value) pair in the list.
func (l *baseList) Fold(f traits.Folder) {
	get := l.get
	if !isStandardAdapter(l.Adapter) {
		get = func(i int) any {
			return l.Adapter.NativeToValue(l.get(i))
		}
	}
	for i := 0; i < l.size; i++ {
		if !f.FoldEntry(i, get(i)) {
			break
		}
	}
}

// Iterator implements the traits.Iterable interface method.
func (l *baseList) Iterator() traits.Iterator {
	return newListIterator(l)
}

// Size implements the traits.Sizer interface method.
func (l *baseList) Size() ref.Val {
	return Int(l.size)
}

// AggregateSize implements the AggregateSizeVisitor interface method.
func (l *baseList) AggregateSize(sizer AggregateSizer) uint32 {
	if sz := atomic.LoadUint32(&l.aggSize); sz != 0 {
		return sz
	}
	var total uint32
	if l.value != nil {
		if t, ok := getSliceElementsAggregateSize(sizer, l.value); ok {
			total = t
		}
	}
	if total == 0 {
		total = uint32(1)
		for i := range l.size {
			total = safeAddUint32(total, sizer.AggregateSize(l.get(i)))
		}
	}
	if cacheableAggregateSize(sizer) {
		atomic.StoreUint32(&l.aggSize, total)
	}
	return total
}

// Type implements the ref.Val interface method.
func (l *baseList) Type() ref.Type {
	return ListType
}

// Value implements the ref.Val interface method.
func (l *baseList) Value() any {
	return l.value
}

// String converts the list to a human readable string form.
func (l *baseList) String() string {
	var sb strings.Builder
	sb.WriteString("[")
	for i := 0; i < l.size; i++ {
		sb.WriteString(fmt.Sprintf("%v", l.get(i)))
		if i != l.size-1 {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

func formatList(l traits.Lister, sb *strings.Builder) {
	sb.WriteString("[")
	n, _ := l.Size().(Int)
	for i := 0; i < int(n); i++ {
		formatTo(sb, l.Get(Int(i)))
		if i != int(n)-1 {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("]")
}

func (l *baseList) format(sb *strings.Builder) {
	formatList(l, sb)
}

// mutableList aggregates values into its internal storage. For use with internal CEL variables only.
type mutableList struct {
	*sliceList[ref.Val]
}

// Add copies elements from the other list into the internal storage of the mutable list.
// The ref.Val returned by Add is the receiver.
func (l *mutableList) Add(other ref.Val) ref.Val {
	switch otherList := other.(type) {
	case *mutableList:
		l.elems = append(l.elems, otherList.elems...)
		atomic.StoreUint32(&l.aggSize, 0)
	case traits.Lister:
		for i := IntZero; i < otherList.Size().(Int); i++ {
			l.elems = append(l.elems, otherList.Get(i))
		}
		atomic.StoreUint32(&l.aggSize, 0)
	default:
		return MaybeNoSuchOverloadErr(otherList)
	}
	return l
}

// ToImmutableList returns an immutable list based on the internal storage of the mutable list.
// It transfers ownership of the underlying buffer to the returned immutable list and clears
// internal elements to prevent subsequent modifications from aliasing.
func (l *mutableList) ToImmutableList() traits.Lister {
	elems := l.elems
	l.elems = nil
	atomic.StoreUint32(&l.aggSize, 0)
	return NewRefValList(l.Adapter, elems)
}

// concatList combines two list implementations together into a view.
// The `Adapter` enables native type to CEL type conversions.
type concatList struct {
	Adapter
	value      any
	valueOnce  sync.Once
	prevList   traits.Lister
	nextList   traits.Lister
	cachedSize ref.Val
	aggSize    uint32
}

func newConcatList(adapter Adapter, prevList, nextList traits.Lister) ref.Val {
	prevSize := prevList.Size().(Int)
	nextSize := nextList.Size().(Int)
	if prevSize == IntZero {
		return nextList.(ref.Val)
	}
	if nextSize == IntZero {
		return prevList.(ref.Val)
	}
	return &concatList{
		Adapter:    adapter,
		prevList:   prevList,
		nextList:   nextList,
		cachedSize: prevSize.Add(nextSize),
	}
}

// Add implements the traits.Adder interface method.
func (l *concatList) Add(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	return newConcatList(l.Adapter, l, otherList)
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
func (l *concatList) ConvertToNative(typeDesc reflect.Type) (any, error) {
	combined := NewDynamicList(l.Adapter, l.Value())
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
	ind, err := GetOrError(index)
	if err != nil {
		return ValOrErr(index, "%v", err)
	}
	i := Int(ind)
	if i < l.prevList.Size().(Int) {
		return l.prevList.Get(i)
	}
	offset := i - l.prevList.Size().(Int)
	return l.nextList.Get(offset)
}

// IsZeroValue returns true if the list is empty.
func (l *concatList) IsZeroValue() bool {
	return l.Size().(Int) == 0
}

// Fold calls the FoldEntry method for each (index, value) pair in the list.
func (l *concatList) Fold(f traits.Folder) {
	for i := Int(0); i < l.Size().(Int); i++ {
		if !f.FoldEntry(i, l.Get(i)) {
			break
		}
	}
}

// Iterator implements the traits.Iterable interface method.
func (l *concatList) Iterator() traits.Iterator {
	return newListIterator(l)
}

// Size implements the traits.Sizer interface method.
func (l *concatList) Size() ref.Val {
	return l.cachedSize
}

// AggregateSize implements the AggregateSizeVisitor interface method.
func (l *concatList) AggregateSize(sizer AggregateSizer) uint32 {
	if sz := atomic.LoadUint32(&l.aggSize); sz != 0 {
		return sz
	}
	total := safeAddUint32(sizer.AggregateSize(l.prevList), sizer.AggregateSize(l.nextList))
	if cacheableAggregateSize(sizer) {
		atomic.StoreUint32(&l.aggSize, total)
	}
	return total
}

// String converts the concatenated list to a human-readable string.
func (l *concatList) String() string {
	var sb strings.Builder
	sb.WriteString("[")
	for i := Int(0); i < l.Size().(Int); i++ {
		sb.WriteString(fmt.Sprintf("%v", l.Get(i)))
		if i != l.Size().(Int)-1 {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

// Type implements the ref.Val interface method.
func (l *concatList) Type() ref.Type {
	return ListType
}

// Value implements the ref.Val interface method.
func (l *concatList) Value() any {
	l.valueOnce.Do(func() {
		merged := make([]ref.Val, l.Size().(Int))
		prevLen := l.prevList.Size().(Int)
		for i := Int(0); i < prevLen; i++ {
			merged[i] = l.prevList.Get(i)
		}
		nextLen := l.nextList.Size().(Int)
		for j := Int(0); j < nextLen; j++ {
			merged[prevLen+j] = l.nextList.Get(j)
		}
		l.value = merged
	})
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

// GetOrError converts an input index value into either a lossless integer index or an error.
func GetOrError(index ref.Val) (int, error) {
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

// ToFoldableList will create a Foldable version of a list suitable for key-value pair iteration.
//
// For values which are already Foldable, this call is a no-op. For all other values, the fold is
// driven via the Size() and Get() calls which means that the folding will function, but take a
// performance hit.
func ToFoldableList(l traits.Lister) traits.Foldable {
	if f, ok := l.(traits.Foldable); ok {
		return f
	}
	return interopFoldableList{Lister: l}
}

type interopFoldableList struct {
	traits.Lister
}

// Fold implements the traits.Foldable interface method and performs an iteration over the
// range of elements of the list.
func (l interopFoldableList) Fold(f traits.Folder) {
	sz := l.Size().(Int)
	for i := Int(0); i < sz; i++ {
		if !f.FoldEntry(i, l.Get(i)) {
			break
		}
	}
}

type sliceList[T any] struct {
	Adapter
	elems         []T
	elemTypePtr   unsafe.Pointer
	meta          *dynamicSliceMeta
	aggSize       uint32
	qualifyRawVal bool
	isNilSlice    bool
}

func (l *sliceList[T]) listAdapter() Adapter {
	return l.Adapter
}

func (l *sliceList[T]) len() int {
	if l.meta != nil && l.meta.elemStride > 0 {
		return len(l.elems) / int(l.meta.elemStride)
	}
	return len(l.elems)
}

func (l *sliceList[T]) nativeSliceValue() any {
	if l.meta != nil {
		n := l.len()
		var data unsafe.Pointer
		if n > 0 {
			data = unsafe.Pointer(unsafe.SliceData(l.elems))
		} else if !l.isNilSlice {
			data = unsafe.Pointer(&emptySliceData)
		}
		hdr := &unsafeSlice{
			Data: data,
			Len:  n,
			Cap:  n,
		}
		return packAny(l.meta.sliceTypePtr, unsafe.Pointer(hdr))
	}
	return l.elems
}

func (l *sliceList[T]) rawElemAt(ind int) (any, bool) {
	switch s := any(l).(type) {
	case *sliceList[unsafe.Pointer]:
		if s.elemTypePtr != nil {
			ptr := s.elems[ind]
			if ptr == nil {
				return NullValue, true
			}
			return packAny(s.elemTypePtr, ptr), true
		}
	case *sliceList[uint8]:
		if s.meta != nil && s.meta.elemStride > 0 && s.elemTypePtr != nil {
			addr := unsafe.Pointer(&s.elems[uintptr(ind)*s.meta.elemStride])
			if !s.qualifyRawVal {
				return reflect.NewAt(s.meta.sliceType.Elem(), addr).Elem().Interface(), true
			}
			return packAny(s.elemTypePtr, addr), true
		}
	}
	if l.qualifyRawVal && l.elemTypePtr != nil {
		return packAny(l.elemTypePtr, unsafe.Pointer(&l.elems[ind])), true
	}
	return nil, false
}

func (l *sliceList[T]) getElem(ind int) ref.Val {
	switch s := any(l).(type) {
	case *sliceList[string]:
		return String(s.elems[ind])
	case *sliceList[ref.Val]:
		return s.elems[ind]
	case *sliceList[int]:
		return Int(s.elems[ind])
	case *sliceList[int64]:
		return Int(s.elems[ind])
	case *sliceList[int32]:
		return Int(s.elems[ind])
	case *sliceList[uint]:
		return Uint(s.elems[ind])
	case *sliceList[uint64]:
		return Uint(s.elems[ind])
	case *sliceList[uint32]:
		return Uint(s.elems[ind])
	case *sliceList[uint8]:
		if s.meta == nil {
			return Uint(s.elems[ind])
		}
	case *sliceList[float64]:
		return Double(s.elems[ind])
	case *sliceList[float32]:
		return Double(s.elems[ind])
	case *sliceList[bool]:
		if s.elems[ind] {
			return True
		}
		return False
	case *sliceList[[]byte]:
		return Bytes(s.elems[ind])
	}
	if raw, ok := l.rawElemAt(ind); ok {
		return l.Adapter.NativeToValue(raw)
	}
	return l.Adapter.NativeToValue(l.elems[ind])
}

// GetInt64Index returns the element at ind for fast index qualification in the interpreter.
// For non-proto Go struct elements, it returns the unboxed struct pointer or zero-copy struct
// emptyInterface so subsequent field accesses can read offsets without allocating a nativeObj.
func (l *sliceList[T]) GetInt64Index(ind int64) (any, bool) {
	n := l.len()
	if ind < 0 || ind >= int64(n) {
		return nil, false
	}
	i := int(ind)
	if !l.qualifyRawVal {
		return l.getElem(i), true
	}
	if raw, ok := l.rawElemAt(i); ok {
		return raw, true
	}
	return l.elems[i], true
}

func (l *sliceList[T]) Add(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return MaybeNoSuchOverloadErr(other)
	}
	return newConcatList(l.Adapter, l, otherList)
}

func (l *sliceList[T]) Contains(elem ref.Val) ref.Val {
	switch s := any(l).(type) {
	case *sliceList[string]:
		if strVal, ok := elem.(String); ok {
			str := string(strVal)
			for _, e := range s.elems {
				if e == str {
					return True
				}
			}
			return False
		}
	case *sliceList[int64]:
		if intVal, ok := elem.(Int); ok {
			iv := int64(intVal)
			for _, e := range s.elems {
				if e == iv {
					return True
				}
			}
			return False
		}
	case *sliceList[int]:
		if intVal, ok := elem.(Int); ok {
			iv := int(intVal)
			for _, e := range s.elems {
				if e == iv {
					return True
				}
			}
			return False
		}
	case *sliceList[int32]:
		if intVal, ok := elem.(Int); ok {
			iv := int32(intVal)
			if int64(iv) == int64(intVal) {
				for _, e := range s.elems {
					if e == iv {
						return True
					}
				}
			}
			return False
		}
	case *sliceList[uint64]:
		if uintVal, ok := elem.(Uint); ok {
			uv := uint64(uintVal)
			for _, e := range s.elems {
				if e == uv {
					return True
				}
			}
			return False
		}
	case *sliceList[uint]:
		if uintVal, ok := elem.(Uint); ok {
			uv := uint(uintVal)
			for _, e := range s.elems {
				if e == uv {
					return True
				}
			}
			return False
		}
	case *sliceList[uint32]:
		if uintVal, ok := elem.(Uint); ok {
			uv := uint32(uintVal)
			if uint64(uv) == uint64(uintVal) {
				for _, e := range s.elems {
					if e == uv {
						return True
					}
				}
			}
			return False
		}
	case *sliceList[float64]:
		if dblVal, ok := elem.(Double); ok {
			dv := float64(dblVal)
			for _, e := range s.elems {
				if e == dv {
					return True
				}
			}
			return False
		}
	case *sliceList[bool]:
		if boolVal, ok := elem.(Bool); ok {
			bv := bool(boolVal)
			for _, e := range s.elems {
				if e == bv {
					return True
				}
			}
			return False
		}
	}
	n := l.len()
	for i := 0; i < n; i++ {
		cmp := elem.Equal(l.getElem(i))
		if b, ok := cmp.(Bool); ok && b == True {
			return True
		}
	}
	return False
}

func (l *sliceList[T]) ConvertToNative(typeDesc reflect.Type) (any, error) {
	if l.meta == nil && typeDesc == reflect.TypeFor[[]T]() {
		return l.elems, nil
	}
	val := l.nativeSliceValue()
	if l.meta != nil && l.meta.sliceType == typeDesc {
		return val, nil
	}
	return convertListToNative(l, val, typeDesc)
}

func (l *sliceList[T]) ConvertToType(typeVal ref.Type) ref.Val {
	switch typeVal {
	case ListType:
		return l
	case TypeType:
		return ListType
	}
	return NewErr("type conversion error from '%s' to '%s'", ListType, typeVal)
}

func (l *sliceList[T]) Equal(other ref.Val) ref.Val {
	otherList, ok := other.(traits.Lister)
	if !ok {
		return False
	}
	if l.Size() != otherList.Size() {
		return False
	}
	n := l.len()
	if otherSlice, ok := other.(*sliceList[T]); ok && l.meta == otherSlice.meta {
		switch s := any(l).(type) {
		case *sliceList[string]:
			o := any(otherSlice).(*sliceList[string])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[int64]:
			o := any(otherSlice).(*sliceList[int64])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[int]:
			o := any(otherSlice).(*sliceList[int])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[int32]:
			o := any(otherSlice).(*sliceList[int32])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[uint64]:
			o := any(otherSlice).(*sliceList[uint64])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[uint]:
			o := any(otherSlice).(*sliceList[uint])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[uint32]:
			o := any(otherSlice).(*sliceList[uint32])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[float64]:
			o := any(otherSlice).(*sliceList[float64])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[float32]:
			o := any(otherSlice).(*sliceList[float32])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[bool]:
			o := any(otherSlice).(*sliceList[bool])
			for i := range n {
				if s.elems[i] != o.elems[i] {
					return False
				}
			}
			return True
		case *sliceList[[]byte]:
			o := any(otherSlice).(*sliceList[[]byte])
			for i := range n {
				if !bytes.Equal(s.elems[i], o.elems[i]) {
					return False
				}
			}
			return True
		}
		for i := range n {
			if Equal(l.getElem(i), otherSlice.getElem(i)) == False {
				return False
			}
		}
		return True
	}
	for i := range n {
		thisElem := l.getElem(i)
		otherElem := otherList.Get(Int(i))
		if Equal(thisElem, otherElem) == False {
			return False
		}
	}
	return True
}

func (l *sliceList[T]) Get(index ref.Val) ref.Val {
	ind, err := GetOrError(index)
	if err != nil {
		return ValOrErr(index, "%v", err)
	}
	n := l.len()
	if ind < 0 || ind >= n {
		return NewErr("index '%d' out of range in list size '%d'", ind, n)
	}
	return l.getElem(ind)
}

func (l *sliceList[T]) IsZeroValue() bool {
	return l.len() == 0
}

func (l *sliceList[T]) foldElems(f traits.Folder) {
	for i := range l.len() {
		if !f.FoldEntry(i, l.getElem(i)) {
			break
		}
	}
}

func (l *sliceList[T]) Fold(f traits.Folder) {
	switch s := any(l).(type) {
	case *sliceList[string]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, String(e)) {
				break
			}
		}
	case *sliceList[ref.Val]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, e) {
				break
			}
		}
	case *sliceList[int]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Int(e)) {
				break
			}
		}
	case *sliceList[int64]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Int(e)) {
				break
			}
		}
	case *sliceList[int32]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Int(e)) {
				break
			}
		}
	case *sliceList[uint]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Uint(e)) {
				break
			}
		}
	case *sliceList[uint64]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Uint(e)) {
				break
			}
		}
	case *sliceList[uint32]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Uint(e)) {
				break
			}
		}
	case *sliceList[uint8]:
		if s.meta != nil && s.meta.elemStride > 0 && s.elemTypePtr != nil {
			if !s.qualifyRawVal {
				s.foldElems(f)
				return
			}
			stride := s.meta.elemStride
			n := len(s.elems) / int(stride)
			for i := 0; i < n; i++ {
				if !f.FoldEntry(i, packAny(s.elemTypePtr, unsafe.Pointer(&s.elems[uintptr(i)*stride]))) {
					break
				}
			}
			return
		}
		for i, e := range s.elems {
			if !f.FoldEntry(i, Uint(e)) {
				break
			}
		}
	case *sliceList[float64]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Double(e)) {
				break
			}
		}
	case *sliceList[float32]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Double(e)) {
				break
			}
		}
	case *sliceList[bool]:
		for i, e := range s.elems {
			val := False
			if e {
				val = True
			}
			if !f.FoldEntry(i, val) {
				break
			}
		}
	case *sliceList[[]byte]:
		for i, e := range s.elems {
			if !f.FoldEntry(i, Bytes(e)) {
				break
			}
		}
	case *sliceList[unsafe.Pointer]:
		if s.elemTypePtr != nil {
			if !s.qualifyRawVal {
				s.foldElems(f)
				return
			}
			for i, ptr := range s.elems {
				if ptr == nil {
					if !f.FoldEntry(i, NullValue) {
						break
					}
					continue
				}
				if !f.FoldEntry(i, packAny(s.elemTypePtr, ptr)) {
					break
				}
			}
			return
		}
	default:
		if l.qualifyRawVal && l.elemTypePtr != nil {
			for i := range l.elems {
				if !f.FoldEntry(i, packAny(l.elemTypePtr, unsafe.Pointer(&l.elems[i]))) {
					break
				}
			}
			return
		}
		if !l.qualifyRawVal && (!isStandardAdapter(l.Adapter) || l.elemTypePtr != nil) {
			l.foldElems(f)
			return
		}
		for i, e := range l.elems {
			if !f.FoldEntry(i, e) {
				break
			}
		}
	}
}

func (l *sliceList[T]) Iterator() traits.Iterator {
	return &sliceListIterator[T]{
		list: l,
		len:  l.len(),
	}
}

func (l *sliceList[T]) Size() ref.Val {
	return Int(l.len())
}

func (l *sliceList[T]) AggregateSize(sizer AggregateSizer) uint32 {
	if sz := atomic.LoadUint32(&l.aggSize); sz != 0 {
		return sz
	}
	var total uint32
	if l.meta == nil && l.elemTypePtr == nil {
		if t, ok := getSliceElementsAggregateSize(sizer, l.elems); ok {
			total = t
		}
	}
	if total == 0 {
		total = uint32(1)
		n := l.len()
		for i := 0; i < n; i++ {
			total = safeAddUint32(total, sizer.AggregateSize(l.getElem(i)))
		}
	}
	if cacheableAggregateSize(sizer) {
		atomic.StoreUint32(&l.aggSize, total)
	}
	return total
}

func (l *sliceList[T]) Type() ref.Type {
	return ListType
}

func (l *sliceList[T]) Value() any {
	return l.nativeSliceValue()
}

func (l *sliceList[T]) String() string {
	var sb strings.Builder
	sb.WriteString("[")
	n := l.len()
	for i := 0; i < n; i++ {
		if l.meta != nil {
			sb.WriteString(fmt.Sprintf("%v", l.getElem(i).Value()))
		} else {
			sb.WriteString(fmt.Sprintf("%v", l.elems[i]))
		}
		if i != n-1 {
			sb.WriteString(", ")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

func (l *sliceList[T]) format(sb *strings.Builder) {
	formatList(l, sb)
}

type sliceListIterator[T any] struct {
	*baseIterator
	list   *sliceList[T]
	cursor int
	len    int
}

func (it *sliceListIterator[T]) HasNext() ref.Val {
	if it.cursor < it.len {
		return True
	}
	return False
}

func (it *sliceListIterator[T]) Next() ref.Val {
	if it.cursor < it.len {
		ind := it.cursor
		it.cursor++
		return it.list.getElem(ind)
	}
	return nil
}

var emptySliceData byte
