// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"bytes"
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync"
)

type encodeFuncs struct {
	ef  encodeFunc
	ief isEmptyFunc
}

var (
	decodingStructTypeCache sync.Map // map[reflect.Type]*decodingStructType
	encodingStructTypeCache sync.Map // map[reflect.Type]*encodingStructType
	encodeFuncCache         sync.Map // map[reflect.Type]encodeFuncs
	typeInfoCache           sync.Map // map[reflect.Type]*typeInfo
)

type specialType int

const (
	specialTypeNone specialType = iota
	specialTypeUnmarshalerIface
	specialTypeEmptyIface
	specialTypeIface
	specialTypeTag
	specialTypeTime
)

type typeInfo struct {
	elemTypeInfo *typeInfo
	keyTypeInfo  *typeInfo
	typ          reflect.Type
	kind         reflect.Kind
	nonPtrType   reflect.Type
	nonPtrKind   reflect.Kind
	spclType     specialType
}

func newTypeInfo(t reflect.Type) *typeInfo {
	tInfo := typeInfo{typ: t, kind: t.Kind()}

	for t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	k := t.Kind()

	tInfo.nonPtrType = t
	tInfo.nonPtrKind = k

	if k == reflect.Interface {
		if t.NumMethod() == 0 {
			tInfo.spclType = specialTypeEmptyIface
		} else {
			tInfo.spclType = specialTypeIface
		}
	} else if t == typeTag {
		tInfo.spclType = specialTypeTag
	} else if t == typeTime {
		tInfo.spclType = specialTypeTime
	} else if reflect.PtrTo(t).Implements(typeUnmarshaler) {
		tInfo.spclType = specialTypeUnmarshalerIface
	}

	switch k {
	case reflect.Array, reflect.Slice:
		tInfo.elemTypeInfo = getTypeInfo(t.Elem())
	case reflect.Map:
		tInfo.keyTypeInfo = getTypeInfo(t.Key())
		tInfo.elemTypeInfo = getTypeInfo(t.Elem())
	}

	return &tInfo
}

type decodingStructType struct {
	fields             fields
	fieldIndicesByName map[string]int
	err                error
	toArray            bool
}

// The stdlib errors.Join was introduced in Go 1.20, and we still support Go 1.17, so instead,
// here's a very basic implementation of an aggregated error.
type multierror []error

func (m multierror) Error() string {
	var sb strings.Builder
	for i, err := range m {
		sb.WriteString(err.Error())
		if i < len(m)-1 {
			sb.WriteString(", ")
		}
	}
	return sb.String()
}

func getDecodingStructType(t reflect.Type) *decodingStructType {
	if v, _ := decodingStructTypeCache.Load(t); v != nil {
		return v.(*decodingStructType)
	}

	flds, structOptions := getFields(t)

	toArray := hasToArrayOption(structOptions)

	var errs []error
	for i := 0; i < len(flds); i++ {
		if flds[i].keyAsInt {
			nameAsInt, numErr := strconv.Atoi(flds[i].name)
			if numErr != nil {
				errs = append(errs, errors.New("cbor: failed to parse field name \""+flds[i].name+"\" to int ("+numErr.Error()+")"))
				break
			}
			flds[i].nameAsInt = int64(nameAsInt)
		}

		flds[i].typInfo = getTypeInfo(flds[i].typ)
	}

	fieldIndicesByName := make(map[string]int, len(flds))
	for i, fld := range flds {
		if _, ok := fieldIndicesByName[fld.name]; ok {
			errs = append(errs, fmt.Errorf("cbor: two or more fields of %v have the same name %q", t, fld.name))
			continue
		}
		fieldIndicesByName[fld.name] = i
	}

	var err error
	{
		var multi multierror
		for _, each := range errs {
			if each != nil {
				multi = append(multi, each)
			}
		}
		if len(multi) == 1 {
			err = multi[0]
		} else if len(multi) > 1 {
			err = multi
		}
	}

	structType := &decodingStructType{
		fields:             flds,
		fieldIndicesByName: fieldIndicesByName,
		err:                err,
		toArray:            toArray,
	}
	decodingStructTypeCache.Store(t, structType)
	return structType
}

type encodingStructType struct {
	fields             fields
	bytewiseFields     fields
	lengthFirstFields  fields
	omitEmptyFieldsIdx []int
	err                error
	toArray            bool
}

func (st *encodingStructType) getFields(em *encMode) fields {
	switch em.sort {
	case SortNone, SortFastShuffle:
		return st.fields
	case SortLengthFirst:
		return st.lengthFirstFields
	default:
		return st.bytewiseFields
	}
}

type bytewiseFieldSorter struct {
	fields fields
}

func (x *bytewiseFieldSorter) Len() int {
	return len(x.fields)
}

func (x *bytewiseFieldSorter) Swap(i, j int) {
	x.fields[i], x.fields[j] = x.fields[j], x.fields[i]
}

func (x *bytewiseFieldSorter) Less(i, j int) bool {
	return bytes.Compare(x.fields[i].cborName, x.fields[j].cborName) <= 0
}

type lengthFirstFieldSorter struct {
	fields fields
}

func (x *lengthFirstFieldSorter) Len() int {
	return len(x.fields)
}

func (x *lengthFirstFieldSorter) Swap(i, j int) {
	x.fields[i], x.fields[j] = x.fields[j], x.fields[i]
}

func (x *lengthFirstFieldSorter) Less(i, j int) bool {
	if len(x.fields[i].cborName) != len(x.fields[j].cborName) {
		return len(x.fields[i].cborName) < len(x.fields[j].cborName)
	}
	return bytes.Compare(x.fields[i].cborName, x.fields[j].cborName) <= 0
}

func getEncodingStructType(t reflect.Type) (*encodingStructType, error) {
	if v, _ := encodingStructTypeCache.Load(t); v != nil {
		structType := v.(*encodingStructType)
		return structType, structType.err
	}

	flds, structOptions := getFields(t)

	if hasToArrayOption(structOptions) {
		return getEncodingStructToArrayType(t, flds)
	}

	var err error
	var hasKeyAsInt bool
	var hasKeyAsStr bool
	var omitEmptyIdx []int
	e := getEncodeBuffer()
	for i := 0; i < len(flds); i++ {
		// Get field's encodeFunc
		flds[i].ef, flds[i].ief = getEncodeFunc(flds[i].typ)
		if flds[i].ef == nil {
			err = &UnsupportedTypeError{t}
			break
		}

		// Encode field name
		if flds[i].keyAsInt {
			nameAsInt, numErr := strconv.Atoi(flds[i].name)
			if numErr != nil {
				err = errors.New("cbor: failed to parse field name \"" + flds[i].name + "\" to int (" + numErr.Error() + ")")
				break
			}
			flds[i].nameAsInt = int64(nameAsInt)
			if nameAsInt >= 0 {
				encodeHead(e, byte(cborTypePositiveInt), uint64(nameAsInt))
			} else {
				n := nameAsInt*(-1) - 1
				encodeHead(e, byte(cborTypeNegativeInt), uint64(n))
			}
			flds[i].cborName = make([]byte, e.Len())
			copy(flds[i].cborName, e.Bytes())
			e.Reset()

			hasKeyAsInt = true
		} else {
			encodeHead(e, byte(cborTypeTextString), uint64(len(flds[i].name)))
			flds[i].cborName = make([]byte, e.Len()+len(flds[i].name))
			n := copy(flds[i].cborName, e.Bytes())
			copy(flds[i].cborName[n:], flds[i].name)
			e.Reset()

			// If cborName contains a text string, then cborNameByteString contains a
			// string that has the byte string major type but is otherwise identical to
			// cborName.
			flds[i].cborNameByteString = make([]byte, len(flds[i].cborName))
			copy(flds[i].cborNameByteString, flds[i].cborName)
			// Reset encoded CBOR type to byte string, preserving the "additional
			// information" bits:
			flds[i].cborNameByteString[0] = byte(cborTypeByteString) |
				getAdditionalInformation(flds[i].cborNameByteString[0])

			hasKeyAsStr = true
		}

		// Check if field can be omitted when empty
		if flds[i].omitEmpty {
			omitEmptyIdx = append(omitEmptyIdx, i)
		}
	}
	putEncodeBuffer(e)

	if err != nil {
		structType := &encodingStructType{err: err}
		encodingStructTypeCache.Store(t, structType)
		return structType, structType.err
	}

	// Sort fields by canonical order
	bytewiseFields := make(fields, len(flds))
	copy(bytewiseFields, flds)
	sort.Sort(&bytewiseFieldSorter{bytewiseFields})

	lengthFirstFields := bytewiseFields
	if hasKeyAsInt && hasKeyAsStr {
		lengthFirstFields = make(fields, len(flds))
		copy(lengthFirstFields, flds)
		sort.Sort(&lengthFirstFieldSorter{lengthFirstFields})
	}

	structType := &encodingStructType{
		fields:             flds,
		bytewiseFields:     bytewiseFields,
		lengthFirstFields:  lengthFirstFields,
		omitEmptyFieldsIdx: omitEmptyIdx,
	}

	encodingStructTypeCache.Store(t, structType)
	return structType, structType.err
}

func getEncodingStructToArrayType(t reflect.Type, flds fields) (*encodingStructType, error) {
	for i := 0; i < len(flds); i++ {
		// Get field's encodeFunc
		flds[i].ef, flds[i].ief = getEncodeFunc(flds[i].typ)
		if flds[i].ef == nil {
			structType := &encodingStructType{err: &UnsupportedTypeError{t}}
			encodingStructTypeCache.Store(t, structType)
			return structType, structType.err
		}
	}

	structType := &encodingStructType{
		fields:  flds,
		toArray: true,
	}
	encodingStructTypeCache.Store(t, structType)
	return structType, structType.err
}

func getEncodeFunc(t reflect.Type) (encodeFunc, isEmptyFunc) {
	if v, _ := encodeFuncCache.Load(t); v != nil {
		fs := v.(encodeFuncs)
		return fs.ef, fs.ief
	}
	ef, ief := getEncodeFuncInternal(t)
	encodeFuncCache.Store(t, encodeFuncs{ef, ief})
	return ef, ief
}

func getTypeInfo(t reflect.Type) *typeInfo {
	if v, _ := typeInfoCache.Load(t); v != nil {
		return v.(*typeInfo)
	}
	tInfo := newTypeInfo(t)
	typeInfoCache.Store(t, tInfo)
	return tInfo
}

func hasToArrayOption(tag string) bool {
	s := ",toarray"
	idx := strings.Index(tag, s)
	return idx >= 0 && (len(tag) == idx+len(s) || tag[idx+len(s)] == ',')
}
