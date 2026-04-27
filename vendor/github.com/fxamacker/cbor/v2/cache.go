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
	izf isZeroFunc
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
	specialTypeUnexportedUnmarshalerIface
	specialTypeEmptyIface
	specialTypeIface
	specialTypeTag
	specialTypeTime
	specialTypeJSONUnmarshalerIface
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

	for t.Kind() == reflect.Pointer {
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
	} else if reflect.PointerTo(t).Implements(typeUnexportedUnmarshaler) {
		tInfo.spclType = specialTypeUnexportedUnmarshalerIface
	} else if reflect.PointerTo(t).Implements(typeUnmarshaler) {
		tInfo.spclType = specialTypeUnmarshalerIface
	} else if reflect.PointerTo(t).Implements(typeJSONUnmarshaler) {
		tInfo.spclType = specialTypeJSONUnmarshalerIface
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
	fields               decodingFields
	fieldIndicesByName   map[string]int // Only populated if toArray is false
	fieldIndicesByIntKey map[int64]int  // Only populated if toArray is false
	err                  error
	toArray              bool
}

func getDecodingStructType(t reflect.Type) (*decodingStructType, error) {
	if v, _ := decodingStructTypeCache.Load(t); v != nil {
		structType := v.(*decodingStructType)
		if structType.err != nil {
			return nil, structType.err
		}
		return structType, nil
	}

	flds, structOptions := getFields(t)

	toArray := hasToArrayOption(structOptions)

	if toArray {
		return getDecodingStructToArrayType(t, flds)
	}

	fieldIndicesByName := make(map[string]int, len(flds))
	var fieldIndicesByIntKey map[int64]int

	decFlds := make(decodingFields, len(flds))
	for i, f := range flds {
		// nameAsInt is set in getFields() except for fields with an unparsable tagged name.
		// Atoi() is called here to catch and save parsing errors.
		if f.keyAsInt && f.nameAsInt == 0 {
			if _, numErr := strconv.Atoi(f.name); numErr != nil {
				structType := &decodingStructType{
					err: errors.New("cbor: failed to parse field name \"" + f.name + "\" to int (" + numErr.Error() + ")"),
				}
				decodingStructTypeCache.Store(t, structType)
				return nil, structType.err
			}
		}

		if f.keyAsInt {
			if fieldIndicesByIntKey == nil {
				fieldIndicesByIntKey = make(map[int64]int, len(flds))
			}
			// The duplication check is only a safeguard, since getFields() already deduplicates fields.
			if _, ok := fieldIndicesByIntKey[f.nameAsInt]; ok {
				structType := &decodingStructType{
					err: fmt.Errorf("cbor: two or more fields of %v have the same keyasint value %d", t, f.nameAsInt),
				}
				decodingStructTypeCache.Store(t, structType)
				return nil, structType.err
			}
			fieldIndicesByIntKey[f.nameAsInt] = i
		} else {
			// The duplication check is only a safeguard, since getFields() already deduplicates fields.
			if _, ok := fieldIndicesByName[f.name]; ok {
				structType := &decodingStructType{
					err: fmt.Errorf("cbor: two or more fields of %v have the same name %q", t, f.name),
				}
				decodingStructTypeCache.Store(t, structType)
				return nil, structType.err
			}
			fieldIndicesByName[f.name] = i
		}

		decFlds[i] = &decodingField{
			field:   *f,
			typInfo: getTypeInfo(f.typ),
		}
	}

	structType := &decodingStructType{
		fields:               decFlds,
		fieldIndicesByName:   fieldIndicesByName,
		fieldIndicesByIntKey: fieldIndicesByIntKey,
	}
	decodingStructTypeCache.Store(t, structType)
	return structType, nil
}

func getDecodingStructToArrayType(t reflect.Type, flds fields) (*decodingStructType, error) {
	decFlds := make(decodingFields, len(flds))
	for i, f := range flds {
		// nameAsInt is set in getFields() except for fields with an unparsable tagged name.
		// Atoi() is called here to catch and save parsing errors.
		if f.keyAsInt && f.nameAsInt == 0 {
			if _, numErr := strconv.Atoi(f.name); numErr != nil {
				structType := &decodingStructType{
					err: errors.New("cbor: failed to parse field name \"" + f.name + "\" to int (" + numErr.Error() + ")"),
				}
				decodingStructTypeCache.Store(t, structType)
				return nil, structType.err
			}
		}

		decFlds[i] = &decodingField{
			field:   *f,
			typInfo: getTypeInfo(f.typ),
		}
	}

	structType := &decodingStructType{
		fields:  decFlds,
		toArray: true,
	}
	decodingStructTypeCache.Store(t, structType)
	return structType, nil
}

type encodingStructType struct {
	fields             encodingFields
	bytewiseFields     encodingFields // Only populated if toArray is false
	lengthFirstFields  encodingFields // Only populated if toArray is false
	omitEmptyFieldsIdx []int          // Only populated if toArray is false
	err                error
	toArray            bool
}

func (st *encodingStructType) getFields(em *encMode) encodingFields {
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
	fields encodingFields
}

func (x *bytewiseFieldSorter) Len() int {
	return len(x.fields)
}

func (x *bytewiseFieldSorter) Swap(i, j int) {
	x.fields[i], x.fields[j] = x.fields[j], x.fields[i]
}

func (x *bytewiseFieldSorter) Less(i, j int) bool {
	return bytes.Compare(x.fields[i].cborName, x.fields[j].cborName) < 0
}

type lengthFirstFieldSorter struct {
	fields encodingFields
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
	return bytes.Compare(x.fields[i].cborName, x.fields[j].cborName) < 0
}

func getEncodingStructType(t reflect.Type) (*encodingStructType, error) {
	if v, _ := encodingStructTypeCache.Load(t); v != nil {
		structType := v.(*encodingStructType)
		if structType.err != nil {
			return nil, structType.err
		}
		return structType, nil
	}

	flds, structOptions := getFields(t)

	if hasToArrayOption(structOptions) {
		return getEncodingStructToArrayType(t, flds)
	}

	var hasKeyAsInt bool
	var hasKeyAsStr bool
	var omitEmptyIdx []int

	encFlds := make(encodingFields, len(flds))

	e := getEncodeBuffer()
	defer putEncodeBuffer(e)

	for i, f := range flds {
		encFlds[i] = &encodingField{field: *f}
		ef := encFlds[i]

		// Get field's encodeFunc
		ef.ef, ef.ief, ef.izf = getEncodeFunc(f.typ)
		if ef.ef == nil {
			structType := &encodingStructType{err: &UnsupportedTypeError{t}}
			encodingStructTypeCache.Store(t, structType)
			return nil, structType.err
		}

		// Encode field name
		if f.keyAsInt {
			if f.nameAsInt == 0 {
				// nameAsInt is set in getFields() except for fields with an unparsable tagged name.
				// Atoi() is called here to catch and save parsing errors.
				if _, numErr := strconv.Atoi(f.name); numErr != nil {
					structType := &encodingStructType{
						err: errors.New("cbor: failed to parse field name \"" + f.name + "\" to int (" + numErr.Error() + ")"),
					}
					encodingStructTypeCache.Store(t, structType)
					return nil, structType.err
				}
			}
			nameAsInt := f.nameAsInt
			if nameAsInt >= 0 {
				encodeHead(e, byte(cborTypePositiveInt), uint64(nameAsInt)) //nolint:gosec
			} else {
				n := nameAsInt*(-1) - 1
				encodeHead(e, byte(cborTypeNegativeInt), uint64(n)) //nolint:gosec
			}
			ef.cborName = make([]byte, e.Len())
			copy(ef.cborName, e.Bytes())
			e.Reset()

			hasKeyAsInt = true
		} else {
			encodeHead(e, byte(cborTypeTextString), uint64(len(f.name)))
			ef.cborName = make([]byte, e.Len()+len(f.name))
			n := copy(ef.cborName, e.Bytes())
			copy(ef.cborName[n:], f.name)
			e.Reset()

			// If cborName contains a text string, then cborNameByteString contains a
			// string that has the byte string major type but is otherwise identical to
			// cborName.
			ef.cborNameByteString = make([]byte, len(ef.cborName))
			copy(ef.cborNameByteString, ef.cborName)
			// Reset encoded CBOR type to byte string, preserving the "additional
			// information" bits:
			ef.cborNameByteString[0] = byte(cborTypeByteString) |
				getAdditionalInformation(ef.cborNameByteString[0])

			hasKeyAsStr = true
		}

		// Check if field can be omitted when empty
		if f.omitEmpty {
			omitEmptyIdx = append(omitEmptyIdx, i)
		}
	}

	// Sort fields by canonical order
	bytewiseFields := make(encodingFields, len(encFlds))
	copy(bytewiseFields, encFlds)
	sort.Sort(&bytewiseFieldSorter{bytewiseFields})

	lengthFirstFields := bytewiseFields
	if hasKeyAsInt && hasKeyAsStr {
		lengthFirstFields = make(encodingFields, len(encFlds))
		copy(lengthFirstFields, encFlds)
		sort.Sort(&lengthFirstFieldSorter{lengthFirstFields})
	}

	structType := &encodingStructType{
		fields:             encFlds,
		bytewiseFields:     bytewiseFields,
		lengthFirstFields:  lengthFirstFields,
		omitEmptyFieldsIdx: omitEmptyIdx,
	}

	encodingStructTypeCache.Store(t, structType)
	return structType, nil
}

func getEncodingStructToArrayType(t reflect.Type, flds fields) (*encodingStructType, error) {
	encFlds := make(encodingFields, len(flds))
	for i, f := range flds {
		encFlds[i] = &encodingField{field: *f}
		encFlds[i].ef, encFlds[i].ief, encFlds[i].izf = getEncodeFunc(f.typ)
		if encFlds[i].ef == nil {
			structType := &encodingStructType{err: &UnsupportedTypeError{t}}
			encodingStructTypeCache.Store(t, structType)
			return nil, structType.err
		}
	}

	structType := &encodingStructType{
		fields:  encFlds,
		toArray: true,
	}
	encodingStructTypeCache.Store(t, structType)
	return structType, nil
}

func getEncodeFunc(t reflect.Type) (encodeFunc, isEmptyFunc, isZeroFunc) {
	if v, _ := encodeFuncCache.Load(t); v != nil {
		fs := v.(encodeFuncs)
		return fs.ef, fs.ief, fs.izf
	}
	ef, ief, izf := getEncodeFuncInternal(t)
	encodeFuncCache.Store(t, encodeFuncs{ef, ief, izf})
	return ef, ief, izf
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
