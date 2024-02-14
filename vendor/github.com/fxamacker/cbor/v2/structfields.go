// Copyright (c) Faye Amacker. All rights reserved.
// Licensed under the MIT License. See LICENSE in the project root for license information.

package cbor

import (
	"reflect"
	"sort"
	"strings"
)

type field struct {
	name               string
	nameAsInt          int64 // used to decoder to match field name with CBOR int
	cborName           []byte
	cborNameByteString []byte // major type 2 name encoding iff cborName has major type 3
	idx                []int
	typ                reflect.Type
	ef                 encodeFunc
	ief                isEmptyFunc
	typInfo            *typeInfo // used to decoder to reuse type info
	tagged             bool      // used to choose dominant field (at the same level tagged fields dominate untagged fields)
	omitEmpty          bool      // used to skip empty field
	keyAsInt           bool      // used to encode/decode field name as int
}

type fields []*field

// indexFieldSorter sorts fields by field idx at each level, breaking ties with idx depth.
type indexFieldSorter struct {
	fields fields
}

func (x *indexFieldSorter) Len() int {
	return len(x.fields)
}

func (x *indexFieldSorter) Swap(i, j int) {
	x.fields[i], x.fields[j] = x.fields[j], x.fields[i]
}

func (x *indexFieldSorter) Less(i, j int) bool {
	iIdx, jIdx := x.fields[i].idx, x.fields[j].idx
	for k := 0; k < len(iIdx) && k < len(jIdx); k++ {
		if iIdx[k] != jIdx[k] {
			return iIdx[k] < jIdx[k]
		}
	}
	return len(iIdx) <= len(jIdx)
}

// nameLevelAndTagFieldSorter sorts fields by field name, idx depth, and presence of tag.
type nameLevelAndTagFieldSorter struct {
	fields fields
}

func (x *nameLevelAndTagFieldSorter) Len() int {
	return len(x.fields)
}

func (x *nameLevelAndTagFieldSorter) Swap(i, j int) {
	x.fields[i], x.fields[j] = x.fields[j], x.fields[i]
}

func (x *nameLevelAndTagFieldSorter) Less(i, j int) bool {
	fi, fj := x.fields[i], x.fields[j]
	if fi.name != fj.name {
		return fi.name < fj.name
	}
	if len(fi.idx) != len(fj.idx) {
		return len(fi.idx) < len(fj.idx)
	}
	if fi.tagged != fj.tagged {
		return fi.tagged
	}
	return i < j // Field i and j have the same name, depth, and tagged status. Nothing else matters.
}

// getFields returns visible fields of struct type t following visibility rules for JSON encoding.
func getFields(t reflect.Type) (flds fields, structOptions string) {
	// Get special field "_" tag options
	if f, ok := t.FieldByName("_"); ok {
		tag := f.Tag.Get("cbor")
		if tag != "-" {
			structOptions = tag
		}
	}

	// nTypes contains next level anonymous fields' types and indexes
	// (there can be multiple fields of the same type at the same level)
	flds, nTypes := appendFields(t, nil, nil, nil)

	if len(nTypes) > 0 {

		var cTypes map[reflect.Type][][]int      // current level anonymous fields' types and indexes
		vTypes := map[reflect.Type]bool{t: true} // visited field types at less nested levels

		for len(nTypes) > 0 {
			cTypes, nTypes = nTypes, nil

			for t, idx := range cTypes {
				// If there are multiple anonymous fields of the same struct type at the same level, all are ignored.
				if len(idx) > 1 {
					continue
				}

				// Anonymous field of the same type at deeper nested level is ignored.
				if vTypes[t] {
					continue
				}
				vTypes[t] = true

				flds, nTypes = appendFields(t, idx[0], flds, nTypes)
			}
		}
	}

	sort.Sort(&nameLevelAndTagFieldSorter{flds})

	// Keep visible fields.
	j := 0 // index of next unique field
	for i := 0; i < len(flds); {
		name := flds[i].name
		if i == len(flds)-1 || // last field
			name != flds[i+1].name || // field i has unique field name
			len(flds[i].idx) < len(flds[i+1].idx) || // field i is at a less nested level than field i+1
			(flds[i].tagged && !flds[i+1].tagged) { // field i is tagged while field i+1 is not
			flds[j] = flds[i]
			j++
		}

		// Skip fields with the same field name.
		for i++; i < len(flds) && name == flds[i].name; i++ { //nolint:revive
		}
	}
	if j != len(flds) {
		flds = flds[:j]
	}

	// Sort fields by field index
	sort.Sort(&indexFieldSorter{flds})

	return flds, structOptions
}

// appendFields appends type t's exportable fields to flds and anonymous struct fields to nTypes .
func appendFields(t reflect.Type, idx []int, flds fields, nTypes map[reflect.Type][][]int) (fields, map[reflect.Type][][]int) {
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)

		ft := f.Type
		for ft.Kind() == reflect.Ptr {
			ft = ft.Elem()
		}

		if !isFieldExportable(f, ft.Kind()) {
			continue
		}

		tag := f.Tag.Get("cbor")
		if tag == "" {
			tag = f.Tag.Get("json")
		}
		if tag == "-" {
			continue
		}

		tagged := len(tag) > 0

		// Parse field tag options
		var tagFieldName string
		var omitempty, keyasint bool
		for j := 0; len(tag) > 0; j++ {
			var token string
			idx := strings.IndexByte(tag, ',')
			if idx == -1 {
				token, tag = tag, ""
			} else {
				token, tag = tag[:idx], tag[idx+1:]
			}
			if j == 0 {
				tagFieldName = token
			} else {
				switch token {
				case "omitempty":
					omitempty = true
				case "keyasint":
					keyasint = true
				}
			}
		}

		fieldName := tagFieldName
		if tagFieldName == "" {
			fieldName = f.Name
		}

		fIdx := make([]int, len(idx)+1)
		copy(fIdx, idx)
		fIdx[len(fIdx)-1] = i

		if !f.Anonymous || ft.Kind() != reflect.Struct || len(tagFieldName) > 0 {
			flds = append(flds, &field{
				name:      fieldName,
				idx:       fIdx,
				typ:       f.Type,
				omitEmpty: omitempty,
				keyAsInt:  keyasint,
				tagged:    tagged})
		} else {
			if nTypes == nil {
				nTypes = make(map[reflect.Type][][]int)
			}
			nTypes[ft] = append(nTypes[ft], fIdx)
		}
	}

	return flds, nTypes
}

// isFieldExportable returns true if f is an exportable (regular or anonymous) field or
// a nonexportable anonymous field of struct type.
// Nonexportable anonymous field of struct type can contain exportable fields.
func isFieldExportable(f reflect.StructField, fk reflect.Kind) bool {
	exportable := f.PkgPath == ""
	return exportable || (f.Anonymous && fk == reflect.Struct)
}

type embeddedFieldNullPtrFunc func(reflect.Value) (reflect.Value, error)

// getFieldValue returns field value of struct v by index.  When encountering null pointer
// to anonymous (embedded) struct field, f is called with the last traversed field value.
func getFieldValue(v reflect.Value, idx []int, f embeddedFieldNullPtrFunc) (fv reflect.Value, err error) {
	fv = v
	for i, n := range idx {
		fv = fv.Field(n)

		if i < len(idx)-1 {
			if fv.Kind() == reflect.Ptr && fv.Type().Elem().Kind() == reflect.Struct {
				if fv.IsNil() {
					// Null pointer to embedded struct field
					fv, err = f(fv)
					if err != nil || !fv.IsValid() {
						return fv, err
					}
				}
				fv = fv.Elem()
			}
		}
	}
	return fv, nil
}
