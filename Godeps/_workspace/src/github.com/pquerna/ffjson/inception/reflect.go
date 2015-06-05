/**
 *  Copyright 2014 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

package ffjsoninception

import (
	fflib "github.com/pquerna/ffjson/fflib/v1"
	"github.com/pquerna/ffjson/shared"

	"bytes"
	"encoding/json"
	"reflect"
	"unicode/utf8"
)

type StructField struct {
	Name             string
	JsonName         string
	FoldFuncName     string
	Typ              reflect.Type
	OmitEmpty        bool
	ForceString      bool
	HasMarshalJSON   bool
	HasUnmarshalJSON bool
	Pointer          bool
	Tagged           bool
}

type FieldByJsonName []*StructField

func (a FieldByJsonName) Len() int           { return len(a) }
func (a FieldByJsonName) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a FieldByJsonName) Less(i, j int) bool { return a[i].JsonName < a[j].JsonName }

type StructInfo struct {
	Name    string
	Obj     interface{}
	Typ     reflect.Type
	Fields  []*StructField
	Options shared.StructOptions
}

func NewStructInfo(obj shared.InceptionType) *StructInfo {
	t := reflect.TypeOf(obj.Obj)
	return &StructInfo{
		Obj:     obj.Obj,
		Name:    t.Name(),
		Typ:     t,
		Fields:  extractFields(obj.Obj),
		Options: obj.Options,
	}
}

func (si *StructInfo) FieldsByFirstByte() map[string][]*StructField {
	rv := make(map[string][]*StructField)
	for _, f := range si.Fields {
		b := string(f.JsonName[1])
		rv[b] = append(rv[b], f)
	}
	return rv
}

func (si *StructInfo) ReverseFields() []*StructField {
	var i int
	rv := make([]*StructField, 0)
	for i = len(si.Fields) - 1; i >= 0; i-- {
		rv = append(rv, si.Fields[i])
	}
	return rv
}

const (
	caseMask = ^byte(0x20) // Mask to ignore case in ASCII.
)

func foldFunc(key []byte) string {
	nonLetter := false
	special := false // special letter
	for _, b := range key {
		if b >= utf8.RuneSelf {
			return "bytes.EqualFold"
		}
		upper := b & caseMask
		if upper < 'A' || upper > 'Z' {
			nonLetter = true
		} else if upper == 'K' || upper == 'S' {
			// See above for why these letters are special.
			special = true
		}
	}
	if special {
		return "fflib.EqualFoldRight"
	}
	if nonLetter {
		return "fflib.AsciiEqualFold"
	}
	return "fflib.SimpleLetterEqualFold"
}

type MarshalerFaster interface {
	MarshalJSONBuf(buf fflib.EncodingBuffer) error
}

type UnmarshalFaster interface {
	UnmarshalJSONFFLexer(l *fflib.FFLexer, state fflib.FFParseState) error
}

var marshalerType = reflect.TypeOf(new(json.Marshaler)).Elem()
var marshalerFasterType = reflect.TypeOf(new(MarshalerFaster)).Elem()
var unmarshalerType = reflect.TypeOf(new(json.Unmarshaler)).Elem()
var unmarshalFasterType = reflect.TypeOf(new(UnmarshalFaster)).Elem()

// extractFields returns a list of fields that JSON should recognize for the given type.
// The algorithm is breadth-first search over the set of structs to include - the top struct
// and then any reachable anonymous structs.
func extractFields(obj interface{}) []*StructField {
	t := reflect.TypeOf(obj)
	// Anonymous fields to explore at the current level and the next.
	current := []StructField{}
	next := []StructField{{Typ: t}}

	// Count of queued names for current level and the next.
	count := map[reflect.Type]int{}
	nextCount := map[reflect.Type]int{}

	// Types already visited at an earlier level.
	visited := map[reflect.Type]bool{}

	// Fields found.
	var fields []*StructField

	for len(next) > 0 {
		current, next = next, current[:0]
		count, nextCount = nextCount, map[reflect.Type]int{}

		for _, f := range current {
			if visited[f.Typ] {
				continue
			}
			visited[f.Typ] = true

			// Scan f.typ for fields to include.
			for i := 0; i < f.Typ.NumField(); i++ {
				sf := f.Typ.Field(i)
				if sf.PkgPath != "" { // unexported
					continue
				}
				tag := sf.Tag.Get("json")
				if tag == "-" {
					continue
				}
				name, opts := parseTag(tag)
				if !isValidTag(name) {
					name = ""
				}

				ft := sf.Type
				ptr := false
				if ft.Kind() == reflect.Ptr {
					ptr = true
				}

				if ft.Name() == "" && ft.Kind() == reflect.Ptr {
					// Follow pointer.
					ft = ft.Elem()
				}

				// Record found field and index sequence.
				if name != "" || !sf.Anonymous || ft.Kind() != reflect.Struct {
					tagged := name != ""
					if name == "" {
						name = sf.Name
					}

					var buf bytes.Buffer
					fflib.WriteJsonString(&buf, name)

					field := &StructField{
						Name:             sf.Name,
						JsonName:         string(buf.Bytes()),
						FoldFuncName:     foldFunc([]byte(name)),
						Typ:              ft,
						HasMarshalJSON:   ft.Implements(marshalerType),
						HasUnmarshalJSON: ft.Implements(unmarshalerType),
						OmitEmpty:        opts.Contains("omitempty"),
						ForceString:      opts.Contains("string"),
						Pointer:          ptr,
						Tagged:           tagged,
					}

					fields = append(fields, field)

					if count[f.Typ] > 1 {
						// If there were multiple instances, add a second,
						// so that the annihilation code will see a duplicate.
						// It only cares about the distinction between 1 or 2,
						// so don't bother generating any more copies.
						fields = append(fields, fields[len(fields)-1])
					}
					continue
				}

				// Record new anonymous struct to explore in next round.
				nextCount[ft]++
				if nextCount[ft] == 1 {
					next = append(next, StructField{
						Name: ft.Name(),
						Typ:  ft,
					})
				}
			}
		}
	}

	// Delete all fields that are hidden by the Go rules for embedded fields,
	// except that fields with JSON tags are promoted.

	// The fields are sorted in primary order of name, secondary order
	// of field index length. Loop over names; for each name, delete
	// hidden fields by choosing the one dominant field that survives.
	out := fields[:0]
	for advance, i := 0, 0; i < len(fields); i += advance {
		// One iteration per name.
		// Find the sequence of fields with the name of this first field.
		fi := fields[i]
		name := fi.JsonName
		for advance = 1; i+advance < len(fields); advance++ {
			fj := fields[i+advance]
			if fj.JsonName != name {
				break
			}
		}
		if advance == 1 { // Only one field with this name
			out = append(out, fi)
			continue
		}
		dominant, ok := dominantField(fields[i : i+advance])
		if ok {
			out = append(out, dominant)
		}
	}

	fields = out

	return fields
}

// dominantField looks through the fields, all of which are known to
// have the same name, to find the single field that dominates the
// others using Go's embedding rules, modified by the presence of
// JSON tags. If there are multiple top-level fields, the boolean
// will be false: This condition is an error in Go and we skip all
// the fields.
func dominantField(fields []*StructField) (*StructField, bool) {
	tagged := -1 // Index of first tagged field.
	for i, f := range fields {
		if f.Tagged {
			if tagged >= 0 {
				// Multiple tagged fields at the same level: conflict.
				// Return no field.
				return nil, false
			}
			tagged = i
		}
	}
	if tagged >= 0 {
		return fields[tagged], true
	}
	// All remaining fields have the same length. If there's more than one,
	// we have a conflict (two fields named "X" at the same level) and we
	// return no field.
	if len(fields) > 1 {
		return nil, false
	}
	return fields[0], true
}
