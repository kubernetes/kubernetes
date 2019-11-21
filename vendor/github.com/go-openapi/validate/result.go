// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package validate

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/go-openapi/errors"
	"github.com/go-openapi/spec"
)

// Result represents a validation result set, composed of
// errors and warnings.
//
// It is used to keep track of all detected errors and warnings during
// the validation of a specification.
//
// Matchcount is used to determine
// which errors are relevant in the case of AnyOf, OneOf
// schema validation. Results from the validation branch
// with most matches get eventually selected.
//
// TODO: keep path of key originating the error
type Result struct {
	Errors     []error
	Warnings   []error
	MatchCount int

	// the object data
	data interface{}

	// Schemata for the root object
	rootObjectSchemata schemata
	// Schemata for object fields
	fieldSchemata []fieldSchemata
	// Schemata for slice items
	itemSchemata []itemSchemata

	cachedFieldSchemta map[FieldKey][]*spec.Schema
	cachedItemSchemata map[ItemKey][]*spec.Schema
}

// FieldKey is a pair of an object and a field, usable as a key for a map.
type FieldKey struct {
	object reflect.Value // actually a map[string]interface{}, but the latter cannot be a key
	field  string
}

// ItemKey is a pair of a slice and an index, usable as a key for a map.
type ItemKey struct {
	slice reflect.Value // actually a []interface{}, but the latter cannot be a key
	index int
}

// NewFieldKey returns a pair of an object and field usable as a key of a map.
func NewFieldKey(obj map[string]interface{}, field string) FieldKey {
	return FieldKey{object: reflect.ValueOf(obj), field: field}
}

// Object returns the underlying object of this key.
func (fk *FieldKey) Object() map[string]interface{} {
	return fk.object.Interface().(map[string]interface{})
}

// Field returns the underlying field of this key.
func (fk *FieldKey) Field() string {
	return fk.field
}

// NewItemKey returns a pair of a slice and index usable as a key of a map.
func NewItemKey(slice interface{}, i int) ItemKey {
	return ItemKey{slice: reflect.ValueOf(slice), index: i}
}

// Slice returns the underlying slice of this key.
func (ik *ItemKey) Slice() []interface{} {
	return ik.slice.Interface().([]interface{})
}

// Index returns the underlying index of this key.
func (ik *ItemKey) Index() int {
	return ik.index
}

type fieldSchemata struct {
	obj      map[string]interface{}
	field    string
	schemata schemata
}

type itemSchemata struct {
	slice    reflect.Value
	index    int
	schemata schemata
}

// Merge merges this result with the other one(s), preserving match counts etc.
func (r *Result) Merge(others ...*Result) *Result {
	for _, other := range others {
		if other == nil {
			continue
		}
		r.mergeWithoutRootSchemata(other)
		r.rootObjectSchemata.Append(other.rootObjectSchemata)
	}
	return r
}

// Data returns the original data object used for validation. Mutating this renders
// the result invalid.
func (r *Result) Data() interface{} {
	return r.data
}

// RootObjectSchemata returns the schemata which apply to the root object.
func (r *Result) RootObjectSchemata() []*spec.Schema {
	return r.rootObjectSchemata.Slice()
}

// FieldSchemata returns the schemata which apply to fields in objects.
// nolint: dupl
func (r *Result) FieldSchemata() map[FieldKey][]*spec.Schema {
	if r.cachedFieldSchemta != nil {
		return r.cachedFieldSchemta
	}

	ret := make(map[FieldKey][]*spec.Schema, len(r.fieldSchemata))
	for _, fs := range r.fieldSchemata {
		key := NewFieldKey(fs.obj, fs.field)
		if fs.schemata.one != nil {
			ret[key] = append(ret[key], fs.schemata.one)
		} else if len(fs.schemata.multiple) > 0 {
			ret[key] = append(ret[key], fs.schemata.multiple...)
		}
	}
	r.cachedFieldSchemta = ret
	return ret
}

// ItemSchemata returns the schemata which apply to items in slices.
// nolint: dupl
func (r *Result) ItemSchemata() map[ItemKey][]*spec.Schema {
	if r.cachedItemSchemata != nil {
		return r.cachedItemSchemata
	}

	ret := make(map[ItemKey][]*spec.Schema, len(r.itemSchemata))
	for _, ss := range r.itemSchemata {
		key := NewItemKey(ss.slice, ss.index)
		if ss.schemata.one != nil {
			ret[key] = append(ret[key], ss.schemata.one)
		} else if len(ss.schemata.multiple) > 0 {
			ret[key] = append(ret[key], ss.schemata.multiple...)
		}
	}
	r.cachedItemSchemata = ret
	return ret
}

func (r *Result) resetCaches() {
	r.cachedFieldSchemta = nil
	r.cachedItemSchemata = nil
}

// mergeForField merges other into r, assigning other's root schemata to the given Object and field name.
// nolint: unparam
func (r *Result) mergeForField(obj map[string]interface{}, field string, other *Result) *Result {
	if other == nil {
		return r
	}
	r.mergeWithoutRootSchemata(other)

	if other.rootObjectSchemata.Len() > 0 {
		if r.fieldSchemata == nil {
			r.fieldSchemata = make([]fieldSchemata, len(obj))
		}
		r.fieldSchemata = append(r.fieldSchemata, fieldSchemata{
			obj:      obj,
			field:    field,
			schemata: other.rootObjectSchemata,
		})
	}

	return r
}

// mergeForSlice merges other into r, assigning other's root schemata to the given slice and index.
// nolint: unparam
func (r *Result) mergeForSlice(slice reflect.Value, i int, other *Result) *Result {
	if other == nil {
		return r
	}
	r.mergeWithoutRootSchemata(other)

	if other.rootObjectSchemata.Len() > 0 {
		if r.itemSchemata == nil {
			r.itemSchemata = make([]itemSchemata, slice.Len())
		}
		r.itemSchemata = append(r.itemSchemata, itemSchemata{
			slice:    slice,
			index:    i,
			schemata: other.rootObjectSchemata,
		})
	}

	return r
}

// addRootObjectSchemata adds the given schemata for the root object of the result.
// The slice schemata might be reused. I.e. do not modify it after being added to a result.
func (r *Result) addRootObjectSchemata(s *spec.Schema) {
	r.rootObjectSchemata.Append(schemata{one: s})
}

// addPropertySchemata adds the given schemata for the object and field.
// The slice schemata might be reused. I.e. do not modify it after being added to a result.
func (r *Result) addPropertySchemata(obj map[string]interface{}, fld string, schema *spec.Schema) {
	if r.fieldSchemata == nil {
		r.fieldSchemata = make([]fieldSchemata, 0, len(obj))
	}
	r.fieldSchemata = append(r.fieldSchemata, fieldSchemata{obj: obj, field: fld, schemata: schemata{one: schema}})
}

/*
// addSliceSchemata adds the given schemata for the slice and index.
// The slice schemata might be reused. I.e. do not modify it after being added to a result.
func (r *Result) addSliceSchemata(slice reflect.Value, i int, schema *spec.Schema) {
	if r.itemSchemata == nil {
		r.itemSchemata = make([]itemSchemata, 0, slice.Len())
	}
	r.itemSchemata = append(r.itemSchemata, itemSchemata{slice: slice, index: i, schemata: schemata{one: schema}})
}
*/

// mergeWithoutRootSchemata merges other into r, ignoring the rootObject schemata.
func (r *Result) mergeWithoutRootSchemata(other *Result) {
	r.resetCaches()
	r.AddErrors(other.Errors...)
	r.AddWarnings(other.Warnings...)
	r.MatchCount += other.MatchCount

	if other.fieldSchemata != nil {
		if r.fieldSchemata == nil {
			r.fieldSchemata = other.fieldSchemata
		} else {
			r.fieldSchemata = append(r.fieldSchemata, other.fieldSchemata...)
		}
	}

	if other.itemSchemata != nil {
		if r.itemSchemata == nil {
			r.itemSchemata = other.itemSchemata
		} else {
			r.itemSchemata = append(r.itemSchemata, other.itemSchemata...)
		}
	}
}

// MergeAsErrors merges this result with the other one(s), preserving match counts etc.
//
// Warnings from input are merged as Errors in the returned merged Result.
func (r *Result) MergeAsErrors(others ...*Result) *Result {
	for _, other := range others {
		if other != nil {
			r.resetCaches()
			r.AddErrors(other.Errors...)
			r.AddErrors(other.Warnings...)
			r.MatchCount += other.MatchCount
		}
	}
	return r
}

// MergeAsWarnings merges this result with the other one(s), preserving match counts etc.
//
// Errors from input are merged as Warnings in the returned merged Result.
func (r *Result) MergeAsWarnings(others ...*Result) *Result {
	for _, other := range others {
		if other != nil {
			r.resetCaches()
			r.AddWarnings(other.Errors...)
			r.AddWarnings(other.Warnings...)
			r.MatchCount += other.MatchCount
		}
	}
	return r
}

// AddErrors adds errors to this validation result (if not already reported).
//
// Since the same check may be passed several times while exploring the
// spec structure (via $ref, ...) reported messages are kept
// unique.
func (r *Result) AddErrors(errors ...error) {
	for _, e := range errors {
		found := false
		if e != nil {
			for _, isReported := range r.Errors {
				if e.Error() == isReported.Error() {
					found = true
					break
				}
			}
			if !found {
				r.Errors = append(r.Errors, e)
			}
		}
	}
}

// AddWarnings adds warnings to this validation result (if not already reported).
func (r *Result) AddWarnings(warnings ...error) {
	for _, e := range warnings {
		found := false
		if e != nil {
			for _, isReported := range r.Warnings {
				if e.Error() == isReported.Error() {
					found = true
					break
				}
			}
			if !found {
				r.Warnings = append(r.Warnings, e)
			}
		}
	}
}

func (r *Result) keepRelevantErrors() *Result {
	// TODO: this one is going to disapear...
	// keepRelevantErrors strips a result from standard errors and keeps
	// the ones which are supposedly more accurate.
	//
	// The original result remains unaffected (creates a new instance of Result).
	// This method is used to work around the "matchCount" filter which would otherwise
	// strip our result from some accurate error reporting from lower level validators.
	//
	// NOTE: this implementation with a placeholder (IMPORTANT!) is neither clean nor
	// very efficient. On the other hand, relying on go-openapi/errors to manipulate
	// codes would require to change a lot here. So, for the moment, let's go with
	// placeholders.
	strippedErrors := []error{}
	for _, e := range r.Errors {
		if strings.HasPrefix(e.Error(), "IMPORTANT!") {
			strippedErrors = append(strippedErrors, fmt.Errorf(strings.TrimPrefix(e.Error(), "IMPORTANT!")))
		}
	}
	strippedWarnings := []error{}
	for _, e := range r.Warnings {
		if strings.HasPrefix(e.Error(), "IMPORTANT!") {
			strippedWarnings = append(strippedWarnings, fmt.Errorf(strings.TrimPrefix(e.Error(), "IMPORTANT!")))
		}
	}
	strippedResult := new(Result)
	strippedResult.Errors = strippedErrors
	strippedResult.Warnings = strippedWarnings
	return strippedResult
}

// IsValid returns true when this result is valid.
//
// Returns true on a nil *Result.
func (r *Result) IsValid() bool {
	if r == nil {
		return true
	}
	return len(r.Errors) == 0
}

// HasErrors returns true when this result is invalid.
//
// Returns false on a nil *Result.
func (r *Result) HasErrors() bool {
	if r == nil {
		return false
	}
	return !r.IsValid()
}

// HasWarnings returns true when this result contains warnings.
//
// Returns false on a nil *Result.
func (r *Result) HasWarnings() bool {
	if r == nil {
		return false
	}
	return len(r.Warnings) > 0
}

// HasErrorsOrWarnings returns true when this result contains
// either errors or warnings.
//
// Returns false on a nil *Result.
func (r *Result) HasErrorsOrWarnings() bool {
	if r == nil {
		return false
	}
	return len(r.Errors) > 0 || len(r.Warnings) > 0
}

// Inc increments the match count
func (r *Result) Inc() {
	r.MatchCount++
}

// AsError renders this result as an error interface
//
// TODO: reporting / pretty print with path ordered and indented
func (r *Result) AsError() error {
	if r.IsValid() {
		return nil
	}
	return errors.CompositeValidationError(r.Errors...)
}

// schemata is an arbitrary number of schemata. It does a distinction between zero,
// one and many schemata to avoid slice allocations.
type schemata struct {
	// one is set if there is exactly one schema. In that case multiple must be nil.
	one *spec.Schema
	// multiple is an arbitrary number of schemas. If it is set, one must be nil.
	multiple []*spec.Schema
}

func (s *schemata) Len() int {
	if s.one != nil {
		return 1
	}
	return len(s.multiple)
}

func (s *schemata) Slice() []*spec.Schema {
	if s == nil {
		return nil
	}
	if s.one != nil {
		return []*spec.Schema{s.one}
	}
	return s.multiple
}

// appendSchemata appends the schemata in other to s. It mutated s in-place.
func (s *schemata) Append(other schemata) {
	if other.one == nil && len(other.multiple) == 0 {
		return
	}
	if s.one == nil && len(s.multiple) == 0 {
		*s = other
		return
	}

	if s.one != nil {
		if other.one != nil {
			s.multiple = []*spec.Schema{s.one, other.one}
		} else {
			t := make([]*spec.Schema, 0, 1+len(other.multiple))
			s.multiple = append(append(t, s.one), other.multiple...)
		}
		s.one = nil
	} else {
		if other.one != nil {
			s.multiple = append(s.multiple, other.one)
		} else {
			if cap(s.multiple) >= len(s.multiple)+len(other.multiple) {
				s.multiple = append(s.multiple, other.multiple...)
			} else {
				t := make([]*spec.Schema, 0, len(s.multiple)+len(other.multiple))
				s.multiple = append(append(t, s.multiple...), other.multiple...)
			}
		}
	}
}
