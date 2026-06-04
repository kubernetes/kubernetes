/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package typed

import (
	"sync"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/schema"
	"sigs.k8s.io/structured-merge-diff/v6/value"
)

// ValidationOptions is the list of all the options available when running the validation.
type ValidationOptions int

const (
	// AllowDuplicates means that sets and associative lists can have duplicate similar items.
	AllowDuplicates ValidationOptions = iota
)

// extractItemsOptions is the options available when extracting items.
type extractItemsOptions struct {
	appendKeyFields bool
}

type ExtractItemsOption func(*extractItemsOptions)

// WithAppendKeyFields configures ExtractItems to include key fields.
// It is exported for use in configuring ExtractItems.
func WithAppendKeyFields() ExtractItemsOption {
	return func(opts *extractItemsOptions) {
		opts.appendKeyFields = true
	}
}

// AsTyped accepts a value and a type and returns a TypedValue. 'v' must have
// type 'typeName' in the schema. An error is returned if the v doesn't conform
// to the schema.
func AsTyped(v value.Value, s *schema.Schema, typeRef schema.TypeRef, opts ...ValidationOptions) (*TypedValue, error) {
	tv := &TypedValue{
		value:   v,
		typeRef: typeRef,
		schema:  s,
	}
	if err := tv.Validate(opts...); err != nil {
		return nil, err
	}
	return tv, nil
}

// AsTypeUnvalidated is just like AsTyped, but doesn't validate that the type
// conforms to the schema, for cases where that has already been checked or
// where you're going to call a method that validates as a side-effect (like
// ToFieldSet).
//
// Deprecated: This function was initially created because validation
// was expensive. Now that this has been solved, objects should always
// be created as validated, using `AsTyped`.
func AsTypedUnvalidated(v value.Value, s *schema.Schema, typeRef schema.TypeRef) *TypedValue {
	tv := &TypedValue{
		value:   v,
		typeRef: typeRef,
		schema:  s,
	}
	return tv
}

// TypedValue is a value of some specific type.
type TypedValue struct {
	value   value.Value
	typeRef schema.TypeRef
	schema  *schema.Schema
}

// TypeRef is the type of the value.
func (tv TypedValue) TypeRef() schema.TypeRef {
	return tv.typeRef
}

// AsValue removes the type from the TypedValue and only keeps the value.
func (tv TypedValue) AsValue() value.Value {
	return tv.value
}

// Schema gets the schema from the TypedValue.
func (tv TypedValue) Schema() *schema.Schema {
	return tv.schema
}

// Validate returns an error with a list of every spec violation.
func (tv TypedValue) Validate(opts ...ValidationOptions) error {
	w := tv.walker()
	for _, opt := range opts {
		switch opt {
		case AllowDuplicates:
			w.allowDuplicates = true
		}
	}
	defer w.finished()
	if errs := w.validate(nil); len(errs) != 0 {
		return errs
	}
	return nil
}

// ToFieldSet creates a set containing every leaf field and item mentioned, or
// validation errors, if any were encountered.
func (tv TypedValue) ToFieldSet() (*fieldpath.Set, error) {
	w := tv.toFieldSetWalker()
	defer w.finished()
	if errs := w.toFieldSet(); len(errs) != 0 {
		return nil, errs
	}
	return w.set, nil
}

// Merge returns the result of merging tv and pso ("partially specified
// object") together. Of note:
//   - No fields can be removed by this operation.
//   - If both tv and pso specify a given leaf field, the result will keep pso's
//     value.
//   - Container typed elements will have their items ordered:
//     1. like tv, if pso doesn't change anything in the container
//     2. like pso, if pso does change something in the container.
//
// tv and pso must both be of the same type (their Schema and TypeRef must
// match), or an error will be returned. Validation errors will be returned if
// the objects don't conform to the schema.
func (tv TypedValue) Merge(pso *TypedValue) (*TypedValue, error) {
	return merge(&tv, pso, ruleKeepRHS, nil)
}

var cmpwPool = sync.Pool{
	New: func() interface{} { return &compareWalker{} },
}

// Compare compares the two objects. See the comments on the `Comparison`
// struct for details on the return value.
//
// tv and rhs must both be of the same type (their Schema and TypeRef must
// match), or an error will be returned. Validation errors will be returned if
// the objects don't conform to the schema.
func (tv TypedValue) Compare(rhs *TypedValue) (c *Comparison, err error) {
	lhs := tv
	if lhs.schema != rhs.schema {
		return nil, errorf("expected objects with types from the same schema")
	}
	if !lhs.typeRef.Equals(&rhs.typeRef) {
		return nil, errorf("expected objects of the same type, but got %v and %v", lhs.typeRef, rhs.typeRef)
	}

	cmpw := cmpwPool.Get().(*compareWalker)
	defer func() {
		cmpw.lhs = nil
		cmpw.rhs = nil
		cmpw.schema = nil
		cmpw.typeRef = schema.TypeRef{}
		cmpw.comparison = nil
		cmpw.inLeaf = false

		cmpwPool.Put(cmpw)
	}()

	cmpw.lhs = lhs.value
	cmpw.rhs = rhs.value
	cmpw.schema = lhs.schema
	cmpw.typeRef = lhs.typeRef
	cmpw.comparison = &Comparison{
		Removed:  fieldpath.NewSet(),
		Modified: fieldpath.NewSet(),
		Added:    fieldpath.NewSet(),
	}
	if cmpw.allocator == nil {
		cmpw.allocator = value.NewFreelistAllocator()
	}

	errs := cmpw.compare(nil)
	if len(errs) > 0 {
		return nil, errs
	}
	return cmpw.comparison, nil
}

// RemoveItems removes each provided list or map item from the value.
func (tv TypedValue) RemoveItems(items *fieldpath.Set) *TypedValue {
	tv.value = removeItemsWithSchema(tv.value, items, tv.schema, tv.typeRef, false)
	return &tv
}

// ExtractItems returns a value with only the provided list or map items extracted from the value.
func (tv TypedValue) ExtractItems(items *fieldpath.Set, opts ...ExtractItemsOption) *TypedValue {
	options := &extractItemsOptions{}
	for _, opt := range opts {
		opt(options)
	}
	if options.appendKeyFields {
		tvPathSet, err := tv.ToFieldSet()
		if err == nil {
			keyFieldPathSet := fieldpath.NewSet()
			items.Iterate(func(path fieldpath.Path) {
				if !tvPathSet.Has(path) {
					return
				}
				for i, pe := range path {
					if pe.Key == nil {
						continue
					}
					for _, keyField := range *pe.Key {
						keyName := keyField.Name
						// Create a new slice with the same elements as path[:i+1], but set its capacity to len(path[:i+1]).
						// This ensures that appending to keyFieldPath creates a new underlying array, avoiding accidental
						// modification of the original slice (path).
						keyFieldPath := append(path[:i+1:i+1], fieldpath.PathElement{FieldName: &keyName})
						keyFieldPathSet.Insert(keyFieldPath)
					}
				}
			})
			items = items.Union(keyFieldPathSet)
		}
	}

	tv.value = removeItemsWithSchema(tv.value, items, tv.schema, tv.typeRef, true)
	return &tv
}

func (tv TypedValue) Empty() *TypedValue {
	tv.value = value.NewValueInterface(nil)
	return &tv
}

var mwPool = sync.Pool{
	New: func() interface{} { return &mergingWalker{} },
}

func merge(lhs, rhs *TypedValue, rule, postRule mergeRule) (*TypedValue, error) {
	if lhs.schema != rhs.schema {
		return nil, errorf("expected objects with types from the same schema")
	}
	if !lhs.typeRef.Equals(&rhs.typeRef) {
		return nil, errorf("expected objects of the same type, but got %v and %v", lhs.typeRef, rhs.typeRef)
	}

	mw := mwPool.Get().(*mergingWalker)
	defer func() {
		mw.lhs = nil
		mw.rhs = nil
		mw.schema = nil
		mw.typeRef = schema.TypeRef{}
		mw.rule = nil
		mw.postItemHook = nil
		mw.out = nil
		mw.inLeaf = false

		mwPool.Put(mw)
	}()

	mw.lhs = lhs.value
	mw.rhs = rhs.value
	mw.schema = lhs.schema
	mw.typeRef = lhs.typeRef
	mw.rule = rule
	mw.postItemHook = postRule
	if mw.allocator == nil {
		mw.allocator = value.NewFreelistAllocator()
	}

	errs := mw.merge(nil)
	if len(errs) > 0 {
		return nil, errs
	}

	out := &TypedValue{
		schema:  lhs.schema,
		typeRef: lhs.typeRef,
	}
	if mw.out != nil {
		out.value = value.NewValueInterface(*mw.out)
	}
	return out, nil
}
