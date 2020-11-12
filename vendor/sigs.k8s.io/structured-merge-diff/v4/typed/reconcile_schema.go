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
	"fmt"
	"sync"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
)

var fmPool = sync.Pool{
	New: func() interface{} { return &reconcileWithSchemaWalker{} },
}

func (v *reconcileWithSchemaWalker) finished() {
	v.fieldSet = nil
	v.schema = nil
	v.value = nil
	v.typeRef = schema.TypeRef{}
	v.path = nil
	v.toRemove = nil
	v.toAdd = nil
	fmPool.Put(v)
}

type reconcileWithSchemaWalker struct {
	value  *TypedValue    // root of the live object
	schema *schema.Schema // root of the live schema

	// state of node being visited by walker
	fieldSet *fieldpath.Set
	typeRef  schema.TypeRef
	path     fieldpath.Path
	isAtomic bool

	// the accumulated diff to perform to apply reconciliation
	toRemove *fieldpath.Set // paths to remove recursively
	toAdd    *fieldpath.Set // paths to add after any removals

	// Allocate only as many walkers as needed for the depth by storing them here.
	spareWalkers *[]*reconcileWithSchemaWalker
}

func (v *reconcileWithSchemaWalker) prepareDescent(pe fieldpath.PathElement, tr schema.TypeRef) *reconcileWithSchemaWalker {
	if v.spareWalkers == nil {
		// first descent.
		v.spareWalkers = &[]*reconcileWithSchemaWalker{}
	}
	var v2 *reconcileWithSchemaWalker
	if n := len(*v.spareWalkers); n > 0 {
		v2, *v.spareWalkers = (*v.spareWalkers)[n-1], (*v.spareWalkers)[:n-1]
	} else {
		v2 = &reconcileWithSchemaWalker{}
	}
	*v2 = *v
	v2.typeRef = tr
	v2.path = append(v.path, pe)
	v2.value = v.value
	return v2
}

func (v *reconcileWithSchemaWalker) finishDescent(v2 *reconcileWithSchemaWalker) {
	v2.fieldSet = nil
	v2.schema = nil
	v2.value = nil
	v2.typeRef = schema.TypeRef{}
	if cap(v2.path) < 20 { // recycle slices that do not have unexpectedly high capacity
		v2.path = v2.path[:0]
	} else {
		v2.path = nil
	}

	// merge any accumulated changes into parent walker
	if v2.toRemove != nil {
		if v.toRemove == nil {
			v.toRemove = v2.toRemove
		} else {
			v.toRemove = v.toRemove.Union(v2.toRemove)
		}
	}
	if v2.toAdd != nil {
		if v.toAdd == nil {
			v.toAdd = v2.toAdd
		} else {
			v.toAdd = v.toAdd.Union(v2.toAdd)
		}
	}
	v2.toRemove = nil
	v2.toAdd = nil

	// if the descent caused a realloc, ensure that we reuse the buffer
	// for the next sibling.
	*v.spareWalkers = append(*v.spareWalkers, v2)
}

// ReconcileFieldSetWithSchema reconciles the a field set with any changes to the
//// object's schema since the field set was written. Returns the reconciled field set, or nil of
// no changes were made to the field set.
//
// Supports:
// - changing types from atomic to granular
// - changing types from granular to atomic
func ReconcileFieldSetWithSchema(fieldset *fieldpath.Set, tv *TypedValue) (*fieldpath.Set, error) {
	v := fmPool.Get().(*reconcileWithSchemaWalker)
	v.fieldSet = fieldset
	v.value = tv

	v.schema = tv.schema
	v.typeRef = tv.typeRef

	// We don't reconcile deduced types, which are primarily for use by unstructured CRDs. Deduced
	// types do not support atomic or granular tags. Nor does the dynamic schema deduction
	// interact well with the reconcile logic.
	if v.schema == DeducedParseableType.Schema {
		return nil, nil
	}

	defer v.finished()
	errs := v.reconcile()

	if len(errs) > 0 {
		return nil, fmt.Errorf("errors reconciling field set with schema: %s", errs.Error())
	}

	// If there are any accumulated changes, apply them
	if v.toAdd != nil || v.toRemove != nil {
		out := v.fieldSet
		if v.toRemove != nil {
			out = out.RecursiveDifference(v.toRemove)
		}
		if v.toAdd != nil {
			out = out.Union(v.toAdd)
		}
		return out, nil
	}
	return nil, nil
}

func (v *reconcileWithSchemaWalker) reconcile() (errs ValidationErrors) {
	a, ok := v.schema.Resolve(v.typeRef)
	if !ok {
		errs = append(errs, errorf("could not resolve %v", v.typeRef)...)
		return
	}
	return handleAtom(a, v.typeRef, v)
}

func (v *reconcileWithSchemaWalker) doScalar(_ *schema.Scalar) (errs ValidationErrors) {
	return errs
}

func (v *reconcileWithSchemaWalker) visitListItems(t *schema.List, element *fieldpath.Set) (errs ValidationErrors) {
	handleElement := func(pe fieldpath.PathElement, isMember bool) {
		var hasChildren bool
		v2 := v.prepareDescent(pe, t.ElementType)
		v2.fieldSet, hasChildren = element.Children.Get(pe)
		v2.isAtomic = isMember && !hasChildren
		errs = append(errs, v2.reconcile()...)
		v.finishDescent(v2)
	}
	element.Children.Iterate(func(pe fieldpath.PathElement) {
		if element.Members.Has(pe) {
			return
		}
		handleElement(pe, false)
	})
	element.Members.Iterate(func(pe fieldpath.PathElement) {
		handleElement(pe, true)
	})
	return errs
}

func (v *reconcileWithSchemaWalker) doList(t *schema.List) (errs ValidationErrors) {
	// reconcile lists changed from granular to atomic
	if !v.isAtomic && t.ElementRelationship == schema.Atomic {
		v.toRemove = fieldpath.NewSet(v.path) // remove all root and all children fields
		v.toAdd = fieldpath.NewSet(v.path)    // add the root of the atomic
		return errs
	}
	// reconcile lists changed from atomic to granular
	if v.isAtomic && t.ElementRelationship == schema.Associative {
		v.toAdd, errs = buildGranularFieldSet(v.path, v.value)
		if errs != nil {
			return errs
		}
	}
	if v.fieldSet != nil {
		errs = v.visitListItems(t, v.fieldSet)
	}
	return errs
}

func (v *reconcileWithSchemaWalker) visitMapItems(t *schema.Map, element *fieldpath.Set) (errs ValidationErrors) {
	handleElement := func(pe fieldpath.PathElement, isMember bool) {
		var hasChildren bool
		if tr, ok := typeRefAtPath(t, pe); ok { // ignore fields not in the schema
			v2 := v.prepareDescent(pe, tr)
			v2.fieldSet, hasChildren = element.Children.Get(pe)
			v2.isAtomic = isMember && !hasChildren
			errs = append(errs, v2.reconcile()...)
			v.finishDescent(v2)
		}
	}
	element.Children.Iterate(func(pe fieldpath.PathElement) {
		if element.Members.Has(pe) {
			return
		}
		handleElement(pe, false)
	})
	element.Members.Iterate(func(pe fieldpath.PathElement) {
		handleElement(pe, true)
	})

	return errs
}

func (v *reconcileWithSchemaWalker) doMap(t *schema.Map) (errs ValidationErrors) {
	// reconcile maps and structs changed from granular to atomic
	if !v.isAtomic && t.ElementRelationship == schema.Atomic {
		if v.fieldSet != nil && v.fieldSet.Size() > 0 {
			v.toRemove = fieldpath.NewSet(v.path) // remove all root and all children fields
			v.toAdd = fieldpath.NewSet(v.path)    // add the root of the atomic
		}
		return errs
	}
	// reconcile maps changed from atomic to granular
	if v.isAtomic && (t.ElementRelationship == schema.Separable || t.ElementRelationship == "") {
		v.toAdd, errs = buildGranularFieldSet(v.path, v.value)
		if errs != nil {
			return errs
		}
	}
	if v.fieldSet != nil {
		errs = v.visitMapItems(t, v.fieldSet)
	}
	return errs
}

func buildGranularFieldSet(path fieldpath.Path, value *TypedValue) (*fieldpath.Set, ValidationErrors) {

	valueFieldSet, err := value.ToFieldSet()
	if err != nil {
		return nil, errorf("toFieldSet: %v", err)
	}
	if valueFieldSetAtPath, ok := fieldSetAtPath(valueFieldSet, path); ok {
		result := fieldpath.NewSet(path)
		resultAtPath := descendToPath(result, path)
		*resultAtPath = *valueFieldSetAtPath
		return result, nil
	}
	return nil, nil
}

func fieldSetAtPath(node *fieldpath.Set, path fieldpath.Path) (*fieldpath.Set, bool) {
	ok := true
	for _, pe := range path {
		if node, ok = node.Children.Get(pe); !ok {
			break
		}
	}
	return node, ok
}

func descendToPath(node *fieldpath.Set, path fieldpath.Path) *fieldpath.Set {
	for _, pe := range path {
		node = node.Children.Descend(pe)
	}
	return node
}

func typeRefAtPath(t *schema.Map, pe fieldpath.PathElement) (schema.TypeRef, bool) {
	tr := t.ElementType
	if pe.FieldName != nil {
		if sf, ok := t.FindField(*pe.FieldName); ok {
			tr = sf.Type
		}
	}
	return tr, tr != schema.TypeRef{}
}
