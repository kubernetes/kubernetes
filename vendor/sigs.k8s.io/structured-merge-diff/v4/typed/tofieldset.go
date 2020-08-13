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

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

var tPool = sync.Pool{
	New: func() interface{} { return &toFieldSetWalker{} },
}

func (tv TypedValue) toFieldSetWalker() *toFieldSetWalker {
	v := tPool.Get().(*toFieldSetWalker)
	v.value = tv.value
	v.schema = tv.schema
	v.typeRef = tv.typeRef
	v.set = &fieldpath.Set{}
	v.allocator = value.NewFreelistAllocator()
	return v
}

func (v *toFieldSetWalker) finished() {
	v.schema = nil
	v.typeRef = schema.TypeRef{}
	v.path = nil
	v.set = nil
	tPool.Put(v)
}

type toFieldSetWalker struct {
	value   value.Value
	schema  *schema.Schema
	typeRef schema.TypeRef

	set  *fieldpath.Set
	path fieldpath.Path

	// Allocate only as many walkers as needed for the depth by storing them here.
	spareWalkers *[]*toFieldSetWalker
	allocator    value.Allocator
}

func (v *toFieldSetWalker) prepareDescent(pe fieldpath.PathElement, tr schema.TypeRef) *toFieldSetWalker {
	if v.spareWalkers == nil {
		// first descent.
		v.spareWalkers = &[]*toFieldSetWalker{}
	}
	var v2 *toFieldSetWalker
	if n := len(*v.spareWalkers); n > 0 {
		v2, *v.spareWalkers = (*v.spareWalkers)[n-1], (*v.spareWalkers)[:n-1]
	} else {
		v2 = &toFieldSetWalker{}
	}
	*v2 = *v
	v2.typeRef = tr
	v2.path = append(v2.path, pe)
	return v2
}

func (v *toFieldSetWalker) finishDescent(v2 *toFieldSetWalker) {
	// if the descent caused a realloc, ensure that we reuse the buffer
	// for the next sibling.
	v.path = v2.path[:len(v2.path)-1]
	*v.spareWalkers = append(*v.spareWalkers, v2)
}

func (v *toFieldSetWalker) toFieldSet() ValidationErrors {
	return resolveSchema(v.schema, v.typeRef, v.value, v)
}

func (v *toFieldSetWalker) doScalar(t *schema.Scalar) ValidationErrors {
	v.set.Insert(v.path)

	return nil
}

func (v *toFieldSetWalker) visitListItems(t *schema.List, list value.List) (errs ValidationErrors) {
	for i := 0; i < list.Length(); i++ {
		child := list.At(i)
		pe, _ := listItemToPathElement(v.allocator, t, i, child)
		v2 := v.prepareDescent(pe, t.ElementType)
		v2.value = child
		errs = append(errs, v2.toFieldSet()...)

		v2.set.Insert(v2.path)
		v.finishDescent(v2)
	}
	return errs
}

func (v *toFieldSetWalker) doList(t *schema.List) (errs ValidationErrors) {
	list, _ := listValue(v.allocator, v.value)
	if list != nil {
		defer v.allocator.Free(list)
	}
	if t.ElementRelationship == schema.Atomic {
		v.set.Insert(v.path)
		return nil
	}

	if list == nil {
		return nil
	}

	errs = v.visitListItems(t, list)

	return errs
}

func (v *toFieldSetWalker) visitMapItems(t *schema.Map, m value.Map) (errs ValidationErrors) {
	m.Iterate(func(key string, val value.Value) bool {
		pe := fieldpath.PathElement{FieldName: &key}

		tr := t.ElementType
		if sf, ok := t.FindField(key); ok {
			tr = sf.Type
		}
		v2 := v.prepareDescent(pe, tr)
		v2.value = val
		errs = append(errs, v2.toFieldSet()...)
		if _, ok := t.FindField(key); !ok {
			v2.set.Insert(v2.path)
		}
		v.finishDescent(v2)
		return true
	})
	return errs
}

func (v *toFieldSetWalker) doMap(t *schema.Map) (errs ValidationErrors) {
	m, _ := mapValue(v.allocator, v.value)
	if m != nil {
		defer v.allocator.Free(m)
	}
	if t.ElementRelationship == schema.Atomic {
		v.set.Insert(v.path)
		return nil
	}

	if m == nil {
		return nil
	}

	errs = v.visitMapItems(t, m)

	return errs
}
