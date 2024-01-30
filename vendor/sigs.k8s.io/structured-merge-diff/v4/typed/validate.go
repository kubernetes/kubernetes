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

var vPool = sync.Pool{
	New: func() interface{} { return &validatingObjectWalker{} },
}

func (tv TypedValue) walker() *validatingObjectWalker {
	v := vPool.Get().(*validatingObjectWalker)
	v.value = tv.value
	v.schema = tv.schema
	v.typeRef = tv.typeRef
	v.allowDuplicates = false
	if v.allocator == nil {
		v.allocator = value.NewFreelistAllocator()
	}
	return v
}

func (v *validatingObjectWalker) finished() {
	v.schema = nil
	v.typeRef = schema.TypeRef{}
	vPool.Put(v)
}

type validatingObjectWalker struct {
	value   value.Value
	schema  *schema.Schema
	typeRef schema.TypeRef
	// If set to true, duplicates will be allowed in
	// associativeLists/sets.
	allowDuplicates bool

	// Allocate only as many walkers as needed for the depth by storing them here.
	spareWalkers *[]*validatingObjectWalker
	allocator    value.Allocator
}

func (v *validatingObjectWalker) prepareDescent(tr schema.TypeRef) *validatingObjectWalker {
	if v.spareWalkers == nil {
		// first descent.
		v.spareWalkers = &[]*validatingObjectWalker{}
	}
	var v2 *validatingObjectWalker
	if n := len(*v.spareWalkers); n > 0 {
		v2, *v.spareWalkers = (*v.spareWalkers)[n-1], (*v.spareWalkers)[:n-1]
	} else {
		v2 = &validatingObjectWalker{}
	}
	*v2 = *v
	v2.typeRef = tr
	return v2
}

func (v *validatingObjectWalker) finishDescent(v2 *validatingObjectWalker) {
	// if the descent caused a realloc, ensure that we reuse the buffer
	// for the next sibling.
	*v.spareWalkers = append(*v.spareWalkers, v2)
}

func (v *validatingObjectWalker) validate(prefixFn func() string) ValidationErrors {
	return resolveSchema(v.schema, v.typeRef, v.value, v).WithLazyPrefix(prefixFn)
}

func validateScalar(t *schema.Scalar, v value.Value, prefix string) (errs ValidationErrors) {
	if v == nil {
		return nil
	}
	if v.IsNull() {
		return nil
	}
	switch *t {
	case schema.Numeric:
		if !v.IsFloat() && !v.IsInt() {
			// TODO: should the schema separate int and float?
			return errorf("%vexpected numeric (int or float), got %T", prefix, v.Unstructured())
		}
	case schema.String:
		if !v.IsString() {
			return errorf("%vexpected string, got %#v", prefix, v)
		}
	case schema.Boolean:
		if !v.IsBool() {
			return errorf("%vexpected boolean, got %v", prefix, v)
		}
	case schema.Untyped:
		if !v.IsFloat() && !v.IsInt() && !v.IsString() && !v.IsBool() {
			return errorf("%vexpected any scalar, got %v", prefix, v)
		}
	default:
		return errorf("%vunexpected scalar type in schema: %v", prefix, *t)
	}
	return nil
}

func (v *validatingObjectWalker) doScalar(t *schema.Scalar) ValidationErrors {
	if errs := validateScalar(t, v.value, ""); len(errs) > 0 {
		return errs
	}
	return nil
}

func (v *validatingObjectWalker) visitListItems(t *schema.List, list value.List) (errs ValidationErrors) {
	observedKeys := fieldpath.MakePathElementSet(list.Length())
	for i := 0; i < list.Length(); i++ {
		child := list.AtUsing(v.allocator, i)
		defer v.allocator.Free(child)
		var pe fieldpath.PathElement
		if t.ElementRelationship != schema.Associative {
			pe.Index = &i
		} else {
			var err error
			pe, err = listItemToPathElement(v.allocator, v.schema, t, child)
			if err != nil {
				errs = append(errs, errorf("element %v: %v", i, err.Error())...)
				// If we can't construct the path element, we can't
				// even report errors deeper in the schema, so bail on
				// this element.
				return
			}
			if observedKeys.Has(pe) && !v.allowDuplicates {
				errs = append(errs, errorf("duplicate entries for key %v", pe.String())...)
			}
			observedKeys.Insert(pe)
		}
		v2 := v.prepareDescent(t.ElementType)
		v2.value = child
		errs = append(errs, v2.validate(pe.String)...)
		v.finishDescent(v2)
	}
	return errs
}

func (v *validatingObjectWalker) doList(t *schema.List) (errs ValidationErrors) {
	list, err := listValue(v.allocator, v.value)
	if err != nil {
		return errorf(err.Error())
	}

	if list == nil {
		return nil
	}

	defer v.allocator.Free(list)
	errs = v.visitListItems(t, list)

	return errs
}

func (v *validatingObjectWalker) visitMapItems(t *schema.Map, m value.Map) (errs ValidationErrors) {
	m.IterateUsing(v.allocator, func(key string, val value.Value) bool {
		pe := fieldpath.PathElement{FieldName: &key}
		tr := t.ElementType
		if sf, ok := t.FindField(key); ok {
			tr = sf.Type
		} else if (t.ElementType == schema.TypeRef{}) {
			errs = append(errs, errorf("field not declared in schema").WithPrefix(pe.String())...)
			return false
		}
		v2 := v.prepareDescent(tr)
		v2.value = val
		// Giving pe.String as a parameter actually increases the allocations.
		errs = append(errs, v2.validate(func() string { return pe.String() })...)
		v.finishDescent(v2)
		return true
	})
	return errs
}

func (v *validatingObjectWalker) doMap(t *schema.Map) (errs ValidationErrors) {
	m, err := mapValue(v.allocator, v.value)
	if err != nil {
		return errorf(err.Error())
	}
	if m == nil {
		return nil
	}
	defer v.allocator.Free(m)
	errs = v.visitMapItems(t, m)

	return errs
}
