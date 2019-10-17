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

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/schema"
	"sigs.k8s.io/structured-merge-diff/value"
)

var vPool = sync.Pool{
	New: func() interface{} { return &validatingObjectWalker{} },
}

func (tv TypedValue) walker() *validatingObjectWalker {
	v := vPool.Get().(*validatingObjectWalker)
	v.value = tv.value
	v.schema = tv.schema
	v.typeRef = tv.typeRef
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

	// Allocate only as many walkers as needed for the depth by storing them here.
	spareWalkers *[]*validatingObjectWalker
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

func (v *validatingObjectWalker) validate() ValidationErrors {
	return resolveSchema(v.schema, v.typeRef, &v.value, v)
}

func validateScalar(t *schema.Scalar, v *value.Value, prefix string) (errs ValidationErrors) {
	if v == nil {
		return nil
	}
	if *v == nil {
		return nil
	}
	switch *t {
	case schema.Numeric:
		if !value.IsFloat(*v) && !value.IsInt(*v) {
			// TODO: should the schema separate int and float?
			return errorf("%vexpected numeric (int or float), got %T", prefix, *v)
		}
	case schema.String:
		if !value.IsString(*v) {
			return errorf("%vexpected string, got %#v", prefix, *v)
		}
	case schema.Boolean:
		if !value.IsBool(*v) {
			return errorf("%vexpected boolean, got %v", prefix, *v)
		}
	}
	return nil
}

func (v *validatingObjectWalker) doScalar(t *schema.Scalar) ValidationErrors {
	if errs := validateScalar(t, &v.value, ""); len(errs) > 0 {
		return errs
	}
	return nil
}

func (v *validatingObjectWalker) visitListItems(t *schema.List, list []interface{}) (errs ValidationErrors) {
	observedKeys := fieldpath.MakePathElementSet(len(list))
	for i, child := range list {
		var pe fieldpath.PathElement
		if t.ElementRelationship != schema.Associative {
			pe.Index = &i
		} else {
			var err error
			pe, err = listItemToPathElement(t, i, child)
			if err != nil {
				errs = append(errs, errorf("element %v: %v", i, err.Error())...)
				// If we can't construct the path element, we can't
				// even report errors deeper in the schema, so bail on
				// this element.
				continue
			}
			if observedKeys.Has(pe) {
				errs = append(errs, errorf("duplicate entries for key %v", pe.String())...)
			}
			observedKeys.Insert(pe)
		}
		v2 := v.prepareDescent(t.ElementType)
		v2.value = child
		if newErrs := v2.validate(); len(newErrs) != 0 {
			errs = append(errs, newErrs.WithPrefix(pe.String())...)
		}
		v.finishDescent(v2)
	}
	return errs
}

func (v *validatingObjectWalker) doList(t *schema.List) (errs ValidationErrors) {
	list, err := listValue(v.value)
	if err != nil {
		return errorf(err.Error())
	}

	if list == nil {
		return nil
	}

	errs = v.visitListItems(t, list)

	return errs
}

func (v *validatingObjectWalker) visitMapItem(t *schema.Map, key string, val value.Value) (errs ValidationErrors) {
	tr := t.ElementType
	if sf, ok := t.FindField(key); ok {
		tr = sf.Type
	}
	v2 := v.prepareDescent(tr)
	v2.value = val
	if newErrs := v2.validate(); len(newErrs) != 0 {
		errs = append(errs, newErrs.WithPrefix(fieldpath.PathElement{FieldName: &key}.String())...)
	}
	v.finishDescent(v2)
	return errs
}

func (v *validatingObjectWalker) visitMapItems(t *schema.Map, m value.Map) (errs ValidationErrors) {
	// Avoiding the closure on m.Iterate here significantly improves
	// performance, so we have to switch on each type of maps.
	switch mt := m.(type) {
	case value.MapString:
		for key, val := range mt {
			errs = append(errs, v.visitMapItem(t, key, val)...)
		}
	case value.MapInterface:
		for key, val := range mt {
			if k, ok := key.(string); !ok {
				continue
			} else {
				errs = append(errs, v.visitMapItem(t, k, val)...)
			}
		}
	}
	return errs
}

func (v *validatingObjectWalker) doMap(t *schema.Map) (errs ValidationErrors) {
	m, err := mapValue(v.value)
	if err != nil {
		return errorf(err.Error())
	}

	if m == nil {
		return nil
	}

	errs = v.visitMapItems(t, m)

	return errs
}
