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
	"reflect"

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/value"
)

// deducedTypedValue holds a value and guesses what it is and what to
// do with it.
type deducedTypedValue struct {
	value value.Value
}

// AsTypedDeduced is going to generate it's own type definition based on
// the content of the object. This is useful for CRDs that don't have a
// validation field.
func AsTypedDeduced(v value.Value) TypedValue {
	return deducedTypedValue{value: v}
}

func (dv deducedTypedValue) AsValue() *value.Value {
	return &dv.value
}

func (deducedTypedValue) Validate() error {
	return nil
}

func (dv deducedTypedValue) ToFieldSet() (*fieldpath.Set, error) {
	set := fieldpath.NewSet()
	fieldsetDeduced(dv.value, fieldpath.Path{}, set)
	return set, nil
}

func fieldsetDeduced(v value.Value, path fieldpath.Path, set *fieldpath.Set) {
	if v.MapValue == nil {
		set.Insert(path)
		return
	}

	// We have a map.
	// copy the existing path, append each item, and recursively call.
	for i := range v.MapValue.Items {
		child := v.MapValue.Items[i]
		np := path.Copy()
		np = append(np, fieldpath.PathElement{FieldName: &child.Name})
		fieldsetDeduced(child.Value, np, set)
	}
}

func (dv deducedTypedValue) Merge(pso TypedValue) (TypedValue, error) {
	tpso, ok := pso.(deducedTypedValue)
	if !ok {
		return nil, errorFormatter{}.
			errorf("can't merge deducedTypedValue with %T", tpso)
	}
	return AsTypedDeduced(mergeDeduced(dv.value, tpso.value)), nil
}

func mergeDeduced(lhs, rhs value.Value) value.Value {
	// If both sides are maps, merge them, otherwise return right
	// side.
	if rhs.MapValue == nil || lhs.MapValue == nil {
		return rhs
	}

	v := value.Value{MapValue: &value.Map{}}
	for i := range lhs.MapValue.Items {
		child := lhs.MapValue.Items[i]
		v.MapValue.Set(child.Name, child.Value)
	}
	for i := range rhs.MapValue.Items {
		child := rhs.MapValue.Items[i]
		if sub, ok := v.MapValue.Get(child.Name); ok {
			new := mergeDeduced(sub.Value, child.Value)
			v.MapValue.Set(child.Name, new)
		} else {
			v.MapValue.Set(child.Name, child.Value)
		}
	}
	return v
}

func (dv deducedTypedValue) Compare(rhs TypedValue) (c *Comparison, err error) {
	trhs, ok := rhs.(deducedTypedValue)
	if !ok {
		return nil, errorFormatter{}.
			errorf("can't merge deducedTypedValue with %T", rhs)
	}

	c = &Comparison{
		Removed:  fieldpath.NewSet(),
		Modified: fieldpath.NewSet(),
		Added:    fieldpath.NewSet(),
	}

	added(dv.value, trhs.value, fieldpath.Path{}, c.Added)
	added(trhs.value, dv.value, fieldpath.Path{}, c.Removed)
	modified(dv.value, trhs.value, fieldpath.Path{}, c.Modified)

	merge, err := dv.Merge(rhs)
	if err != nil {
		return nil, err
	}
	c.Merged = merge
	return c, nil
}

func added(lhs, rhs value.Value, path fieldpath.Path, set *fieldpath.Set) {
	if lhs.MapValue == nil && rhs.MapValue == nil {
		// Both non-maps, nothing added, do nothing.
	} else if lhs.MapValue == nil && rhs.MapValue != nil {
		// From leaf to map, add leaf fields of map.
		fieldsetDeduced(rhs, path, set)
	} else if lhs.MapValue != nil && rhs.MapValue == nil {
		// Went from map to field, add field.
		set.Insert(path)
	} else {
		// Both are maps.
		for i := range rhs.MapValue.Items {
			child := rhs.MapValue.Items[i]
			np := path.Copy()
			np = append(np, fieldpath.PathElement{FieldName: &child.Name})

			if v, ok := lhs.MapValue.Get(child.Name); ok {
				added(v.Value, child.Value, np, set)
			} else {
				fieldsetDeduced(child.Value, np, set)
			}
		}
	}
}

func modified(lhs, rhs value.Value, path fieldpath.Path, set *fieldpath.Set) {
	if lhs.MapValue == nil && rhs.MapValue == nil {
		if !reflect.DeepEqual(lhs, rhs) {
			set.Insert(path)
		}
	} else if lhs.MapValue != nil && rhs.MapValue != nil {
		// Both are maps.
		for i := range rhs.MapValue.Items {
			child := rhs.MapValue.Items[i]

			v, ok := lhs.MapValue.Get(child.Name)
			if !ok {
				continue
			}

			np := path.Copy()
			np = append(np, fieldpath.PathElement{FieldName: &child.Name})
			modified(v.Value, child.Value, np, set)
		}
	}
}

// RemoveItems does nothing because all lists in a deducedTypedValue are considered atomic,
// and there are no maps because it is indistinguishable from a struct.
func (dv deducedTypedValue) RemoveItems(_ *fieldpath.Set) TypedValue {
	return dv
}
