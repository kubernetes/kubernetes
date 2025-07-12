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
	"strings"

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/schema"
	"sigs.k8s.io/structured-merge-diff/v6/value"
)

// Comparison is the return value of a TypedValue.Compare() operation.
//
// No field will appear in more than one of the three fieldsets. If all of the
// fieldsets are empty, then the objects must have been equal.
type Comparison struct {
	// Removed contains any fields removed by rhs (the right-hand-side
	// object in the comparison).
	Removed *fieldpath.Set
	// Modified contains fields present in both objects but different.
	Modified *fieldpath.Set
	// Added contains any fields added by rhs.
	Added *fieldpath.Set
}

// IsSame returns true if the comparison returned no changes (the two
// compared objects are similar).
func (c *Comparison) IsSame() bool {
	return c.Removed.Empty() && c.Modified.Empty() && c.Added.Empty()
}

// String returns a human readable version of the comparison.
func (c *Comparison) String() string {
	bld := strings.Builder{}
	if !c.Modified.Empty() {
		bld.WriteString(fmt.Sprintf("- Modified Fields:\n%v\n", c.Modified))
	}
	if !c.Added.Empty() {
		bld.WriteString(fmt.Sprintf("- Added Fields:\n%v\n", c.Added))
	}
	if !c.Removed.Empty() {
		bld.WriteString(fmt.Sprintf("- Removed Fields:\n%v\n", c.Removed))
	}
	return bld.String()
}

// ExcludeFields fields from the compare recursively removes the fields
// from the entire comparison
func (c *Comparison) ExcludeFields(fields *fieldpath.Set) *Comparison {
	if fields == nil || fields.Empty() {
		return c
	}
	c.Removed = c.Removed.RecursiveDifference(fields)
	c.Modified = c.Modified.RecursiveDifference(fields)
	c.Added = c.Added.RecursiveDifference(fields)
	return c
}

func (c *Comparison) FilterFields(filter fieldpath.Filter) *Comparison {
	if filter == nil {
		return c
	}
	c.Removed = filter.Filter(c.Removed)
	c.Modified = filter.Filter(c.Modified)
	c.Added = filter.Filter(c.Added)
	return c
}

type compareWalker struct {
	lhs     value.Value
	rhs     value.Value
	schema  *schema.Schema
	typeRef schema.TypeRef

	// Current path that we are comparing
	path fieldpath.Path

	// Resulting comparison.
	comparison *Comparison

	// internal housekeeping--don't set when constructing.
	inLeaf bool // Set to true if we're in a "big leaf"--atomic map/list

	// Allocate only as many walkers as needed for the depth by storing them here.
	spareWalkers *[]*compareWalker

	allocator value.Allocator
}

// compare compares stuff.
func (w *compareWalker) compare(prefixFn func() string) (errs ValidationErrors) {
	if w.lhs == nil && w.rhs == nil {
		// check this condidition here instead of everywhere below.
		return errorf("at least one of lhs and rhs must be provided")
	}
	a, ok := w.schema.Resolve(w.typeRef)
	if !ok {
		return errorf("schema error: no type found matching: %v", *w.typeRef.NamedType)
	}

	alhs := deduceAtom(a, w.lhs)
	arhs := deduceAtom(a, w.rhs)

	// deduceAtom does not fix the type for nil values
	// nil is a wildcard and will accept whatever form the other operand takes
	if w.rhs == nil {
		errs = append(errs, handleAtom(alhs, w.typeRef, w)...)
	} else if w.lhs == nil || alhs.Equals(&arhs) {
		errs = append(errs, handleAtom(arhs, w.typeRef, w)...)
	} else {
		w2 := *w
		errs = append(errs, handleAtom(alhs, w.typeRef, &w2)...)
		errs = append(errs, handleAtom(arhs, w.typeRef, w)...)
	}

	if !w.inLeaf {
		if w.lhs == nil {
			w.comparison.Added.Insert(w.path)
		} else if w.rhs == nil {
			w.comparison.Removed.Insert(w.path)
		}
	}
	return errs.WithLazyPrefix(prefixFn)
}

// doLeaf should be called on leaves before descending into children, if there
// will be a descent. It modifies w.inLeaf.
func (w *compareWalker) doLeaf() {
	if w.inLeaf {
		// We're in a "big leaf", an atomic map or list. Ignore
		// subsequent leaves.
		return
	}
	w.inLeaf = true

	// We don't recurse into leaf fields for merging.
	if w.lhs == nil {
		w.comparison.Added.Insert(w.path)
	} else if w.rhs == nil {
		w.comparison.Removed.Insert(w.path)
	} else if !value.EqualsUsing(w.allocator, w.rhs, w.lhs) {
		// TODO: Equality is not sufficient for this.
		// Need to implement equality check on the value type.
		w.comparison.Modified.Insert(w.path)
	}
}

func (w *compareWalker) doScalar(t *schema.Scalar) ValidationErrors {
	// Make sure at least one side is a valid scalar.
	lerrs := validateScalar(t, w.lhs, "lhs: ")
	rerrs := validateScalar(t, w.rhs, "rhs: ")
	if len(lerrs) > 0 && len(rerrs) > 0 {
		return append(lerrs, rerrs...)
	}

	// All scalars are leaf fields.
	w.doLeaf()

	return nil
}

func (w *compareWalker) prepareDescent(pe fieldpath.PathElement, tr schema.TypeRef, cmp *Comparison) *compareWalker {
	if w.spareWalkers == nil {
		// first descent.
		w.spareWalkers = &[]*compareWalker{}
	}
	var w2 *compareWalker
	if n := len(*w.spareWalkers); n > 0 {
		w2, *w.spareWalkers = (*w.spareWalkers)[n-1], (*w.spareWalkers)[:n-1]
	} else {
		w2 = &compareWalker{}
	}
	*w2 = *w
	w2.typeRef = tr
	w2.path = append(w2.path, pe)
	w2.lhs = nil
	w2.rhs = nil
	w2.comparison = cmp
	return w2
}

func (w *compareWalker) finishDescent(w2 *compareWalker) {
	// if the descent caused a realloc, ensure that we reuse the buffer
	// for the next sibling.
	w.path = w2.path[:len(w2.path)-1]
	*w.spareWalkers = append(*w.spareWalkers, w2)
}

func (w *compareWalker) derefMap(prefix string, v value.Value) (value.Map, ValidationErrors) {
	if v == nil {
		return nil, nil
	}
	m, err := mapValue(w.allocator, v)
	if err != nil {
		return nil, errorf("%v: %v", prefix, err)
	}
	return m, nil
}

func (w *compareWalker) visitListItems(t *schema.List, lhs, rhs value.List) (errs ValidationErrors) {
	rLen := 0
	if rhs != nil {
		rLen = rhs.Length()
	}
	lLen := 0
	if lhs != nil {
		lLen = lhs.Length()
	}

	maxLength := rLen
	if lLen > maxLength {
		maxLength = lLen
	}
	// Contains all the unique PEs between lhs and rhs, exactly once.
	// Order doesn't matter since we're just tracking ownership in a set.
	allPEs := make([]fieldpath.PathElement, 0, maxLength)

	// Gather all the elements from lhs, indexed by PE, in a list for duplicates.
	lValues := fieldpath.MakePathElementMap(lLen)
	for i := 0; i < lLen; i++ {
		child := lhs.At(i)
		pe, err := listItemToPathElement(w.allocator, w.schema, t, child)
		if err != nil {
			errs = append(errs, errorf("element %v: %v", i, err.Error())...)
			// If we can't construct the path element, we can't
			// even report errors deeper in the schema, so bail on
			// this element.
			continue
		}

		if v, found := lValues.Get(pe); found {
			list := v.([]value.Value)
			lValues.Insert(pe, append(list, child))
		} else {
			lValues.Insert(pe, []value.Value{child})
			allPEs = append(allPEs, pe)
		}
	}

	// Gather all the elements from rhs, indexed by PE, in a list for duplicates.
	rValues := fieldpath.MakePathElementMap(rLen)
	for i := 0; i < rLen; i++ {
		rValue := rhs.At(i)
		pe, err := listItemToPathElement(w.allocator, w.schema, t, rValue)
		if err != nil {
			errs = append(errs, errorf("element %v: %v", i, err.Error())...)
			// If we can't construct the path element, we can't
			// even report errors deeper in the schema, so bail on
			// this element.
			continue
		}
		if v, found := rValues.Get(pe); found {
			list := v.([]value.Value)
			rValues.Insert(pe, append(list, rValue))
		} else {
			rValues.Insert(pe, []value.Value{rValue})
			if _, found := lValues.Get(pe); !found {
				allPEs = append(allPEs, pe)
			}
		}
	}

	for _, pe := range allPEs {
		lList := []value.Value(nil)
		if l, ok := lValues.Get(pe); ok {
			lList = l.([]value.Value)
		}
		rList := []value.Value(nil)
		if l, ok := rValues.Get(pe); ok {
			rList = l.([]value.Value)
		}

		switch {
		case len(lList) == 0 && len(rList) == 0:
			// We shouldn't be here anyway.
			return
		// Normal use-case:
		// We have no duplicates for this PE, compare items one-to-one.
		case len(lList) <= 1 && len(rList) <= 1:
			lValue := value.Value(nil)
			if len(lList) != 0 {
				lValue = lList[0]
			}
			rValue := value.Value(nil)
			if len(rList) != 0 {
				rValue = rList[0]
			}
			errs = append(errs, w.compareListItem(t, pe, lValue, rValue)...)
		// Duplicates before & after use-case:
		// Compare the duplicates lists as if they were atomic, mark modified if they changed.
		case len(lList) >= 2 && len(rList) >= 2:
			listEqual := func(lList, rList []value.Value) bool {
				if len(lList) != len(rList) {
					return false
				}
				for i := range lList {
					if !value.Equals(lList[i], rList[i]) {
						return false
					}
				}
				return true
			}
			if !listEqual(lList, rList) {
				w.comparison.Modified.Insert(append(w.path, pe))
			}
		// Duplicates before & not anymore use-case:
		// Rcursively add new non-duplicate items, Remove duplicate marker,
		case len(lList) >= 2:
			if len(rList) != 0 {
				errs = append(errs, w.compareListItem(t, pe, nil, rList[0])...)
			}
			w.comparison.Removed.Insert(append(w.path, pe))
		// New duplicates use-case:
		// Recursively remove old non-duplicate items, add duplicate marker.
		case len(rList) >= 2:
			if len(lList) != 0 {
				errs = append(errs, w.compareListItem(t, pe, lList[0], nil)...)
			}
			w.comparison.Added.Insert(append(w.path, pe))
		}
	}

	return
}

func (w *compareWalker) indexListPathElements(t *schema.List, list value.List) ([]fieldpath.PathElement, fieldpath.PathElementValueMap, ValidationErrors) {
	var errs ValidationErrors
	length := 0
	if list != nil {
		length = list.Length()
	}
	observed := fieldpath.MakePathElementValueMap(length)
	pes := make([]fieldpath.PathElement, 0, length)
	for i := 0; i < length; i++ {
		child := list.At(i)
		pe, err := listItemToPathElement(w.allocator, w.schema, t, child)
		if err != nil {
			errs = append(errs, errorf("element %v: %v", i, err.Error())...)
			// If we can't construct the path element, we can't
			// even report errors deeper in the schema, so bail on
			// this element.
			continue
		}
		// Ignore repeated occurences of `pe`.
		if _, found := observed.Get(pe); found {
			continue
		}
		observed.Insert(pe, child)
		pes = append(pes, pe)
	}
	return pes, observed, errs
}

func (w *compareWalker) compareListItem(t *schema.List, pe fieldpath.PathElement, lChild, rChild value.Value) ValidationErrors {
	w2 := w.prepareDescent(pe, t.ElementType, w.comparison)
	w2.lhs = lChild
	w2.rhs = rChild
	errs := w2.compare(pe.String)
	w.finishDescent(w2)
	return errs
}

func (w *compareWalker) derefList(prefix string, v value.Value) (value.List, ValidationErrors) {
	if v == nil {
		return nil, nil
	}
	l, err := listValue(w.allocator, v)
	if err != nil {
		return nil, errorf("%v: %v", prefix, err)
	}
	return l, nil
}

func (w *compareWalker) doList(t *schema.List) (errs ValidationErrors) {
	lhs, _ := w.derefList("lhs: ", w.lhs)
	if lhs != nil {
		defer w.allocator.Free(lhs)
	}
	rhs, _ := w.derefList("rhs: ", w.rhs)
	if rhs != nil {
		defer w.allocator.Free(rhs)
	}

	// If both lhs and rhs are empty/null, treat it as a
	// leaf: this helps preserve the empty/null
	// distinction.
	emptyPromoteToLeaf := (lhs == nil || lhs.Length() == 0) && (rhs == nil || rhs.Length() == 0)

	if t.ElementRelationship == schema.Atomic || emptyPromoteToLeaf {
		w.doLeaf()
		return nil
	}

	if lhs == nil && rhs == nil {
		return nil
	}

	errs = w.visitListItems(t, lhs, rhs)

	return errs
}

func (w *compareWalker) visitMapItem(t *schema.Map, out map[string]interface{}, key string, lhs, rhs value.Value) (errs ValidationErrors) {
	fieldType := t.ElementType
	if sf, ok := t.FindField(key); ok {
		fieldType = sf.Type
	}
	pe := fieldpath.PathElement{FieldName: &key}
	w2 := w.prepareDescent(pe, fieldType, w.comparison)
	w2.lhs = lhs
	w2.rhs = rhs
	errs = append(errs, w2.compare(pe.String)...)
	w.finishDescent(w2)
	return errs
}

func (w *compareWalker) visitMapItems(t *schema.Map, lhs, rhs value.Map) (errs ValidationErrors) {
	out := map[string]interface{}{}

	value.MapZipUsing(w.allocator, lhs, rhs, value.Unordered, func(key string, lhsValue, rhsValue value.Value) bool {
		errs = append(errs, w.visitMapItem(t, out, key, lhsValue, rhsValue)...)
		return true
	})

	return errs
}

func (w *compareWalker) doMap(t *schema.Map) (errs ValidationErrors) {
	lhs, _ := w.derefMap("lhs: ", w.lhs)
	if lhs != nil {
		defer w.allocator.Free(lhs)
	}
	rhs, _ := w.derefMap("rhs: ", w.rhs)
	if rhs != nil {
		defer w.allocator.Free(rhs)
	}
	// If both lhs and rhs are empty/null, treat it as a
	// leaf: this helps preserve the empty/null
	// distinction.
	emptyPromoteToLeaf := (lhs == nil || lhs.Empty()) && (rhs == nil || rhs.Empty())

	if t.ElementRelationship == schema.Atomic || emptyPromoteToLeaf {
		w.doLeaf()
		return nil
	}

	if lhs == nil && rhs == nil {
		return nil
	}

	errs = append(errs, w.visitMapItems(t, lhs, rhs)...)

	return errs
}
