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
	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/schema"
	"sigs.k8s.io/structured-merge-diff/value"
)

type mergingWalker struct {
	errorFormatter
	lhs     *value.Value
	rhs     *value.Value
	schema  *schema.Schema
	typeRef schema.TypeRef

	// How to merge. Called after schema validation for all leaf fields.
	rule mergeRule

	// If set, called after non-leaf items have been merged. (`out` is
	// probably already set.)
	postItemHook mergeRule

	// output of the merge operation (nil if none)
	out *value.Value

	// internal housekeeping--don't set when constructing.
	inLeaf bool // Set to true if we're in a "big leaf"--atomic map/list

	// Allocate only as many walkers as needed for the depth by storing them here.
	spareWalkers *[]*mergingWalker
}

// merge rules examine w.lhs and w.rhs (up to one of which may be nil) and
// optionally set w.out. If lhs and rhs are both set, they will be of
// comparable type.
type mergeRule func(w *mergingWalker)

var (
	ruleKeepRHS = mergeRule(func(w *mergingWalker) {
		if w.rhs != nil {
			v := *w.rhs
			w.out = &v
		} else if w.lhs != nil {
			v := *w.lhs
			w.out = &v
		}
	})
)

// merge sets w.out.
func (w *mergingWalker) merge() (errs ValidationErrors) {
	if w.lhs == nil && w.rhs == nil {
		// check this condidition here instead of everywhere below.
		return w.errorf("at least one of lhs and rhs must be provided")
	}
	a, ok := w.schema.Resolve(w.typeRef)
	if !ok {
		return w.errorf("schema error: no type found matching: %v", *w.typeRef.NamedType)
	}

	alhs := deduceAtom(a, w.lhs)
	arhs := deduceAtom(a, w.rhs)
	if alhs.Equals(arhs) {
		errs = append(errs, handleAtom(arhs, w.typeRef, w)...)
	} else {
		w2 := *w
		errs = append(errs, handleAtom(alhs, w.typeRef, &w2)...)
		errs = append(errs, handleAtom(arhs, w.typeRef, w)...)
	}

	if !w.inLeaf && w.postItemHook != nil {
		w.postItemHook(w)
	}
	return errs
}

// doLeaf should be called on leaves before descending into children, if there
// will be a descent. It modifies w.inLeaf.
func (w *mergingWalker) doLeaf() {
	if w.inLeaf {
		// We're in a "big leaf", an atomic map or list. Ignore
		// subsequent leaves.
		return
	}
	w.inLeaf = true

	// We don't recurse into leaf fields for merging.
	w.rule(w)
}

func (w *mergingWalker) doScalar(t *schema.Scalar) (errs ValidationErrors) {
	errs = append(errs, w.validateScalar(t, w.lhs, "lhs: ")...)
	errs = append(errs, w.validateScalar(t, w.rhs, "rhs: ")...)
	if len(errs) > 0 {
		return errs
	}

	// All scalars are leaf fields.
	w.doLeaf()

	return nil
}

func (w *mergingWalker) prepareDescent(pe fieldpath.PathElement, tr schema.TypeRef) *mergingWalker {
	if w.spareWalkers == nil {
		// first descent.
		w.spareWalkers = &[]*mergingWalker{}
	}
	var w2 *mergingWalker
	if n := len(*w.spareWalkers); n > 0 {
		w2, *w.spareWalkers = (*w.spareWalkers)[n-1], (*w.spareWalkers)[:n-1]
	} else {
		w2 = &mergingWalker{}
	}
	*w2 = *w
	w2.typeRef = tr
	w2.errorFormatter.descend(pe)
	w2.lhs = nil
	w2.rhs = nil
	w2.out = nil
	return w2
}

func (w *mergingWalker) finishDescent(w2 *mergingWalker) {
	// if the descent caused a realloc, ensure that we reuse the buffer
	// for the next sibling.
	w.errorFormatter = w2.errorFormatter.parent()
	*w.spareWalkers = append(*w.spareWalkers, w2)
}

func (w *mergingWalker) derefMap(prefix string, v *value.Value, dest **value.Map) (errs ValidationErrors) {
	// taking dest as input so that it can be called as a one-liner with
	// append.
	if v == nil {
		return nil
	}
	m, err := mapValue(*v)
	if err != nil {
		return w.prefixError(prefix, err)
	}
	*dest = m
	return nil
}

func (w *mergingWalker) visitListItems(t *schema.List, lhs, rhs *value.List) (errs ValidationErrors) {
	out := &value.List{}

	// TODO: ordering is totally wrong.
	// TODO: might as well make the map order work the same way.

	// This is a cheap hack to at least make the output order stable.
	rhsOrder := []fieldpath.PathElement{}

	// First, collect all RHS children.
	var observedRHS fieldpath.PathElementValueMap
	if rhs != nil {
		observedRHS = fieldpath.MakePathElementValueMap(len(rhs.Items))
		for i, child := range rhs.Items {
			pe, err := listItemToPathElement(t, i, child)
			if err != nil {
				errs = append(errs, w.errorf("rhs: element %v: %v", i, err.Error())...)
				// If we can't construct the path element, we can't
				// even report errors deeper in the schema, so bail on
				// this element.
				continue
			}
			if _, ok := observedRHS.Get(pe); ok {
				errs = append(errs, w.errorf("rhs: duplicate entries for key %v", pe.String())...)
			}
			observedRHS.Insert(pe, child)
			rhsOrder = append(rhsOrder, pe)
		}
	}

	// Then merge with LHS children.
	var observedLHS fieldpath.PathElementSet
	if lhs != nil {
		observedLHS = fieldpath.MakePathElementSet(len(lhs.Items))
		for i, child := range lhs.Items {
			pe, err := listItemToPathElement(t, i, child)
			if err != nil {
				errs = append(errs, w.errorf("lhs: element %v: %v", i, err.Error())...)
				// If we can't construct the path element, we can't
				// even report errors deeper in the schema, so bail on
				// this element.
				continue
			}
			if observedLHS.Has(pe) {
				errs = append(errs, w.errorf("lhs: duplicate entries for key %v", pe.String())...)
				continue
			}
			observedLHS.Insert(pe)
			w2 := w.prepareDescent(pe, t.ElementType)
			w2.lhs = &child
			if rchild, ok := observedRHS.Get(pe); ok {
				w2.rhs = &rchild
			}
			if newErrs := w2.merge(); len(newErrs) > 0 {
				errs = append(errs, newErrs...)
			} else if w2.out != nil {
				out.Items = append(out.Items, *w2.out)
			}
			w.finishDescent(w2)
		}
	}

	for _, pe := range rhsOrder {
		if observedLHS.Has(pe) {
			continue
		}
		value, _ := observedRHS.Get(pe)
		w2 := w.prepareDescent(pe, t.ElementType)
		w2.rhs = &value
		if newErrs := w2.merge(); len(newErrs) > 0 {
			errs = append(errs, newErrs...)
		} else if w2.out != nil {
			out.Items = append(out.Items, *w2.out)
		}
		w.finishDescent(w2)
	}

	if len(out.Items) > 0 {
		w.out = &value.Value{ListValue: out}
	}
	return errs
}

func (w *mergingWalker) derefList(prefix string, v *value.Value, dest **value.List) (errs ValidationErrors) {
	// taking dest as input so that it can be called as a one-liner with
	// append.
	if v == nil {
		return nil
	}
	l, err := listValue(*v)
	if err != nil {
		return w.prefixError(prefix, err)
	}
	*dest = l
	return nil
}

func (w *mergingWalker) doList(t *schema.List) (errs ValidationErrors) {
	var lhs, rhs *value.List
	w.derefList("lhs: ", w.lhs, &lhs)
	w.derefList("rhs: ", w.rhs, &rhs)

	// If both lhs and rhs are empty/null, treat it as a
	// leaf: this helps preserve the empty/null
	// distinction.
	emptyPromoteToLeaf := (lhs == nil || len(lhs.Items) == 0) &&
		(rhs == nil || len(rhs.Items) == 0)

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

func (w *mergingWalker) visitMapItems(t *schema.Map, lhs, rhs *value.Map) (errs ValidationErrors) {
	out := &value.Map{}

	if lhs != nil {
		for i := range lhs.Items {
			litem := &lhs.Items[i]
			fieldType := t.ElementType
			if sf, ok := t.FindField(litem.Name); ok {
				fieldType = sf.Type
			}
			w2 := w.prepareDescent(fieldpath.PathElement{FieldName: &litem.Name}, fieldType)
			w2.lhs = &litem.Value
			if rhs != nil {
				if ritem, ok := rhs.Get(litem.Name); ok {
					w2.rhs = &ritem.Value
				}
			}
			if newErrs := w2.merge(); len(newErrs) > 0 {
				errs = append(errs, newErrs...)
			} else if w2.out != nil {
				out.Items = append(out.Items, value.Field{litem.Name, *w2.out})
			}
			w.finishDescent(w2)
		}
	}

	if rhs != nil {
		for j := range rhs.Items {
			ritem := &rhs.Items[j]
			if lhs != nil {
				if _, ok := lhs.Get(ritem.Name); ok {
					continue
				}
			}

			fieldType := t.ElementType
			if sf, ok := t.FindField(ritem.Name); ok {
				fieldType = sf.Type
			}
			w2 := w.prepareDescent(fieldpath.PathElement{FieldName: &ritem.Name}, fieldType)
			w2.rhs = &ritem.Value
			if newErrs := w2.merge(); len(newErrs) > 0 {
				errs = append(errs, newErrs...)
			} else if w2.out != nil {
				out.Items = append(out.Items, value.Field{ritem.Name, *w2.out})
			}
			w.finishDescent(w2)
		}
	}

	if len(out.Items) > 0 {
		w.out = &value.Value{MapValue: out}
	}
	return errs
}

func (w *mergingWalker) doMap(t *schema.Map) (errs ValidationErrors) {
	var lhs, rhs *value.Map
	w.derefMap("lhs: ", w.lhs, &lhs)
	w.derefMap("rhs: ", w.rhs, &rhs)

	// If both lhs and rhs are empty/null, treat it as a
	// leaf: this helps preserve the empty/null
	// distinction.
	emptyPromoteToLeaf := (lhs == nil || len(lhs.Items) == 0) &&
		(rhs == nil || len(rhs.Items) == 0)

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
