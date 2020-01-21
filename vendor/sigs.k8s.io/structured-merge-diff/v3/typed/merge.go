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
	"math"

	"sigs.k8s.io/structured-merge-diff/v3/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v3/schema"
	"sigs.k8s.io/structured-merge-diff/v3/value"
)

type mergingWalker struct {
	lhs     value.Value
	rhs     value.Value
	schema  *schema.Schema
	typeRef schema.TypeRef

	// Current path that we are merging
	path fieldpath.Path

	// How to merge. Called after schema validation for all leaf fields.
	rule mergeRule

	// If set, called after non-leaf items have been merged. (`out` is
	// probably already set.)
	postItemHook mergeRule

	// output of the merge operation (nil if none)
	out *interface{}

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
			v := w.rhs.Unstructured()
			w.out = &v
		} else if w.lhs != nil {
			v := w.lhs.Unstructured()
			w.out = &v
		}
	})
)

// merge sets w.out.
func (w *mergingWalker) merge(prefixFn func() string) (errs ValidationErrors) {
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
	if alhs.Equals(&arhs) {
		errs = append(errs, handleAtom(arhs, w.typeRef, w)...)
	} else {
		w2 := *w
		errs = append(errs, handleAtom(alhs, w.typeRef, &w2)...)
		errs = append(errs, handleAtom(arhs, w.typeRef, w)...)
	}

	if !w.inLeaf && w.postItemHook != nil {
		w.postItemHook(w)
	}
	return errs.WithLazyPrefix(prefixFn)
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
	errs = append(errs, validateScalar(t, w.lhs, "lhs: ")...)
	errs = append(errs, validateScalar(t, w.rhs, "rhs: ")...)
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
	w2.path = append(w2.path, pe)
	w2.lhs = nil
	w2.rhs = nil
	w2.out = nil
	return w2
}

func (w *mergingWalker) finishDescent(w2 *mergingWalker) {
	// if the descent caused a realloc, ensure that we reuse the buffer
	// for the next sibling.
	w.path = w2.path[:len(w2.path)-1]
	*w.spareWalkers = append(*w.spareWalkers, w2)
}

func (w *mergingWalker) derefMap(prefix string, v value.Value, dest *value.Map) (errs ValidationErrors) {
	// taking dest as input so that it can be called as a one-liner with
	// append.
	if v == nil {
		return nil
	}
	m, err := mapValue(v)
	if err != nil {
		return errorf("%v: %v", prefix, err)
	}
	*dest = m
	return nil
}

func (w *mergingWalker) visitListItems(t *schema.List, lhs, rhs value.List) (errs ValidationErrors) {
	rLen := 0
	if rhs != nil {
		rLen = rhs.Length()
	}
	lLen := 0
	if lhs != nil {
		lLen = lhs.Length()
	}
	out := make([]interface{}, 0, int(math.Max(float64(rLen), float64(lLen))))

	// TODO: ordering is totally wrong.
	// TODO: might as well make the map order work the same way.

	// This is a cheap hack to at least make the output order stable.
	rhsOrder := make([]fieldpath.PathElement, 0, rLen)

	// First, collect all RHS children.
	observedRHS := fieldpath.MakePathElementValueMap(rLen)
	if rhs != nil {
		for i := 0; i < rhs.Length(); i++ {
			child := rhs.At(i)
			pe, err := listItemToPathElement(t, i, child)
			if err != nil {
				errs = append(errs, errorf("rhs: element %v: %v", i, err.Error())...)
				// If we can't construct the path element, we can't
				// even report errors deeper in the schema, so bail on
				// this element.
				continue
			}
			if _, ok := observedRHS.Get(pe); ok {
				errs = append(errs, errorf("rhs: duplicate entries for key %v", pe.String())...)
			}
			observedRHS.Insert(pe, child)
			rhsOrder = append(rhsOrder, pe)
		}
	}

	// Then merge with LHS children.
	observedLHS := fieldpath.MakePathElementSet(lLen)
	if lhs != nil {
		for i := 0; i < lhs.Length(); i++ {
			child := lhs.At(i)
			pe, err := listItemToPathElement(t, i, child)
			if err != nil {
				errs = append(errs, errorf("lhs: element %v: %v", i, err.Error())...)
				// If we can't construct the path element, we can't
				// even report errors deeper in the schema, so bail on
				// this element.
				continue
			}
			if observedLHS.Has(pe) {
				errs = append(errs, errorf("lhs: duplicate entries for key %v", pe.String())...)
				continue
			}
			observedLHS.Insert(pe)
			w2 := w.prepareDescent(pe, t.ElementType)
			w2.lhs = value.Value(child)
			if rchild, ok := observedRHS.Get(pe); ok {
				w2.rhs = rchild
			}
			errs = append(errs, w2.merge(pe.String)...)
			if w2.out != nil {
				out = append(out, *w2.out)
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
		w2.rhs = value
		errs = append(errs, w2.merge(pe.String)...)
		if w2.out != nil {
			out = append(out, *w2.out)
		}
		w.finishDescent(w2)
	}

	if len(out) > 0 {
		i := interface{}(out)
		w.out = &i
	}

	return errs
}

func (w *mergingWalker) derefList(prefix string, v value.Value, dest *value.List) (errs ValidationErrors) {
	// taking dest as input so that it can be called as a one-liner with
	// append.
	if v == nil {
		return nil
	}
	l, err := listValue(v)
	if err != nil {
		return errorf("%v: %v", prefix, err)
	}
	*dest = l
	return nil
}

func (w *mergingWalker) doList(t *schema.List) (errs ValidationErrors) {
	var lhs, rhs value.List
	w.derefList("lhs: ", w.lhs, &lhs)
	w.derefList("rhs: ", w.rhs, &rhs)

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

func (w *mergingWalker) visitMapItem(t *schema.Map, out map[string]interface{}, key string, lhs, rhs value.Value) (errs ValidationErrors) {
	fieldType := t.ElementType
	if sf, ok := t.FindField(key); ok {
		fieldType = sf.Type
	}
	pe := fieldpath.PathElement{FieldName: &key}
	w2 := w.prepareDescent(pe, fieldType)
	w2.lhs = lhs
	w2.rhs = rhs
	errs = append(errs, w2.merge(pe.String)...)
	if w2.out != nil {
		out[key] = *w2.out
	}
	w.finishDescent(w2)
	return errs
}

func (w *mergingWalker) visitMapItems(t *schema.Map, lhs, rhs value.Map) (errs ValidationErrors) {
	out := map[string]interface{}{}

	if lhs != nil {
		lhs.Iterate(func(key string, val value.Value) bool {
			var rval value.Value
			if rhs != nil {
				if item, ok := rhs.Get(key); ok {
					rval = item
					defer rval.Recycle()
				}
			}
			errs = append(errs, w.visitMapItem(t, out, key, val, rval)...)
			return true
		})
	}

	if rhs != nil {
		rhs.Iterate(func(key string, val value.Value) bool {
			if lhs != nil {
				if lhs.Has(key) {
					return true
				}
			}
			errs = append(errs, w.visitMapItem(t, out, key, nil, val)...)
			return true
		})
	}
	if len(out) > 0 {
		i := interface{}(out)
		w.out = &i
	}

	return errs
}

func (w *mergingWalker) doMap(t *schema.Map) (errs ValidationErrors) {
	var lhs, rhs value.Map
	w.derefMap("lhs: ", w.lhs, &lhs)
	w.derefMap("rhs: ", w.rhs, &rhs)

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

	errs = append(errs, w.visitMapItems(t, lhs, rhs)...)

	return errs
}
