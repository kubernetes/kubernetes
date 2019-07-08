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
	//"reflect"
	"sort"

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

	// TODO: The postItemHook doesn't really need to be "post"
	// rather than "pre". But since we need to it be done before to
	// insert in the set in the right order.
	if !w.inLeaf && w.postItemHook != nil {
		w.postItemHook(w)
	}

	//alhs := deduceAtom(a, w.lhs)
	arhs := deduceAtom(a, w.rhs)
	// if reflect.DeepEqual(alhs, arhs) {
	errs = append(errs, handleAtom(arhs, w.typeRef, w)...)
	// } else {
	// 	w2 := *w
	// 	errs = append(errs, handleAtom(alhs, w.typeRef, &w2)...)
	// 	errs = append(errs, handleAtom(arhs, w.typeRef, w)...)
	// }

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

func (w *mergingWalker) doScalar(t schema.Scalar) (errs ValidationErrors) {
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
	w2 := *w
	w2.typeRef = tr
	w2.errorFormatter.descend(pe)
	w2.lhs = nil
	w2.rhs = nil
	w2.out = nil
	return &w2
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

func (w *mergingWalker) visitListItems(t schema.List, lhs, rhs *value.List) (errs ValidationErrors) {
	out := &value.List{}

	// TODO: ordering is totally wrong.
	// TODO: might as well make the map order work the same way.

	// Collect and index all items
	rhsItems := map[string]value.Value{}
	lhsItems := map[string]value.Value{}
	order := []fieldpath.PathElement{}
	if lhs != nil {
		for i, child := range lhs.Items {
			pe, err := listItemToPathElement(t, i, child)
			if err != nil {
				errs = append(errs, w.errorf("lhs: element %v: %v", i, err.Error())...)
				// If we can't construct the path element, we can't
				// even report errors deeper in the schema, so bail on
				// this element.
				continue
			}
			keyStr := pe.String()
			if _, found := lhsItems[keyStr]; found {
				errs = append(errs, w.errorf("lhs: duplicate entries for key %v", keyStr)...)
				continue
			}
			lhsItems[keyStr] = child
			order = append(order, pe)
		}
	}
	if rhs != nil {
		for i, child := range rhs.Items {
			pe, err := listItemToPathElement(t, i, child)
			if err != nil {
				errs = append(errs, w.errorf("rhs: element %v: %v", i, err.Error())...)
				// If we can't construct the path element, we can't
				// even report errors deeper in the schema, so bail on
				// this element.
				continue
			}
			keyStr := pe.String()
			if _, found := rhsItems[keyStr]; found {
				errs = append(errs, w.errorf("rhs: duplicate entries for key %v", keyStr)...)
				continue
			}
			rhsItems[keyStr] = child
			// If the key was already in lhsItems, don't add it again.
			if _, ok := lhsItems[keyStr]; !ok {
				order = append(order, pe)
			}
		}
	}

	sorted := []fieldpath.PathElement{}
	for _, pe := range order {
		sorted = append(sorted, pe)
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Compare(sorted[j]) < 0
	})

	outItems := map[string]value.Value{}

	// Now we want to merge the items, in sorted-order.
	for _, pe := range sorted {
		keyStr := pe.String()

		w2 := w.prepareDescent(pe, t.ElementType)
		if lchild, ok := lhsItems[keyStr]; ok {
			w2.lhs = &lchild
		}
		if rchild, ok := rhsItems[keyStr]; ok {
			w2.rhs = &rchild
		}
		if newErrs := w2.merge(); len(newErrs) > 0 {
			errs = append(errs, newErrs...)
		} else if w2.out != nil {
			outItems[keyStr] = *w2.out
		}
	}

	// Insert items in chosen order
	for _, pe := range order {
		out.Items = append(out.Items, outItems[pe.String()])
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

func (w *mergingWalker) doList(t schema.List) (errs ValidationErrors) {
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

func (w *mergingWalker) visitMapItems(t schema.Map, lhs, rhs *value.Map) (errs ValidationErrors) {
	out := &value.Map{}

	fieldTypes := map[string]schema.TypeRef{}
	for i := range t.Fields {
		// I don't want to use the loop variable since a reference
		// might outlive the loop iteration (in an error message).
		f := t.Fields[i]
		fieldTypes[f.Name] = f.Type
	}

	items := map[string]struct{}{}
	if lhs != nil {
		for _, item := range lhs.Items {
			items[item.Name] = struct{}{}
		}
	}
	if rhs != nil {
		for _, item := range rhs.Items {
			items[item.Name] = struct{}{}
		}
	}
	keys := []string{}
	for key := range items {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		name := key
		fieldType := t.ElementType
		if ft, ok := fieldTypes[name]; ok {
			fieldType = ft
		}
		w2 := w.prepareDescent(fieldpath.PathElement{FieldName: &name}, fieldType)
		if lhs != nil {
			if litem, ok := lhs.Get(key); ok {
				w2.lhs = &litem.Value
			}
		}
		if rhs != nil {
			if ritem, ok := rhs.Get(key); ok {
				w2.rhs = &ritem.Value
			}
		}
		if newErrs := w2.merge(); len(newErrs) > 0 {
			errs = append(errs, newErrs...)
		} else if w2.out != nil {
			out.Set(name, *w2.out)
		}
	}

	if len(out.Items) > 0 {
		w.out = &value.Value{MapValue: out}
	}
	return errs
}

func (w *mergingWalker) doMap(t schema.Map) (errs ValidationErrors) {
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
