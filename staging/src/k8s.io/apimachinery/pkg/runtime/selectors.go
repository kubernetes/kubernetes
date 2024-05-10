/*
Copyright 2024 The Kubernetes Authors.

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

package runtime

import (
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
)

// SelectableObject is a runtime.Object with at least the minimum metadata
// required for selection by label and/or field.
type SelectableObject interface {
	SelectableObjectMeta
	Object
}

// SelectableObjectMeta exposes at least the minimum metadata required for
// selection by label and/or field.
type SelectableObjectMeta interface {
	LabelSelectable
	FieldSelectable
}

// LabelSelectable exposes the minimum metadata required for
// selection by label.
type LabelSelectable interface {
	GetLabels() map[string]string
}

// FieldSelectable exposes the minimum metadata required for
// selection by field.
type FieldSelectable interface {
	GetName() string
	GetNamespace() string
}

// DefaultSelectableFields extracts the default selectable fields and their
// values from the specified object.
func DefaultSelectableFields(selectable FieldSelectable) fields.Set {
	return fields.Set{
		"metadata.name": selectable.GetName(),
		// Not all resources are namespaced, but supporting metadata.namespace
		// field selection on non-namespaced types greatly simplifies the code.
		// This allows the Scheme to remain ignorant of whether a Kind is
		// namespaced or not. The value will just always be the empty string for
		// non-namespaced (aka cluster-scoped) resources.
		"metadata.namespace": selectable.GetNamespace(),
	}
}

// SelectorFunc returns true if the object is matched by the specified selectors.
type SelectorFunc func(Object, Selectors) (bool, error)

// DefaultSelectorFunc returns matches objects on the default selectable fields.
func DefaultSelectorFunc(obj Object, selectors Selectors) (bool, error) {
	return DefaultMatcherFunc(selectors).Matches(obj)
}

type MatcherFunc func(Selectors) SelectionPredicate

// DefaultMatcherFunc returns a default SelectionPredicate that matches on
// the default selectable fields.
func DefaultMatcherFunc(selectors Selectors) SelectionPredicate {
	return SelectionPredicate{
		Selectors: selectors,
		GetAttrs:  DefaultAttrFunc,
	}
}

// AttrFunc returns label and field sets and the uninitialized flag for List or Watch to match.
// In any failure to parse given object, it returns error.
type AttrFunc func(obj Object) (labels.Set, fields.Set, error)

// DefaultAttrFunc returns labels and fields of a given object for filtering purposes.
func DefaultAttrFunc(obj Object) (labels.Set, fields.Set, error) {
	var l labels.Set
	var f fields.Set
	// TODO: Do we need to account for metav1.ObjectMetaAccessor to unwrap object objects?
	// If so, this can't be in runtime...
	if s, ok := obj.(LabelSelectable); ok {
		l = s.GetLabels()
	}
	if s, ok := obj.(FieldSelectable); ok {
		f = DefaultSelectableFields(s)
	}
	return l, f, nil
}

// MergeFieldsSets merges fields from a fragment into the source.
func MergeFieldsSets(source fields.Set, fragment fields.Set) fields.Set {
	for k, value := range fragment {
		source[k] = value
	}
	return source
}

// Selectors is a pair of label and field selectors.
// Methods on this struct assume nil selectors match everything.
// However, if you call methods on Labels or Fields directly, be sure to do a
// nil check first, as many of their implementations do not have methods with
// pointer receivers.
type Selectors struct {
	Labels labels.Selector
	Fields fields.Selector
}

func (s Selectors) Matches(l labels.Set, f fields.Set) bool {
	if s.Empty() {
		return true
	}
	if s.Labels != nil && !s.Labels.Matches(l) {
		return false
	}
	if s.Fields != nil && !s.Fields.Matches(f) {
		return false
	}
	return true
}

func (s Selectors) Empty() bool {
	return (s.Labels == nil || s.Labels.Empty()) &&
		(s.Fields == nil || s.Fields.Empty())
}

func (s Selectors) DeepCopySelectors() Selectors {
	copy := Selectors{}
	if s.Labels != nil {
		copy.Labels = s.Labels.DeepCopySelector()
	}
	if s.Fields != nil {
		copy.Fields = s.Fields.DeepCopySelector()
	}
	return copy
}
