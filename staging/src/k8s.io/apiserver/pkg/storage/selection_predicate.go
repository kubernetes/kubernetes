/*
Copyright 2016 The Kubernetes Authors.

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

package storage

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
)

// AttrFunc returns label and field sets and the uninitialized flag for List or Watch to match.
// In any failure to parse given object, it returns error.
type AttrFunc func(obj runtime.Object) (labels.Set, fields.Set, bool, error)

// FieldMutationFunc allows the mutation of the field selection fields.  It is mutating to
// avoid the extra allocation on this common path
type FieldMutationFunc func(obj runtime.Object, fieldSet fields.Set) error

func DefaultClusterScopedAttr(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	metadata, err := meta.Accessor(obj)
	if err != nil {
		return nil, nil, false, err
	}
	fieldSet := fields.Set{
		"metadata.name": metadata.GetName(),
	}

	return labels.Set(metadata.GetLabels()), fieldSet, metadata.GetInitializers() != nil, nil
}

func DefaultNamespaceScopedAttr(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	metadata, err := meta.Accessor(obj)
	if err != nil {
		return nil, nil, false, err
	}
	fieldSet := fields.Set{
		"metadata.name":      metadata.GetName(),
		"metadata.namespace": metadata.GetNamespace(),
	}

	return labels.Set(metadata.GetLabels()), fieldSet, metadata.GetInitializers() != nil, nil
}

func (f AttrFunc) WithFieldMutation(fieldMutator FieldMutationFunc) AttrFunc {
	return func(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
		labelSet, fieldSet, initialized, err := f(obj)
		if err != nil {
			return nil, nil, false, err
		}
		if err := fieldMutator(obj, fieldSet); err != nil {
			return nil, nil, false, err
		}
		return labelSet, fieldSet, initialized, nil
	}
}

// SelectionPredicate is used to represent the way to select objects from api storage.
type SelectionPredicate struct {
	Label                labels.Selector
	Field                fields.Selector
	IncludeUninitialized bool
	GetAttrs             AttrFunc
	IndexFields          []string
}

// Matches returns true if the given object's labels and fields (as
// returned by s.GetAttrs) match s.Label and s.Field. An error is
// returned if s.GetAttrs fails.
func (s *SelectionPredicate) Matches(obj runtime.Object) (bool, error) {
	if s.Empty() {
		return true, nil
	}
	labels, fields, uninitialized, err := s.GetAttrs(obj)
	if err != nil {
		return false, err
	}
	if !s.IncludeUninitialized && uninitialized {
		return false, nil
	}
	matched := s.Label.Matches(labels)
	if matched && s.Field != nil {
		matched = (matched && s.Field.Matches(fields))
	}
	return matched, nil
}

// MatchesObjectAttributes returns true if the given labels and fields
// match s.Label and s.Field.
func (s *SelectionPredicate) MatchesObjectAttributes(l labels.Set, f fields.Set, uninitialized bool) bool {
	if !s.IncludeUninitialized && uninitialized {
		return false
	}
	if s.Label.Empty() && s.Field.Empty() {
		return true
	}
	matched := s.Label.Matches(l)
	if matched && s.Field != nil {
		matched = (matched && s.Field.Matches(f))
	}
	return matched
}

// MatchesSingle will return (name, true) if and only if s.Field matches on the object's
// name.
func (s *SelectionPredicate) MatchesSingle() (string, bool) {
	// TODO: should be namespace.name
	if name, ok := s.Field.RequiresExactMatch("metadata.name"); ok {
		return name, true
	}
	return "", false
}

// For any index defined by IndexFields, if a matcher can match only (a subset)
// of objects that return <value> for a given index, a pair (<index name>, <value>)
// wil be returned.
// TODO: Consider supporting also labels.
func (s *SelectionPredicate) MatcherIndex() []MatchValue {
	var result []MatchValue
	for _, field := range s.IndexFields {
		if value, ok := s.Field.RequiresExactMatch(field); ok {
			result = append(result, MatchValue{IndexName: field, Value: value})
		}
	}
	return result
}

// Empty returns true if the predicate performs no filtering.
func (s *SelectionPredicate) Empty() bool {
	return s.Label.Empty() && s.Field.Empty() && s.IncludeUninitialized
}
