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

package runtime

import (
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
)

// SelectionPredicate is used to select objects with field and label selectors.
type SelectionPredicate struct {
	Selectors Selectors
	GetAttrs  AttrFunc
}

// Matches returns true if the given object's labels and fields (as
// returned by s.GetAttrs) match s.Label and s.Field. An error is
// returned if s.GetAttrs fails.
func (s SelectionPredicate) Matches(obj Object) (bool, error) {
	if s.Selectors.Empty() {
		return true, nil
	}
	l, f, err := s.GetAttrs(obj)
	if err != nil {
		return false, err
	}
	return s.Selectors.Matches(l, f), nil
}

// MatchesObjectAttributes returns true if the given labels and fields
// match s.Label and s.Field.
func (s SelectionPredicate) MatchesObjectAttributes(l labels.Set, f fields.Set) bool {
	if s.Selectors.Empty() {
		return true
	}
	return s.Selectors.Matches(l, f)
}

// MatchesSingleNamespace will return (namespace, true) if and only if s.Field matches on the object's
// namespace.
func (s SelectionPredicate) MatchesSingleNamespace() (string, bool) {
	if s.Selectors.Fields != nil {
		if namespace, ok := s.Selectors.Fields.RequiresExactMatch("metadata.namespace"); ok {
			return namespace, true
		}
	}
	return "", false
}

// MatchesSingle will return (name, true) if and only if s.Field matches on the object's
// name.
func (s SelectionPredicate) MatchesSingle() (string, bool) {
	if s.Selectors.Fields != nil {
		if name, ok := s.Selectors.Fields.RequiresExactMatch("metadata.name"); ok {
			return name, true
		}
	}
	return "", false
}

// Empty returns true if the predicate performs no filtering.
func (s SelectionPredicate) Empty() bool {
	return s.Selectors.Empty()
}
