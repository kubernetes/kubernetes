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
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// PredicateFunc returns a SelectionPredicate that matches on labels fields.
// This abstraction allows resource-specific label and field selection, but
// in most cases, all labels are selectable and only a few fields are selectable.
type PredicateFunc func(runtime.Selectors) SelectionPredicate

// DefaultPredicateFunc returns a default SelectionPredicate that matches on
// the default selectable fields.
func DefaultPredicateFunc(selectors runtime.Selectors) SelectionPredicate {
	return SelectionPredicate{
		SelectionPredicate: runtime.DefaultMatcherFunc(selectors),
	}
}

// PredicateFuncFromMatcherFunc wraps a MatcherFunc as a PredicateFunc.
// This is needed because storage.SelectionPredicate composes runtime.SelectionPredicate.
func PredicateFuncFromMatcherFunc(matcherFunc runtime.MatcherFunc) PredicateFunc {
	return func(selectors runtime.Selectors) SelectionPredicate {
		return SelectionPredicate{
			SelectionPredicate: matcherFunc(selectors),
		}
	}
}

// SelectionPredicate is used to represent the way to select objects from api storage.
type SelectionPredicate struct {
	runtime.SelectionPredicate
	IndexLabels         []string
	IndexFields         []string
	Limit               int64
	Continue            string
	AllowWatchBookmarks bool
}

// MatchesSingleNamespace will return (namespace, true) if and only if s.Field matches on the object's
// namespace.
func (s *SelectionPredicate) MatchesSingleNamespace() (string, bool) {
	if len(s.Continue) > 0 {
		return "", false
	}
	return s.SelectionPredicate.MatchesSingleNamespace()
}

// MatchesSingle will return (name, true) if and only if s.Field matches on the object's
// name.
func (s *SelectionPredicate) MatchesSingle() (string, bool) {
	if len(s.Continue) > 0 {
		return "", false
	}
	return s.SelectionPredicate.MatchesSingle()
}

// Empty returns true if the predicate performs no filtering.
func (s *SelectionPredicate) Empty() bool {
	return s.SelectionPredicate.Empty()
}

// For any index defined by IndexFields, if a matcher can match only (a subset)
// of objects that return <value> for a given index, a pair (<index name>, <value>)
// wil be returned.
func (s *SelectionPredicate) MatcherIndex(ctx context.Context) []MatchValue {
	var result []MatchValue
	for _, field := range s.IndexFields {
		if s.Selectors.Fields != nil {
			if value, ok := s.Selectors.Fields.RequiresExactMatch(field); ok {
				// select objects with a specific field value. i.e metadata.namespace=kube-system
				result = append(result, MatchValue{IndexName: FieldIndex(field), Value: value})
				continue
			}
		}
		if field == "metadata.namespace" {
			// list objects in a namespace. i.e. /api/v1/namespaces/kube-system/pods
			if namespace, isNamespaceScope := isNamespaceScopedRequest(ctx); isNamespaceScope {
				result = append(result, MatchValue{IndexName: FieldIndex(field), Value: namespace})
			}
		}
	}
	for _, label := range s.IndexLabels {
		if s.Selectors.Labels != nil {
			if value, ok := s.Selectors.Labels.RequiresExactMatch(label); ok {
				result = append(result, MatchValue{IndexName: LabelIndex(label), Value: value})
			}
		}
	}
	return result
}

func isNamespaceScopedRequest(ctx context.Context) (string, bool) {
	re, _ := request.RequestInfoFrom(ctx)
	if re == nil || len(re.Namespace) == 0 {
		return "", false
	}
	return re.Namespace, true
}

// LabelIndex add prefix for label index.
func LabelIndex(label string) string {
	return "l:" + label
}

// FiledIndex add prefix for field index.
func FieldIndex(field string) string {
	return "f:" + field
}
