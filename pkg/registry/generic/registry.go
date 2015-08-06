/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package generic

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// AttrFunc returns label and field sets for List or Watch to compare against, or an error.
type AttrFunc func(obj runtime.Object) (label labels.Set, field fields.Set, err error)

// SelectionPredicate implements a generic predicate that can be passed to
// GenericRegistry's List or Watch methods. Implements the Matcher interface.
type SelectionPredicate struct {
	Label    labels.Selector
	Field    fields.Selector
	GetAttrs AttrFunc
}

// Matches returns true if the given object's labels and fields (as
// returned by s.GetAttrs) match s.Label and s.Field. An error is
// returned if s.GetAttrs fails.
func (s *SelectionPredicate) Matches(obj runtime.Object) (bool, error) {
	if s.Label.Empty() && s.Field.Empty() {
		return true, nil
	}
	labels, fields, err := s.GetAttrs(obj)
	if err != nil {
		return false, err
	}
	return s.Label.Matches(labels) && s.Field.Matches(fields), nil
}

// MatchesSingle will return (name, true) iff s.Field matches on the object's
// name.
func (s *SelectionPredicate) MatchesSingle() (string, bool) {
	// TODO: should be namespace.name
	if name, ok := s.Field.RequiresExactMatch("metadata.name"); ok {
		return name, true
	}
	return "", false
}

// Matcher can return true if an object matches the Matcher's selection
// criteria. If it is known that the matcher will match only a single object
// then MatchesSingle should return the key of that object and true. This is an
// optimization only--Matches() should continue to work.
type Matcher interface {
	// Matches should return true if obj matches this matcher's requirements.
	Matches(obj runtime.Object) (matchesThisObject bool, err error)

	// If this matcher matches a single object, return the key for that
	// object and true here. This will greatly increase efficiency. You
	// must still implement Matches(). Note that key does NOT need to
	// include the object's namespace.
	MatchesSingle() (key string, matchesSingleObject bool)

	// TODO: when we start indexing objects, add something like the below:
	//         MatchesIndices() (indexName []string, indexValue []string)
	//       where indexName/indexValue are the same length.
}

// MatcherFunc makes a matcher from the provided function. For easy definition
// of matchers for testing. Note: use SelectionPredicate above for real code!
func MatcherFunc(f func(obj runtime.Object) (bool, error)) Matcher {
	return matcherFunc(f)
}

type matcherFunc func(obj runtime.Object) (bool, error)

// Matches calls the embedded function.
func (m matcherFunc) Matches(obj runtime.Object) (bool, error) {
	return m(obj)
}

// MatchesSingle always returns "", false-- because this is a predicate
// implementation of Matcher.
func (m matcherFunc) MatchesSingle() (string, bool) {
	return "", false
}

// MatchOnKey returns a matcher that will send only the object matching key
// through the matching function f. For testing!
// Note: use SelectionPredicate above for real code!
func MatchOnKey(key string, f func(obj runtime.Object) (bool, error)) Matcher {
	return matchKey{key, f}
}

type matchKey struct {
	key string
	matcherFunc
}

// MatchesSingle always returns its key, true.
func (m matchKey) MatchesSingle() (string, bool) {
	return m.key, true
}

var (
	// Assert implementations match the interface.
	_ = Matcher(matchKey{})
	_ = Matcher(&SelectionPredicate{})
	_ = Matcher(matcherFunc(nil))
)

// DecoratorFunc can mutate the provided object prior to being returned.
type DecoratorFunc func(obj runtime.Object) error

// Registry knows how to store & list any runtime.Object. Can be used for
// any object types which don't require special features from the storage
// layer.
// DEPRECATED: replace with direct implementation of RESTStorage
type Registry interface {
	ListPredicate(api.Context, Matcher) (runtime.Object, error)
	CreateWithName(ctx api.Context, id string, obj runtime.Object) error
	UpdateWithName(ctx api.Context, id string, obj runtime.Object) error
	Get(ctx api.Context, id string) (runtime.Object, error)
	Delete(ctx api.Context, id string, options *api.DeleteOptions) (runtime.Object, error)
	WatchPredicate(ctx api.Context, m Matcher, resourceVersion string) (watch.Interface, error)
}

// FilterList filters any list object that conforms to the api conventions,
// provided that 'm' works with the concrete type of list. d is an optional
// decorator for the returned functions. Only matching items are decorated.
func FilterList(list runtime.Object, m Matcher, d DecoratorFunc) (filtered runtime.Object, err error) {
	// TODO: push a matcher down into tools.etcdHelper to avoid all this
	// nonsense. This is a lot of unnecessary copies.
	items, err := runtime.ExtractList(list)
	if err != nil {
		return nil, err
	}
	var filteredItems []runtime.Object
	for _, obj := range items {
		match, err := m.Matches(obj)
		if err != nil {
			return nil, err
		}
		if match {
			if d != nil {
				if err := d(obj); err != nil {
					return nil, err
				}
			}
			filteredItems = append(filteredItems, obj)
		}
	}
	err = runtime.SetList(list, filteredItems)
	if err != nil {
		return nil, err
	}
	return list, nil
}
