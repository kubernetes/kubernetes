/*
Copyright 2014 Google Inc. All rights reserved.

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
	"errors"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

type Ignored struct {
	ID string
}

type IgnoredList struct {
	Items []Ignored
}

func (*Ignored) IsAnAPIObject()     {}
func (*IgnoredList) IsAnAPIObject() {}

func TestSelectionPredicate(t *testing.T) {
	table := map[string]struct {
		labelSelector, fieldSelector string
		labels, fields               labels.Set
		err                          error
		shouldMatch                  bool
	}{
		"A": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			labels:        labels.Set{"name": "foo"},
			fields:        labels.Set{"uid": "12345"},
			shouldMatch:   true,
		},
		"B": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			labels:        labels.Set{"name": "foo"},
			fields:        labels.Set{},
			shouldMatch:   false,
		},
		"C": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			labels:        labels.Set{},
			fields:        labels.Set{"uid": "12345"},
			shouldMatch:   false,
		},
		"error": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			err:           errors.New("maybe this is a 'wrong object type' error"),
			shouldMatch:   false,
		},
	}

	for name, item := range table {
		parsedLabel, err := labels.Parse(item.labelSelector)
		if err != nil {
			panic(err)
		}
		parsedField, err := labels.ParseSelector(item.fieldSelector)
		if err != nil {
			panic(err)
		}
		sp := &SelectionPredicate{
			Label: parsedLabel,
			Field: parsedField,
			GetAttrs: func(runtime.Object) (label, field labels.Set, err error) {
				return item.labels, item.fields, item.err
			},
		}
		got, err := sp.Matches(&Ignored{})
		if e, a := item.err, err; e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
			continue
		}
		if e, a := item.shouldMatch, got; e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
		}
	}
}

func TestFilterList(t *testing.T) {
	try := &IgnoredList{
		Items: []Ignored{
			{"foo"},
			{"bar"},
			{"baz"},
			{"qux"},
			{"zot"},
		},
	}
	expect := &IgnoredList{
		Items: []Ignored{
			{"bar"},
			{"baz"},
		},
	}

	got, err := FilterList(try,
		MatcherFunc(func(obj runtime.Object) (bool, error) {
			i, ok := obj.(*Ignored)
			if !ok {
				return false, errors.New("wrong type")
			}
			return i.ID[0] == 'b', nil
		}),
		nil,
	)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}

	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}
