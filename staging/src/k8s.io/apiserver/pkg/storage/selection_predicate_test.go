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
	"errors"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type Ignored struct {
	ID string
}

func (obj *Ignored) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (obj *Ignored) DeepCopyObject() runtime.Object {
	panic("Ignored does not support DeepCopy")
}

func TestSelectionPredicate(t *testing.T) {
	table := map[string]struct {
		labelSelector, fieldSelector string
		labels                       labels.Set
		fields                       fields.Set
		err                          error
		shouldMatch                  bool
		matchSingleKey               string
	}{
		"A": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			labels:        labels.Set{"name": "foo"},
			fields:        fields.Set{"uid": "12345"},
			shouldMatch:   true,
		},
		"B": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			labels:        labels.Set{"name": "foo"},
			fields:        fields.Set{},
			shouldMatch:   false,
		},
		"C": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			labels:        labels.Set{},
			fields:        fields.Set{"uid": "12345"},
			shouldMatch:   false,
		},
		"D": {
			fieldSelector:  "metadata.name=12345",
			labels:         labels.Set{},
			fields:         fields.Set{"metadata.name": "12345"},
			shouldMatch:    true,
			matchSingleKey: "12345",
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
		parsedField, err := fields.ParseSelector(item.fieldSelector)
		if err != nil {
			panic(err)
		}
		sp := &SelectionPredicate{
			Label: parsedLabel,
			Field: parsedField,
			GetAttrs: func(runtime.Object) (label labels.Set, field fields.Set, err error) {
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
		got = sp.MatchesObjectAttributes(item.labels, item.fields)
		if e, a := item.shouldMatch, got; e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
		}
		if key := item.matchSingleKey; key != "" {
			got, ok := sp.MatchesSingle()
			if !ok {
				t.Errorf("%v: expected single match", name)
			}
			if e, a := key, got; e != a {
				t.Errorf("%v: expected %v, got %v", name, e, a)
			}
		}
	}
}

func TestSelectionPredicateMatcherIndex(t *testing.T) {
	testCases := map[string]struct {
		labelSelector, fieldSelector string
		indexLabels                  []string
		indexFields                  []string
		expected                     []MatchValue
	}{
		"Match nil": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{"bar"},
			indexFields:   []string{},
			expected:      nil,
		},
		"Match field": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{},
			indexFields:   []string{"uid"},
			expected:      []MatchValue{{IndexName: FieldIndex("uid"), Value: "12345"}},
		},
		"Match label": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{},
			expected:      []MatchValue{{IndexName: LabelIndex("name"), Value: "foo"}},
		},
		"Match field and label": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{"uid"},
			expected:      []MatchValue{{IndexName: FieldIndex("uid"), Value: "12345"}, {IndexName: LabelIndex("name"), Value: "foo"}},
		},
		"Negative match field and label": {
			labelSelector: "name!=foo",
			fieldSelector: "uid!=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{"uid"},
			expected:      nil,
		},
		"Negative match field and match label": {
			labelSelector: "name=foo",
			fieldSelector: "uid!=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{"uid"},
			expected:      []MatchValue{{IndexName: LabelIndex("name"), Value: "foo"}},
		},
		"Negative match label and match field": {
			labelSelector: "name!=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{"uid"},
			expected:      []MatchValue{{IndexName: FieldIndex("uid"), Value: "12345"}},
		},
	}
	for name, testCase := range testCases {
		parsedLabel, err := labels.Parse(testCase.labelSelector)
		if err != nil {
			panic(err)
		}
		parsedField, err := fields.ParseSelector(testCase.fieldSelector)
		if err != nil {
			panic(err)
		}

		sp := &SelectionPredicate{
			Label:       parsedLabel,
			Field:       parsedField,
			IndexLabels: testCase.indexLabels,
			IndexFields: testCase.indexFields,
		}
		actual := sp.MatcherIndex()
		if !reflect.DeepEqual(testCase.expected, actual) {
			t.Errorf("%v: expected %v, got %v", name, testCase.expected, actual)
		}
	}
}
