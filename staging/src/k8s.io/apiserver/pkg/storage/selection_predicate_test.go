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
	"errors"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/sharding"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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

// shardSelectorMatchingEverything matches all objects but, unlike
// sharding.Everything(), is not Empty().
func shardSelectorMatchingEverything() sharding.Selector {
	return sharding.NewSelector(sharding.ShardRangeRequirement{
		Key:   "object.metadata.uid",
		Start: "0x0000000000000000",
		End:   "0x10000000000000000",
	})
}

func shardSelectorExcludingUID(uid string) sharding.Selector {
	return sharding.NewSelector(sharding.ShardRangeRequirement{
		Key:   "object.metadata.uid",
		Start: "0x0000000000000000",
		End:   "0x" + sharding.HashField(uid),
	})
}

func TestSelectionPredicateEmpty(t *testing.T) {
	mustParseLabel := func(s string) labels.Selector {
		parsed, err := labels.Parse(s)
		if err != nil {
			t.Fatal(err)
		}
		return parsed
	}
	testCases := map[string]struct {
		enableShardedListAndWatch bool
		label                     labels.Selector
		field                     fields.Selector
		shardSelector             sharding.Selector
		expectEmpty               bool
	}{
		"no selectors, gate off": {
			enableShardedListAndWatch: false,
			label:                     labels.Everything(),
			field:                     fields.Everything(),
			expectEmpty:               true,
		},
		"nil selectors, gate off": {
			enableShardedListAndWatch: false,
			expectEmpty:               true,
		},
		"nil selectors, gate on": {
			enableShardedListAndWatch: true,
			expectEmpty:               true,
		},
		"no selectors, gate on": {
			enableShardedListAndWatch: true,
			label:                     labels.Everything(),
			field:                     fields.Everything(),
			expectEmpty:               true,
		},
		"empty shard selector, gate on": {
			enableShardedListAndWatch: true,
			label:                     labels.Everything(),
			field:                     fields.Everything(),
			shardSelector:             sharding.Everything(),
			expectEmpty:               true,
		},
		"non-empty shard selector, gate on": {
			enableShardedListAndWatch: true,
			label:                     labels.Everything(),
			field:                     fields.Everything(),
			shardSelector:             shardSelectorMatchingEverything(),
			expectEmpty:               false,
		},
		"non-empty shard selector, gate off": {
			enableShardedListAndWatch: false,
			label:                     labels.Everything(),
			field:                     fields.Everything(),
			shardSelector:             shardSelectorMatchingEverything(),
			expectEmpty:               true,
		},
		"label selector, gate on": {
			enableShardedListAndWatch: true,
			label:                     mustParseLabel("name=foo"),
			field:                     fields.Everything(),
			expectEmpty:               false,
		},
		"field selector, gate off": {
			enableShardedListAndWatch: false,
			label:                     labels.Everything(),
			field:                     fields.OneTermEqualSelector("metadata.name", "foo"),
			expectEmpty:               false,
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListAndWatch, tc.enableShardedListAndWatch)
			sp := &SelectionPredicate{
				Label:         tc.label,
				Field:         tc.field,
				ShardSelector: tc.shardSelector,
			}
			if got := sp.Empty(); got != tc.expectEmpty {
				t.Errorf("Empty() = %v, want %v", got, tc.expectEmpty)
			}
		})
	}
}

func TestSelectionPredicateMatchesShardOnly(t *testing.T) {
	uid := "shard-only-uid"
	obj := &metav1.PartialObjectMetadata{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID(uid)},
	}
	testCases := map[string]struct {
		enableShardedListAndWatch bool
		shardSelector             sharding.Selector
		expectMatch               bool
	}{
		"in shard, gate on": {
			enableShardedListAndWatch: true,
			shardSelector:             shardSelectorMatchingEverything(),
			expectMatch:               true,
		},
		"out of shard, gate on": {
			enableShardedListAndWatch: true,
			shardSelector:             shardSelectorExcludingUID(uid),
			expectMatch:               false,
		},
		"out of shard, gate off": {
			enableShardedListAndWatch: false,
			shardSelector:             shardSelectorExcludingUID(uid),
			expectMatch:               true,
		},
	}
	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ShardedListAndWatch, tc.enableShardedListAndWatch)
			sp := &SelectionPredicate{
				Label:         labels.Everything(),
				Field:         fields.Everything(),
				ShardSelector: tc.shardSelector,
				GetAttrs: func(runtime.Object) (labels.Set, fields.Set, error) {
					t.Error("GetAttrs must not be called for empty label/field selectors")
					return nil, nil, errors.New("GetAttrs must not be called")
				},
			}
			match, err := sp.Matches(obj)
			if err != nil {
				t.Fatalf("Matches() returned unexpected error: %v", err)
			}
			if match != tc.expectMatch {
				t.Errorf("Matches() = %v, want %v", match, tc.expectMatch)
			}
		})
	}
}

func TestSelectionPredicateMatcherIndex(t *testing.T) {
	testCases := map[string]struct {
		labelSelector, fieldSelector string
		indexLabels                  []string
		indexFields                  []string
		expected                     []MatchValue
		ctx                          context.Context
	}{
		"Match nil": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{"bar"},
			indexFields:   []string{},
			ctx:           context.Background(),
			expected:      nil,
		},
		"Match field": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{},
			indexFields:   []string{"uid"},
			ctx:           context.Background(),
			expected:      []MatchValue{{IndexName: FieldIndex("uid"), Value: "12345"}},
		},
		"Match field for listing namespace pods without metadata.namespace field selector": {
			labelSelector: "",
			fieldSelector: "",
			indexLabels:   []string{},
			indexFields:   []string{"metadata.namespace"},
			ctx: request.WithRequestInfo(context.Background(), &request.RequestInfo{
				IsResourceRequest: true,
				Path:              "/api/v1/namespaces/default/pods",
				Verb:              "list",
				APIPrefix:         "api",
				APIGroup:          "",
				APIVersion:        "v1",
				Namespace:         "default",
				Resource:          "pods",
			}),
			expected: []MatchValue{{IndexName: FieldIndex("metadata.namespace"), Value: "default"}},
		},
		"Match field for listing namespace pods with metadata.namespace field selector": {
			labelSelector: "",
			fieldSelector: "metadata.namespace=kube-system",
			indexLabels:   []string{},
			indexFields:   []string{"metadata.namespace"},
			ctx: request.WithRequestInfo(context.Background(), &request.RequestInfo{
				IsResourceRequest: true,
				Path:              "/api/v1/namespaces/default/pods",
				Verb:              "list",
				APIPrefix:         "api",
				APIGroup:          "",
				APIVersion:        "v1",
				Namespace:         "default",
				Resource:          "pods",
			}),
			expected: []MatchValue{{IndexName: FieldIndex("metadata.namespace"), Value: "kube-system"}},
		},
		"Match field for listing all pods without metadata.namespace field selector": {
			labelSelector: "",
			fieldSelector: "",
			indexLabels:   []string{},
			indexFields:   []string{"metadata.namespace"},
			ctx: request.WithRequestInfo(context.Background(), &request.RequestInfo{
				IsResourceRequest: true,
				Path:              "/api/v1/pods",
				Verb:              "list",
				APIPrefix:         "api",
				APIGroup:          "",
				APIVersion:        "v1",
				Namespace:         "",
				Resource:          "pods",
			}),
			expected: nil,
		},
		"Match field for listing all pods with metadata.namespace field selector": {
			labelSelector: "",
			fieldSelector: "metadata.namespace=default",
			indexLabels:   []string{},
			indexFields:   []string{"metadata.namespace"},
			ctx: request.WithRequestInfo(context.Background(), &request.RequestInfo{
				IsResourceRequest: true,
				Path:              "/api/v1/pods",
				Verb:              "list",
				APIPrefix:         "api",
				APIGroup:          "",
				APIVersion:        "v1",
				Namespace:         "default",
				Resource:          "pods",
			}),
			expected: []MatchValue{{IndexName: FieldIndex("metadata.namespace"), Value: "default"}},
		},
		"Match label": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{},
			ctx:           context.Background(),
			expected:      []MatchValue{{IndexName: LabelIndex("name"), Value: "foo"}},
		},
		"Match field and label": {
			labelSelector: "name=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{"uid"},
			ctx:           context.Background(),
			expected:      []MatchValue{{IndexName: FieldIndex("uid"), Value: "12345"}, {IndexName: LabelIndex("name"), Value: "foo"}},
		},
		"Negative match field and label": {
			labelSelector: "name!=foo",
			fieldSelector: "uid!=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{"uid"},
			ctx:           context.Background(),
			expected:      nil,
		},
		"Negative match field and match label": {
			labelSelector: "name=foo",
			fieldSelector: "uid!=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{"uid"},
			ctx:           context.Background(),
			expected:      []MatchValue{{IndexName: LabelIndex("name"), Value: "foo"}},
		},
		"Negative match label and match field": {
			labelSelector: "name!=foo",
			fieldSelector: "uid=12345",
			indexLabels:   []string{"name"},
			indexFields:   []string{"uid"},
			ctx:           context.Background(),
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
		actual := sp.MatcherIndex(testCase.ctx)
		if !reflect.DeepEqual(testCase.expected, actual) {
			t.Errorf("%v: expected %v, got %v", name, testCase.expected, actual)
		}
	}
}
