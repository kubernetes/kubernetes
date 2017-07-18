/*
Copyright 2017 The Kubernetes Authors.

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

package core

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/predicates"
)

func TestUpdateCachedPredicateItem(t *testing.T) {
	tests := []struct {
		name               string
		pod                *v1.Pod
		predicateKey       string
		nodeName           string
		fit                bool
		reasons            []algorithm.PredicateFailureReason
		equivalenceHash    uint64
		expectPredicateMap bool
		expectCacheItem    HostPredicate
	}{
		{
			name:               "test 1",
			pod:                &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "testPod"}},
			predicateKey:       "GeneralPredicates",
			nodeName:           "node1",
			fit:                true,
			equivalenceHash:    123,
			expectPredicateMap: false,
			expectCacheItem: HostPredicate{
				Fit: true,
			},
		},
		{
			name:               "test 2",
			pod:                &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "testPod"}},
			predicateKey:       "GeneralPredicates",
			nodeName:           "node2",
			fit:                false,
			equivalenceHash:    123,
			expectPredicateMap: true,
			expectCacheItem: HostPredicate{
				Fit: false,
			},
		},
	}
	for _, test := range tests {
		// this case does not need to calculate equivalence hash, just pass an empty function
		fakeGetEquivalencePodFunc := func(pod *v1.Pod) interface{} { return nil }
		ecache := NewEquivalenceCache(fakeGetEquivalencePodFunc)
		if test.expectPredicateMap {
			ecache.algorithmCache[test.nodeName] = newAlgorithmCache()
			predicateItem := HostPredicate{
				Fit: true,
			}
			ecache.algorithmCache[test.nodeName].predicatesCache.Add(test.predicateKey,
				PredicateMap{
					test.equivalenceHash: predicateItem,
				})
		}
		ecache.UpdateCachedPredicateItem(test.pod, test.nodeName, test.predicateKey, test.fit, test.reasons, test.equivalenceHash)

		value, ok := ecache.algorithmCache[test.nodeName].predicatesCache.Get(test.predicateKey)
		if !ok {
			t.Errorf("Failed : %s, can't find expected cache item: %v", test.name, test.expectCacheItem)
		} else {
			cachedMapItem := value.(PredicateMap)
			if !reflect.DeepEqual(cachedMapItem[test.equivalenceHash], test.expectCacheItem) {
				t.Errorf("Failed : %s, expected cached item: %v, but got: %v", test.name, test.expectCacheItem, cachedMapItem[test.equivalenceHash])
			}
		}
	}
}

type predicateItemType struct {
	fit     bool
	reasons []algorithm.PredicateFailureReason
}

func TestCachedPredicateItem(t *testing.T) {
	tests := []struct {
		name                              string
		pod                               *v1.Pod
		nodeName                          string
		predicateKey                      string
		equivalenceHashForUpdatePredicate uint64
		equivalenceHashForCalPredicate    uint64
		cachedItem                        predicateItemType
		expectedInvalidPredicateKey       bool
		expectedInvalidEquivalenceHash    bool
		expectedPredicateItem             predicateItemType
	}{
		{
			name:     "test 1",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "testPod"}},
			nodeName: "node1",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedInvalidPredicateKey: true,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{},
			},
		},
		{
			name:     "test 2",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "testPod"}},
			nodeName: "node2",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			cachedItem: predicateItemType{
				fit: true,
			},
			expectedInvalidPredicateKey: false,
			expectedPredicateItem: predicateItemType{
				fit:     true,
				reasons: []algorithm.PredicateFailureReason{},
			},
		},
		{
			name:     "test 3",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "testPod"}},
			nodeName: "node3",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    123,
			predicateKey:                      "GeneralPredicates",
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedInvalidPredicateKey: false,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
		},
		{
			name:     "test 4",
			pod:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "testPod"}},
			nodeName: "node4",
			equivalenceHashForUpdatePredicate: 123,
			equivalenceHashForCalPredicate:    456,
			predicateKey:                      "GeneralPredicates",
			cachedItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{predicates.ErrPodNotFitsHostPorts},
			},
			expectedInvalidPredicateKey:    false,
			expectedInvalidEquivalenceHash: true,
			expectedPredicateItem: predicateItemType{
				fit:     false,
				reasons: []algorithm.PredicateFailureReason{},
			},
		},
	}

	for _, test := range tests {
		// this case does not need to calculate equivalence hash, just pass an empty function
		fakeGetEquivalencePodFunc := func(pod *v1.Pod) interface{} { return nil }
		ecache := NewEquivalenceCache(fakeGetEquivalencePodFunc)
		// set cached item to equivalence cache
		ecache.UpdateCachedPredicateItem(test.pod, test.nodeName, test.predicateKey, test.cachedItem.fit, test.cachedItem.reasons, test.equivalenceHashForUpdatePredicate)
		// if we want to do invalid, invalid the cached item
		if test.expectedInvalidPredicateKey {
			predicateKeys := sets.NewString()
			predicateKeys.Insert(test.predicateKey)
			ecache.InvalidateCachedPredicateItem(test.nodeName, predicateKeys)
		}
		// calculate predicate with equivalence cache
		fit, reasons, invalid := ecache.PredicateWithECache(test.pod, test.nodeName, test.predicateKey, test.equivalenceHashForCalPredicate)
		// returned invalid should match expectedInvalidPredicateKey or expectedInvalidEquivalenceHash
		if test.equivalenceHashForUpdatePredicate != test.equivalenceHashForCalPredicate {
			if invalid != test.expectedInvalidEquivalenceHash {
				t.Errorf("Failed : %s when using invalid equivalenceHash, expected invalid: %v, but got: %v", test.name, test.expectedInvalidEquivalenceHash, invalid)
			}
		} else {
			if invalid != test.expectedInvalidPredicateKey {
				t.Errorf("Failed : %s, expected invalid: %v, but got: %v", test.name, test.expectedInvalidPredicateKey, invalid)
			}
		}
		// returned predicate result should match expected predicate item
		if fit != test.expectedPredicateItem.fit {
			t.Errorf("Failed : %s, expected fit: %v, but got: %v", test.name, test.cachedItem.fit, fit)
		}
		if !reflect.DeepEqual(reasons, test.expectedPredicateItem.reasons) {
			t.Errorf("Failed : %s, expected reasons: %v, but got: %v", test.name, test.cachedItem.reasons, reasons)
		}
	}
}
