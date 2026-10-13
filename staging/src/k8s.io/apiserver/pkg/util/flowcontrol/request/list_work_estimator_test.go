/*
Copyright 2021 The Kubernetes Authors.

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

package request

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/storage"
)

func TestListWorkEstimator(t *testing.T) {

	defaultCfg := DefaultWorkEstimatorConfig()

	tests := []struct {
		name                      string
		totalSize                 int64
		objectCount               int64
		request                   metav1.ListOptions
		isListFromCache           bool
		matchesSingle             bool
		expectObjectCountEstimate uint64
		expectObjectSizeEstimate  uint64
	}{
		{
			name:                      "100KB resource, 100 objects, storage",
			totalSize:                 100_000 - 1,
			objectCount:               99,
			isListFromCache:           false,
			expectObjectCountEstimate: 2,
			expectObjectSizeEstimate:  1,
		},
		{
			name:                      "100KB resource, 10 objects, storage",
			totalSize:                 100_000 - 1,
			objectCount:               10,
			isListFromCache:           false,
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  1,
		},
		{
			name:                      "100KB resource, 99 objects, cache",
			totalSize:                 100_000 - 1,
			objectCount:               99,
			isListFromCache:           true,
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  1,
		},
		{
			name:                      "100KB resource, 10 objects, cache",
			totalSize:                 100_000 - 1,
			objectCount:               10,
			isListFromCache:           true,
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  1,
		},
		{
			name:                      "1MB resource, 1000 objects, storage",
			totalSize:                 1_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           false,
			expectObjectCountEstimate: 20,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "1MB resource, 1000 objects, storage, limit 100",
			totalSize:                 1_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           false,
			request:                   metav1.ListOptions{Limit: 100},
			expectObjectCountEstimate: 2,
			expectObjectSizeEstimate:  1,
		},
		{
			name:                      "1MB resource, 1000 objects, storage, limit 100, selector",
			totalSize:                 1_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           false,
			request:                   metav1.ListOptions{Limit: 100, LabelSelector: "a"},
			expectObjectCountEstimate: 11,
			expectObjectSizeEstimate:  5,
		},
		{
			name:                      "1MB resource, 1000 objects, cache",
			totalSize:                 1_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           true,
			expectObjectCountEstimate: 10,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "1MB resource, 1000 objects, cache, limit 100",
			totalSize:                 1_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           true,
			request:                   metav1.ListOptions{Limit: 100},
			expectObjectCountEstimate: 10,
			expectObjectSizeEstimate:  1,
		},
		{
			name:                      "1MB resource, 1000 objects, cache, limit 100, selector",
			totalSize:                 1_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           true,
			request:                   metav1.ListOptions{Limit: 100, LabelSelector: "a"},
			expectObjectCountEstimate: 10,
			expectObjectSizeEstimate:  1,
		},
		{
			name:                      "1MB resource, 99 objects, storage",
			totalSize:                 1_000_000 - 1,
			objectCount:               99,
			isListFromCache:           false,
			expectObjectCountEstimate: 2,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "1MB resource, 99 objects, storage, limit 10",
			totalSize:                 1_000_000 - 1,
			objectCount:               99,
			isListFromCache:           false,
			request:                   metav1.ListOptions{Limit: 10},
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  2,
		},
		{
			name:                      "1MB resource, 99 objects, storage, limit 10, selector",
			totalSize:                 1_000_000 - 1,
			objectCount:               99,
			isListFromCache:           false,
			request:                   metav1.ListOptions{Limit: 10, LabelSelector: "a"},
			expectObjectCountEstimate: 2,
			expectObjectSizeEstimate:  5,
		},
		{
			name:                      "1MB resource, 99 objects, cache",
			totalSize:                 1_000_000 - 1,
			objectCount:               99,
			isListFromCache:           true,
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "10MB resource, 1000 objects, store",
			totalSize:                 10_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           false,
			expectObjectCountEstimate: 20,
			expectObjectSizeEstimate:  100,
		},
		{
			name:                      "10MB resource, 1000 objects, store, limit 100",
			totalSize:                 10_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           false,
			request:                   metav1.ListOptions{Limit: 100},
			expectObjectCountEstimate: 2,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "10MB resource, 1000 objects, store, limit 100, selector",
			totalSize:                 10_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           false,
			request:                   metav1.ListOptions{Limit: 100, LabelSelector: "a"},
			expectObjectCountEstimate: 11,
			expectObjectSizeEstimate:  50,
		},
		{
			name:                      "10MB resource, 1000 objects, cache",
			totalSize:                 10_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           true,
			expectObjectCountEstimate: 10,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "10MB resource, 1000 objects, cache, limit 100",
			totalSize:                 10_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           true,
			request:                   metav1.ListOptions{Limit: 100},
			expectObjectCountEstimate: 10,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "10MB resource, 1000 objects, cache, limit 100, selector",
			totalSize:                 10_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           true,
			request:                   metav1.ListOptions{Limit: 100, LabelSelector: "a"},
			expectObjectCountEstimate: 10,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "10MB resource, 99 objects, store",
			totalSize:                 10_000_000 - 1,
			objectCount:               99,
			isListFromCache:           false,
			expectObjectCountEstimate: 2,
			expectObjectSizeEstimate:  100,
		},
		{
			name:                      "10MB resource, 99 objects, store, limit 10",
			totalSize:                 10_000_000 - 1,
			objectCount:               99,
			isListFromCache:           false,
			request:                   metav1.ListOptions{Limit: 10},
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  11,
		},
		{
			name:                      "10MB resource, 99 objects, store, limit 10, selector",
			totalSize:                 10_000_000 - 1,
			objectCount:               99,
			isListFromCache:           false,
			request:                   metav1.ListOptions{Limit: 10, LabelSelector: "a"},
			expectObjectCountEstimate: 2,
			expectObjectSizeEstimate:  50,
		},
		{
			name:                      "10MB resource, 99 objects, cache",
			totalSize:                 10_000_000 - 1,
			objectCount:               99,
			isListFromCache:           true,
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "10MB resource, 99 objects, cache, limit 10",
			totalSize:                 10_000_000 - 1,
			objectCount:               99,
			isListFromCache:           true,
			request:                   metav1.ListOptions{Limit: 10},
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "10MB resource, 99 objects, cache, limit 10, selector",
			totalSize:                 10_000_000 - 1,
			objectCount:               99,
			isListFromCache:           true,
			request:                   metav1.ListOptions{Limit: 10, LabelSelector: "a"},
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "1000MB resource, 1000 objects, store, matchesSingle",
			totalSize:                 1_000_000_000 - 1,
			objectCount:               1000,
			matchesSingle:             true,
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  10,
		},
		{
			name:                      "1000MB resource, 1000 objects, cache, matchesSingle",
			totalSize:                 1_000_000_000 - 1,
			objectCount:               1000,
			isListFromCache:           true,
			matchesSingle:             true,
			expectObjectCountEstimate: 1,
			expectObjectSizeEstimate:  10,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			estimator := newListWorkEstimator(nil, defaultCfg, nil)

			stats := storage.Stats{ObjectCount: test.objectCount, EstimatedAverageObjectSizeBytes: test.totalSize / test.objectCount}

			objectCountEstimate := estimator.seatsBasedOnObjectCount(stats, test.request, test.isListFromCache, test.matchesSingle)
			objectSizeEstimage := estimator.seatsBasedOnObjectSize(stats, test.request, test.isListFromCache, test.matchesSingle)

			if objectCountEstimate != test.expectObjectCountEstimate {
				t.Errorf("Expected object count work estimate to match, expected: %d, but got: %d", test.expectObjectCountEstimate, objectCountEstimate)
			}
			if objectSizeEstimage != test.expectObjectSizeEstimate {
				t.Errorf("Expected object size work estimate to match, expected: %d, but got: %d", test.expectObjectSizeEstimate, objectSizeEstimage)
			}
		})
	}
}
