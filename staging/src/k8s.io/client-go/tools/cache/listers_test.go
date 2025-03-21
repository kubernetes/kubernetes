/*
Copyright 2025 The Kubernetes Authors.

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

package cache

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
)

func TestListAll(t *testing.T) {
	testCases := []struct {
		name           string
		numObjects     int
		numMatching    int
		matchingLabels map[string]string
		selector       labels.Selector
		expectedCount  int
	}{
		{
			name:           "subset match",
			numObjects:     100000,
			numMatching:    10000,
			matchingLabels: map[string]string{"match": "true"},
			selector:       labels.SelectorFromSet(map[string]string{"match": "true"}),
			expectedCount:  10000,
		},
		{
			name:           "all match",
			numObjects:     1000000,
			numMatching:    0,
			matchingLabels: map[string]string{},
			selector:       labels.Everything(),
			expectedCount:  1000000,
		},
		{
			name:           "no match",
			numObjects:     1000000,
			numMatching:    0,
			matchingLabels: map[string]string{"nomatch": "true"},
			selector:       labels.SelectorFromSet(map[string]string{"match": "true"}),
			expectedCount:  0,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			store := mustCreateStore(tc.numObjects, tc.numMatching, tc.matchingLabels)

			var matchingObjects int
			appendFn := func(obj interface{}) {
				matchingObjects++
			}

			err := ListAll(store, tc.selector, appendFn)
			if err != nil {
				t.Fatalf("ListAll returned an error: %v", err)
			}

			if matchingObjects != tc.expectedCount {
				t.Errorf("ListAll returned %d objects, expected %d", matchingObjects, tc.expectedCount)
			}
		})
	}
}

func mustCreateStore(numObjects int, numMatching int, labels map[string]string) Store {
	if numMatching > numObjects {
		panic("there can not be more matches than objects")
	}
	store := NewStore(func(obj interface{}) (string, error) {
		meta, err := meta.Accessor(obj)
		if err != nil {
			return "", err
		}
		return meta.GetName(), nil
	})
	// add matching objects to the store
	for i := 0; i < numObjects; i++ {
		obj := &metav1.PartialObjectMetadata{
			ObjectMeta: metav1.ObjectMeta{
				Name:   fmt.Sprintf("obj-%d", i),
				Labels: map[string]string{},
			},
		}
		if i < numMatching {
			obj.Labels = labels
		}
		err := store.Add(obj)
		if err != nil {
			panic("unexpected error")
		}
	}
	return store
}

func benchmarkLister(b *testing.B, numObjects int, numMatching int, label map[string]string) {
	store := mustCreateStore(numObjects, numMatching, label)
	selector := labels.SelectorFromSet(label)
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		err := ListAll(store, selector, func(m interface{}) {
		})
		if err != nil {
			b.Fatalf("ListAll returned an error: %v", err)
		}
	}
}

func BenchmarkLister_Match_1k_100(b *testing.B) {
	benchmarkLister(b, 1000, 100, map[string]string{"match": "true"})
}
func BenchmarkLister_Match_10k_100(b *testing.B) {
	benchmarkLister(b, 10000, 100, map[string]string{"match": "true"})
}

func BenchmarkLister_Match_100k_100(b *testing.B) {
	benchmarkLister(b, 100000, 100, map[string]string{"match": "true"})
}

func BenchmarkLister_Match_1M_100(b *testing.B) {
	benchmarkLister(b, 1000000, 100, map[string]string{"match": "true"})
}
