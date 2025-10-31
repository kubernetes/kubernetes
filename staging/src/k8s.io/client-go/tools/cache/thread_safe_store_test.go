/*
Copyright 2019 The Kubernetes Authors.

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
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestThreadSafeStoreDeleteRemovesEmptySetsFromIndex(t *testing.T) {
	testIndexer := "testIndexer"

	indexers := Indexers{
		testIndexer: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)}
			return indexes, nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	testKey := "testKey"

	store.Add(testKey, testKey)

	// Assumption check, there should be a set for the `testKey` with one element in the added index
	set := store.index.indices[testIndexer][testKey]

	if len(set) != 1 {
		t.Errorf("Initial assumption of index backing string set having 1 element failed. Actual elements: %d", len(set))
		return
	}

	store.Delete(testKey)
	set, present := store.index.indices[testIndexer][testKey]

	if present {
		t.Errorf("Index backing string set not deleted from index. Set length: %d", len(set))
	}
}

func TestThreadSafeStoreAddKeepsNonEmptySetPostDeleteFromIndex(t *testing.T) {
	testIndexer := "testIndexer"
	testIndex := "testIndex"

	indexers := Indexers{
		testIndexer: func(obj interface{}) (strings []string, e error) {
			indexes := []string{testIndex}
			return indexes, nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	store.Add("retain", "retain")
	store.Add("delete", "delete")

	// Assumption check, there should be a set for the `testIndex` with two elements
	set := store.index.indices[testIndexer][testIndex]

	if len(set) != 2 {
		t.Errorf("Initial assumption of index backing string set having 2 elements failed. Actual elements: %d", len(set))
		return
	}

	store.Delete("delete")
	set, present := store.index.indices[testIndexer][testIndex]

	if !present {
		t.Errorf("Index backing string set erroneously deleted from index.")
		return
	}

	if len(set) != 1 {
		t.Errorf("Index backing string set has incorrect length, expect 1. Set length: %d", len(set))
	}
}

func TestThreadSafeStoreIndexingFunctionsWithMultipleValues(t *testing.T) {
	testIndexer := "testIndexer"

	indexers := Indexers{
		testIndexer: func(obj interface{}) ([]string, error) {
			return strings.Split(obj.(string), ","), nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	store.Add("key1", "foo")
	store.Add("key2", "bar")

	assert := assert.New(t)

	compare := func(key string, expected []string) error {
		values := sets.List(store.index.indices[testIndexer][key])
		if cmp.Equal(values, expected) {
			return nil
		}
		return fmt.Errorf("unexpected index for key %s, diff=%s", key, cmp.Diff(values, expected))
	}

	assert.NoError(compare("foo", []string{"key1"}))
	assert.NoError(compare("bar", []string{"key2"}))

	store.Update("key2", "foo,bar")

	assert.NoError(compare("foo", []string{"key1", "key2"}))
	assert.NoError(compare("bar", []string{"key2"}))

	store.Update("key1", "foo,bar")

	assert.NoError(compare("foo", []string{"key1", "key2"}))
	assert.NoError(compare("bar", []string{"key1", "key2"}))

	store.Add("key3", "foo,bar,baz")

	assert.NoError(compare("foo", []string{"key1", "key2", "key3"}))
	assert.NoError(compare("bar", []string{"key1", "key2", "key3"}))
	assert.NoError(compare("baz", []string{"key3"}))

	store.Update("key1", "foo")

	assert.NoError(compare("foo", []string{"key1", "key2", "key3"}))
	assert.NoError(compare("bar", []string{"key2", "key3"}))
	assert.NoError(compare("baz", []string{"key3"}))

	store.Update("key2", "bar")

	assert.NoError(compare("foo", []string{"key1", "key3"}))
	assert.NoError(compare("bar", []string{"key2", "key3"}))
	assert.NoError(compare("baz", []string{"key3"}))

	store.Delete("key1")

	assert.NoError(compare("foo", []string{"key3"}))
	assert.NoError(compare("bar", []string{"key2", "key3"}))
	assert.NoError(compare("baz", []string{"key3"}))

	store.Delete("key3")

	assert.NoError(compare("foo", []string{}))
	assert.NoError(compare("bar", []string{"key2"}))
	assert.NoError(compare("baz", []string{}))
}

func TestThreadSafeStoreRV(t *testing.T) {
	t.Run("Initial state", func(t *testing.T) {
		store := NewThreadSafeStore(Indexers{}, Indices{}).(*threadSafeMap)
		if rv := store.GetObservedResourceVersion(); rv != "" {
			t.Errorf("Expected initial RV to be \"\", got %q", rv)
		}
	})

	t.Run("Add Update and Delete", func(t *testing.T) {
		store := NewThreadSafeStore(Indexers{}, Indices{}).(*threadSafeMap)

		// Add obj with RV "10"
		store.Add("key1", &metav1.ObjectMeta{ResourceVersion: "10"})
		if rv := store.GetObservedResourceVersion(); rv != "10" {
			t.Errorf("Expected RV to be \"10\", got %q", rv)
		}

		// Add obj with lower RV "5"
		store.Add("key2", &metav1.ObjectMeta{ResourceVersion: "5"})
		if rv := store.GetObservedResourceVersion(); rv != "10" {
			t.Errorf("Expected RV to remain \"10\" after adding lower RV, got %q", rv)
		}

		// Add obj with same RV "10"
		store.Add("key3", &metav1.ObjectMeta{ResourceVersion: "10"})
		if rv := store.GetObservedResourceVersion(); rv != "10" {
			t.Errorf("Expected RV to remain \"10\" after adding same RV, got %q", rv)
		}

		// Add obj with higher RV "20"
		store.Add("key4", &metav1.ObjectMeta{ResourceVersion: "20"})
		if rv := store.GetObservedResourceVersion(); rv != "20" {
			t.Errorf("Expected RV to be \"20\", got %q", rv)
		}

		// Update obj with lower RV "15"
		store.Update("key4", &metav1.ObjectMeta{ResourceVersion: "15"})
		if rv := store.GetObservedResourceVersion(); rv != "20" {
			t.Errorf("Expected RV to remain \"20\" after updating with lower RV, got %q", rv)
		}

		// Add obj with invalid RV "abc" (should be ignored due to comparison error)
		store.Add("key5", &metav1.ObjectMeta{ResourceVersion: "abc"})
		if rv := store.GetObservedResourceVersion(); rv != "20" {
			t.Errorf("Expected RV to remain \"20\" after adding invalid RV, got %q", rv)
		}

		// Delete an earlier object with a newer RV
		store.DeleteWithObject("key4", &metav1.ObjectMeta{ResourceVersion: "30"})
		if rv := store.GetObservedResourceVersion(); rv != "30" {
			t.Errorf("Expected RV to remain \"30\" after adding empty RV, got %q", rv)
		}

		// Add obj with no RV
		store.Add("key6", &metav1.ObjectMeta{ResourceVersion: ""})
		if rv := store.GetObservedResourceVersion(); rv != "30" {
			t.Errorf("Expected RV to remain \"30\" after adding empty RV, got %q", rv)
		}

		// Add non-meta object
		store.Add("key7", "just a string")
		if rv := store.GetObservedResourceVersion(); rv != "30" {
			t.Errorf("Expected RV to remain \"30\" after adding non-meta object, got %q", rv)
		}
	})

	t.Run("Replace", func(t *testing.T) {
		store := NewThreadSafeStore(Indexers{}, Indices{}).(*threadSafeMap)
		store.Add("key1", &metav1.ObjectMeta{ResourceVersion: "10"})

		if rv := store.GetObservedResourceVersion(); rv != "10" {
			t.Fatalf("Setup failed, expected RV \"10\", got %q", rv)
		}

		items := map[string]interface{}{
			"key3": &metav1.ObjectMeta{ResourceVersion: "40"},
			"key2": &metav1.ObjectMeta{ResourceVersion: "30"},
		}

		store.Replace(items, "50")

		if rv := store.GetObservedResourceVersion(); rv != "50" {
			t.Errorf("Expected RV to be \"50\" after Replace(), got %q", rv)
		}
	})

	t.Run("Concurrency", func(t *testing.T) {
		store := NewThreadSafeStore(Indexers{}, Indices{}).(*threadSafeMap)

		var wg sync.WaitGroup
		numWriters := 50
		numReaders := 50

		// Start writers
		for i := 0; i < numWriters; i++ {
			wg.Add(1)
			go func(rv int) {
				defer wg.Done()
				key := fmt.Sprintf("key-%d", rv)
				// RVs will be "1", "3", "5", ..., "99"
				obj := &metav1.ObjectMeta{ResourceVersion: strconv.Itoa(rv*2 + 1)}
				store.Add(key, obj)
			}(i)
		}

		// Start readers
		for i := 0; i < numReaders; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_ = store.GetObservedResourceVersion()
			}()
		}

		wg.Wait()

		// The highest RV will be from i=49. (49*2 + 1) = 99
		expectedRV := "99"
		if rv := store.GetObservedResourceVersion(); rv != expectedRV {
			t.Errorf("Expected final RV to be %q after concurrent access, got %q", expectedRV, rv)
		}
	})
}

func BenchmarkIndexer(b *testing.B) {
	testIndexer := "testIndexer"

	indexers := Indexers{
		testIndexer: func(obj interface{}) (strings []string, e error) {
			indexes := []string{obj.(string)}
			return indexes, nil
		},
	}

	indices := Indices{}
	store := NewThreadSafeStore(indexers, indices).(*threadSafeMap)

	// The following benchmark imitates what is happening in indexes
	// used in storage layer, where indexing is mostly static (e.g.
	// indexing objects by their (namespace, name)).
	// The 5000 number imitates indexing nodes in 5000-node cluster.
	objectCount := 5000
	objects := make([]string, 0, 5000)
	for i := 0; i < objectCount; i++ {
		objects = append(objects, fmt.Sprintf("object-number-%d", i))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.Update(objects[i%objectCount], objects[i%objectCount])
	}
}
