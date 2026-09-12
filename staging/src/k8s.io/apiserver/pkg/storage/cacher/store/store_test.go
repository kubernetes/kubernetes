/*
Copyright 2022 The Kubernetes Authors.

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

package store

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
)

func TestStoreSingleKey(t *testing.T) {
	t.Run("cache.Indexer", func(t *testing.T) {
		store := NewIndexer(testStoreIndexers())
		testStoreSingleKey(t, store)
	})
	t.Run("btree", func(t *testing.T) {
		store := newThreadedBtreeStoreIndexer(ElementIndexers(testStoreIndexers()), btreeDegree)
		testStoreSingleKey(t, store)
	})
}

func testStoreSingleKey(t *testing.T, store Indexer) {
	assertStoreEmpty(t, store, "foo")

	require.NoError(t, store.Add(testStorageElement("foo", "bar", 1)))
	assertStoreSingleKey(t, store, "foo", "bar", 1)

	require.NoError(t, store.Update(testStorageElement("foo", "baz", 2)))
	assertStoreSingleKey(t, store, "foo", "baz", 2)

	require.NoError(t, store.Update(testStorageElement("foo", "baz", 3)))
	assertStoreSingleKey(t, store, "foo", "baz", 3)

	require.NoError(t, store.Replace([]*Element{testStorageElement("foo", "bar", 4)}))
	assertStoreSingleKey(t, store, "foo", "bar", 4)

	require.NoError(t, store.Delete(testStorageElement("foo", "", 0)))
	assertStoreEmpty(t, store, "foo")

	require.NoError(t, store.Delete(testStorageElement("foo", "", 0)))
}

func TestStoreIndexerSingleKey(t *testing.T) {
	t.Run("cache.Indexer", func(t *testing.T) {
		store := NewIndexer(testStoreIndexers())
		testStoreIndexerSingleKey(t, store)
	})
	t.Run("btree", func(t *testing.T) {
		store := newThreadedBtreeStoreIndexer(ElementIndexers(testStoreIndexers()), btreeDegree)
		testStoreIndexerSingleKey(t, store)
	})
}

func testStoreIndexerSingleKey(t *testing.T, store Indexer) {
	items, err := store.ByIndex("by_val", "bar")
	require.NoError(t, err)
	assert.Empty(t, items)

	require.NoError(t, store.Add(testStorageElement("foo", "bar", 1)))
	items, err = store.ByIndex("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, []*Element{
		testStorageElement("foo", "bar", 1),
	}, items)

	require.NoError(t, store.Update(testStorageElement("foo", "baz", 2)))
	items, err = store.ByIndex("by_val", "bar")
	require.NoError(t, err)
	assert.Empty(t, items)
	items, err = store.ByIndex("by_val", "baz")
	require.NoError(t, err)
	assert.Equal(t, []*Element{
		testStorageElement("foo", "baz", 2),
	}, items)

	require.NoError(t, store.Update(testStorageElement("foo", "baz", 3)))
	items, err = store.ByIndex("by_val", "bar")
	require.NoError(t, err)
	assert.Empty(t, items)
	items, err = store.ByIndex("by_val", "baz")
	require.NoError(t, err)
	assert.Equal(t, []*Element{
		testStorageElement("foo", "baz", 3),
	}, items)

	require.NoError(t, store.Replace([]*Element{
		testStorageElement("foo", "bar", 4),
	}))
	items, err = store.ByIndex("by_val", "bar")
	require.NoError(t, err)
	assert.Equal(t, []*Element{
		testStorageElement("foo", "bar", 4),
	}, items)
	items, err = store.ByIndex("by_val", "baz")
	require.NoError(t, err)
	assert.Empty(t, items)

	require.NoError(t, store.Delete(testStorageElement("foo", "", 0)))
	items, err = store.ByIndex("by_val", "baz")
	require.NoError(t, err)
	assert.Empty(t, items)

	require.NoError(t, store.Delete(testStorageElement("foo", "", 0)))
}

func assertStoreEmpty(t *testing.T, store Indexer, nonExistingKey string) {
	elem, ok := store.GetByKey(nonExistingKey)
	assert.False(t, ok)
	assert.Nil(t, elem)

	items := store.List()
	assert.Empty(t, items)
}

func assertStoreSingleKey(t *testing.T, store Indexer, expectKey, expectValue string, expectRV int) {
	elem, ok := store.GetByKey(expectKey)
	assert.True(t, ok)
	assert.Equal(t, expectValue, elem.Object.(fakeObj).value)

	items := store.List()
	assert.Equal(t, []*Element{testStorageElement(expectKey, expectValue, expectRV)}, items)
}

func testStorageElement(key, value string, rv int) *Element {
	return &Element{Key: key, Object: fakeObj{value: value, rv: rv}}
}

type fakeObj struct {
	value string
	rv    int
}

func (f fakeObj) GetObjectKind() schema.ObjectKind { return nil }
func (f fakeObj) DeepCopyObject() runtime.Object   { return nil }

var _ runtime.Object = (*fakeObj)(nil)

func testStoreIndexFunc(obj interface{}) ([]string, error) {
	return []string{obj.(fakeObj).value}, nil
}

func testStoreIndexers() *cache.Indexers {
	indexers := cache.Indexers{}
	indexers["by_val"] = testStoreIndexFunc
	return &indexers
}
