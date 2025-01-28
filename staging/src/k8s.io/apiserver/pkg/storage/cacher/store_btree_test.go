/*
Copyright 2024 The Kubernetes Authors.

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

package cacher

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStoreListOrdered(t *testing.T) {
	store := newThreadedBtreeStoreIndexer(nil, btreeDegree)
	assert.NoError(t, store.Add(testStorageElement("foo3", "bar3", 1)))
	assert.NoError(t, store.Add(testStorageElement("foo1", "bar2", 2)))
	assert.NoError(t, store.Add(testStorageElement("foo2", "bar1", 3)))
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, store.List())
}

func TestStoreListPrefix(t *testing.T) {
	store := newThreadedBtreeStoreIndexer(nil, btreeDegree)
	assert.NoError(t, store.Add(testStorageElement("foo3", "bar3", 1)))
	assert.NoError(t, store.Add(testStorageElement("foo1", "bar2", 2)))
	assert.NoError(t, store.Add(testStorageElement("foo2", "bar1", 3)))
	assert.NoError(t, store.Add(testStorageElement("bar", "baz", 4)))

	items, hasMore := store.ListPrefix("foo", "", 0)
	assert.False(t, hasMore)
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
		testStorageElement("foo2", "bar1", 3),
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items, hasMore = store.ListPrefix("foo2", "", 0)
	assert.False(t, hasMore)
	assert.Equal(t, []interface{}{
		testStorageElement("foo2", "bar1", 3),
	}, items)

	items, hasMore = store.ListPrefix("foo", "", 1)
	assert.True(t, hasMore)
	assert.Equal(t, []interface{}{
		testStorageElement("foo1", "bar2", 2),
	}, items)

	items, hasMore = store.ListPrefix("foo", "foo1\x00", 1)
	assert.True(t, hasMore)
	assert.Equal(t, []interface{}{
		testStorageElement("foo2", "bar1", 3),
	}, items)

	items, hasMore = store.ListPrefix("foo", "foo2\x00", 1)
	assert.False(t, hasMore)
	assert.Equal(t, []interface{}{
		testStorageElement("foo3", "bar3", 1),
	}, items)

	items, hasMore = store.ListPrefix("bar", "", 0)
	assert.False(t, hasMore)
	assert.Equal(t, []interface{}{
		testStorageElement("bar", "baz", 4),
	}, items)
}
