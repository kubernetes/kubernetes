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

package orderedmap

import (
	"encoding/hex"
	"fmt"
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestBasicFeatures(t *testing.T) {
	n := 100
	om := New()

	// set(i, 2 * i)
	for i := 0; i < n; i++ {
		assertLenEqual(t, om, i)
		oldValue, present := om.Set(i, 2*i)
		assertLenEqual(t, om, i+1)

		assert.Nil(t, oldValue)
		assert.False(t, present)
	}

	// get what we just set
	for i := 0; i < n; i++ {
		value, present := om.Get(i)

		assert.Equal(t, 2*i, value)
		assert.True(t, present)
	}

	// forward iteration
	i := 0
	for pair := om.Oldest(); pair != nil; pair = pair.Next() {
		assert.Equal(t, i, pair.Key)
		assert.Equal(t, 2*i, pair.Value)
		i++
	}
	// backward iteration
	i = n - 1
	for pair := om.Newest(); pair != nil; pair = pair.Prev() {
		assert.Equal(t, i, pair.Key)
		assert.Equal(t, 2*i, pair.Value)
		i--
	}

	// double values for pairs with even keys
	for j := 0; j < n/2; j++ {
		i = 2 * j
		oldValue, present := om.Set(i, 4*i)

		assert.Equal(t, 2*i, oldValue)
		assert.True(t, present)
	}
	// and delete pairs with odd keys
	for j := 0; j < n/2; j++ {
		i = 2*j + 1
		assertLenEqual(t, om, n-j)
		value, present := om.Delete(i)
		assertLenEqual(t, om, n-j-1)

		assert.Equal(t, 2*i, value)
		assert.True(t, present)

		// deleting again shouldn't change anything
		value, present = om.Delete(i)
		assertLenEqual(t, om, n-j-1)
		assert.Nil(t, value)
		assert.False(t, present)
	}

	// get the whole range
	for j := 0; j < n/2; j++ {
		i = 2 * j
		value, present := om.Get(i)
		assert.Equal(t, 4*i, value)
		assert.True(t, present)

		i = 2*j + 1
		value, present = om.Get(i)
		assert.Nil(t, value)
		assert.False(t, present)
	}

	// check iterations again
	i = 0
	for pair := om.Oldest(); pair != nil; pair = pair.Next() {
		assert.Equal(t, i, pair.Key)
		assert.Equal(t, 4*i, pair.Value)
		i += 2
	}
	i = 2 * ((n - 1) / 2)
	for pair := om.Newest(); pair != nil; pair = pair.Prev() {
		assert.Equal(t, i, pair.Key)
		assert.Equal(t, 4*i, pair.Value)
		i -= 2
	}
}

func TestUpdatingDoesntChangePairsOrder(t *testing.T) {
	om := New()
	om.Set("foo", "bar")
	om.Set(12, 28)
	om.Set(78, 100)
	om.Set("bar", "baz")

	oldValue, present := om.Set(78, 102)
	assert.Equal(t, 100, oldValue)
	assert.True(t, present)

	assertOrderedPairsEqual(t, om,
		[]interface{}{"foo", 12, 78, "bar"},
		[]interface{}{"bar", 28, 102, "baz"})
}

func TestDeletingAndReinsertingChangesPairsOrder(t *testing.T) {
	om := New()
	om.Set("foo", "bar")
	om.Set(12, 28)
	om.Set(78, 100)
	om.Set("bar", "baz")

	// delete a pair
	oldValue, present := om.Delete(78)
	assert.Equal(t, 100, oldValue)
	assert.True(t, present)

	// re-insert the same pair
	oldValue, present = om.Set(78, 100)
	assert.Nil(t, oldValue)
	assert.False(t, present)

	assertOrderedPairsEqual(t, om,
		[]interface{}{"foo", 12, "bar", 78},
		[]interface{}{"bar", 28, "baz", 100})
}

func TestEmptyMapOperations(t *testing.T) {
	om := New()

	oldValue, present := om.Get("foo")
	assert.Nil(t, oldValue)
	assert.False(t, present)

	oldValue, present = om.Delete("bar")
	assert.Nil(t, oldValue)
	assert.False(t, present)

	assertLenEqual(t, om, 0)

	assert.Nil(t, om.Oldest())
	assert.Nil(t, om.Newest())
}

type dummyTestStruct struct {
	value string
}

func TestPackUnpackStructs(t *testing.T) {
	om := New()
	om.Set("foo", dummyTestStruct{"foo!"})
	om.Set("bar", dummyTestStruct{"bar!"})

	value, present := om.Get("foo")
	assert.True(t, present)
	if assert.NotNil(t, value) {
		assert.Equal(t, "foo!", value.(dummyTestStruct).value)
	}

	value, present = om.Set("bar", dummyTestStruct{"baz!"})
	assert.True(t, present)
	if assert.NotNil(t, value) {
		assert.Equal(t, "bar!", value.(dummyTestStruct).value)
	}

	value, present = om.Get("bar")
	assert.True(t, present)
	if assert.NotNil(t, value) {
		assert.Equal(t, "baz!", value.(dummyTestStruct).value)
	}
}

// shamelessly stolen from https://github.com/python/cpython/blob/e19a91e45fd54a56e39c2d12e6aaf4757030507f/Lib/test/test_ordered_dict.py#L55-L61
func TestShuffle(t *testing.T) {
	ranLen := 100

	for _, n := range []int{0, 10, 20, 100, 1000, 10000} {
		t.Run(fmt.Sprintf("shuffle test with %d items", n), func(t *testing.T) {
			om := New()

			keys := make([]interface{}, n)
			values := make([]interface{}, n)

			for i := 0; i < n; i++ {
				// we prefix with the number to ensure that we don't get any duplicates
				keys[i] = fmt.Sprintf("%d_%s", i, randomHexString(t, ranLen))
				values[i] = randomHexString(t, ranLen)

				value, present := om.Set(keys[i], values[i])
				assert.Nil(t, value)
				assert.False(t, present)
			}

			assertOrderedPairsEqual(t, om, keys, values)
		})
	}
}

/* Test helpers */

func assertOrderedPairsEqual(t *testing.T, om *OrderedMap, expectedKeys, expectedValues []interface{}) {
	assertOrderedPairsEqualFromNewest(t, om, expectedKeys, expectedValues)
	assertOrderedPairsEqualFromOldest(t, om, expectedKeys, expectedValues)
}

func assertOrderedPairsEqualFromNewest(t *testing.T, om *OrderedMap, expectedKeys, expectedValues []interface{}) {
	if assert.Equal(t, len(expectedKeys), len(expectedValues)) && assert.Equal(t, len(expectedKeys), om.Len()) {
		i := om.Len() - 1
		for pair := om.Newest(); pair != nil; pair = pair.Prev() {
			assert.Equal(t, expectedKeys[i], pair.Key)
			assert.Equal(t, expectedValues[i], pair.Value)
			i--
		}
	}
}

func assertOrderedPairsEqualFromOldest(t *testing.T, om *OrderedMap, expectedKeys, expectedValues []interface{}) {
	if assert.Equal(t, len(expectedKeys), len(expectedValues)) && assert.Equal(t, len(expectedKeys), om.Len()) {
		i := om.Len() - 1
		for pair := om.Newest(); pair != nil; pair = pair.Prev() {
			assert.Equal(t, expectedKeys[i], pair.Key)
			assert.Equal(t, expectedValues[i], pair.Value)
			i--
		}
	}
}

func assertLenEqual(t *testing.T, om *OrderedMap, expectedLen int) {
	assert.Equal(t, expectedLen, om.Len())

	// also check the list length, for good measure
	assert.Equal(t, expectedLen, om.list.Len())
}

func shuffle(a []interface{}) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := len(a) - 1; i > 0; i-- {
		j := r.Int() % (i + 1)
		tmp := a[j]
		a[j] = a[i]
		a[i] = tmp
	}
}

func randomHexString(t *testing.T, length int) string {
	b := length / 2
	randBytes := make([]byte, b)

	if n, err := rand.Read(randBytes); err != nil || n != b {
		if err == nil {
			err = fmt.Errorf("only got %v random bytes, expected %v", n, b)
		}
		t.Fatal(err)
	}

	return hex.EncodeToString(randBytes)
}
