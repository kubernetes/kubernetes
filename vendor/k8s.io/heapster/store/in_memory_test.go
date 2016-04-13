// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package store

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// putWrapper puts a new TimePoint object to a TimeStore, given the time and value.
func putWrapper(store TimeStore, timestamp time.Time, value interface{}) error {
	new_point := TimePoint{
		Timestamp: timestamp,
		Value:     value,
	}
	return store.Put(new_point)
}

func TestInitialization(t *testing.T) {
	store := NewTimeStore()
	data := store.Get(time.Now().Add(-time.Minute), time.Now())
	assert.Empty(t, len(data), time.Now())
}

func TestNilInsert(t *testing.T) {
	store := NewTimeStore()
	assert.Error(t, putWrapper(store, time.Now(), nil))
}

func expectElements(t *testing.T, data []TimePoint) {
	for i := 0; i < len(data); i++ {
		assert.Equal(t, len(data)-i-1, data[i].Value.(int))
		if i != len(data)-1 {
			assert.True(t, data[i].Timestamp.After(data[i+1].Timestamp))
		}
	}
}

func TestInsert(t *testing.T) {
	store := NewTimeStore()
	now := time.Now()
	assert := assert.New(t)
	assert.NoError(putWrapper(store, now, 2))
	assert.NoError(putWrapper(store, now.Add(-time.Second), 1))
	assert.NoError(putWrapper(store, now.Add(time.Second), 3))
	assert.NoError(putWrapper(store, now.Add(-2*time.Second), 0))
	actual := store.Get(time.Time{}, now)
	require.Len(t, actual, 3)
	expectElements(t, actual)
	actual = store.Get(time.Time{}, time.Time{})
	require.Len(t, actual, 4)
	expectElements(t, actual)
}

func TestDelete(t *testing.T) {
	store := NewTimeStore()
	now := time.Now()
	assert := assert.New(t)
	assert.NoError(putWrapper(store, now, 2))
	assert.NoError(putWrapper(store, now.Add(-time.Second), 1))
	assert.NoError(putWrapper(store, now.Add(-2*time.Second), 0))
	assert.NoError(putWrapper(store, now.Add(time.Second), 3))
	assert.NoError(store.Delete(now.Add(-time.Second), now))
	actual := store.Get(now.Add(-time.Second), time.Time{})
	require.Len(t, actual, 1)
	assert.Equal(3, actual[0].Value.(int))
	assert.NoError(store.Delete(time.Time{}, time.Time{}))
	assert.Empty(store.Get(time.Time{}, time.Time{}))
}
