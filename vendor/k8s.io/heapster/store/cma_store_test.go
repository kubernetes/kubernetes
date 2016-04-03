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

func TestCMAInit(t *testing.T) {
	store := NewCMAStore()
	data := store.Get(time.Now().Add(-time.Minute), time.Now())
	assert.Empty(t, len(data), time.Now())
}

func TestCMANilInsert(t *testing.T) {
	store := NewCMAStore()
	assert.Error(t, store.Put(TimePoint{time.Now(), nil}))
}

func TestCMAInsertNormal(t *testing.T) {
	store := NewCMAStore()
	now := time.Now()
	assert := assert.New(t)
	assert.NoError(store.Put(TimePoint{now, uint64(2)}))
	assert.NoError(store.Put(TimePoint{now.Add(-time.Second), uint64(1)}))
	assert.NoError(store.Put(TimePoint{now.Add(time.Second), uint64(3)}))
	assert.NoError(store.Put(TimePoint{now.Add(-2 * time.Second), uint64(0)}))
	actual := store.Get(time.Time{}, now)
	require.Len(t, actual, 3)
	actual = store.Get(time.Time{}, time.Time{})
	require.Len(t, actual, 4)
}

func TestCMAInsertTime(t *testing.T) {
	zeroTime := time.Time{}
	store := NewCMAStore()
	now := time.Now()
	assert := assert.New(t)

	// Errors
	assert.Error(store.Put(TimePoint{zeroTime, uint64(2)}))
	assert.Error(store.Put(TimePoint{zeroTime, nil}))

	// Put 2 values
	assert.NoError(store.Put(TimePoint{now, uint64(2)}))
	assert.NoError(store.Put(TimePoint{now, uint64(6)}))
	assert.NoError(store.Put(TimePoint{now.Add(2 * time.Second), uint64(0)}))
	assert.NoError(store.Put(TimePoint{now.Add(2 * time.Second), uint64(20)}))
	assert.NoError(store.Put(TimePoint{now.Add(2 * time.Second), uint64(10)}))
	assert.NoError(store.Put(TimePoint{now.Add(2 * time.Second), uint64(1000)}))

	// Get Invocation with values 2 and 4
	actual := store.Get(zeroTime, now)
	require.Len(t, actual, 1)
	assert.Equal(actual[0].Value, uint64(4))

	// Get Invocation with values 0, 20, 10 and 1000
	actual = store.Get(zeroTime, zeroTime)
	require.Len(t, actual, 2)
	assert.Equal(actual[0].Value, uint64(257)) // 257.5 rounded down
	assert.Equal(actual[1].Value, uint64(4))

	// Get correct averages after new puts
	assert.NoError(store.Put(TimePoint{now.Add(2 * time.Second), uint64(0)}))
	assert.NoError(store.Put(TimePoint{now.Add(2 * time.Second), uint64(0)}))
	assert.NoError(store.Put(TimePoint{now.Add(2 * time.Second), uint64(0)}))
	assert.NoError(store.Put(TimePoint{now.Add(2 * time.Second), uint64(0)}))
	assert.NoError(store.Put(TimePoint{now.Add(-2 * time.Hour), uint64(999)}))
	actual = store.Get(zeroTime, zeroTime)
	require.Len(t, actual, 3)
	assert.Equal(actual[0].Value, uint64(126)) // 128.75 approximation
	assert.Equal(actual[1].Value, uint64(4))
	assert.Equal(actual[2].Value, uint64(999))

	// Get up until a point
	assert.NoError(store.Put(TimePoint{now.Add(-time.Hour), uint64(30)}))
	assert.NoError(store.Put(TimePoint{now.Add(-time.Hour), uint64(40)}))
	actual = store.Get(zeroTime, now.Add(-time.Hour))
	require.Len(t, actual, 2)
	assert.Equal(actual[0].Value, uint64(35))
	assert.Equal(actual[1].Value, uint64(999))

	// Get from a starting value
	actual = store.Get(now.Add(-time.Hour), zeroTime)
	require.Len(t, actual, 2)
	assert.Equal(actual[0].Value, uint64(126))
	assert.Equal(actual[1].Value, uint64(4))
}

func TestCMADelete(t *testing.T) {
	zeroTime := time.Time{}
	store := NewCMAStore()
	now := time.Now()
	assert := assert.New(t)

	// Invocation with empty buffer
	assert.NoError(store.Delete(now, now.Add(time.Minute)))

	assert.NoError(store.Put(TimePoint{now, uint64(2)}))
	assert.NoError(store.Put(TimePoint{now.Add(-time.Second), uint64(1)}))
	assert.NoError(store.Put(TimePoint{now.Add(-2 * time.Second), uint64(0)}))
	assert.NoError(store.Put(TimePoint{now.Add(time.Second), uint64(3)}))

	// Normal Invocation
	assert.NoError(store.Delete(now.Add(-time.Second), now))
	actual := store.Get(now.Add(-time.Second), zeroTime)
	require.Len(t, actual, 1)
	assert.Equal(uint64(3), actual[0].Value.(uint64))
	assert.NoError(store.Delete(zeroTime, zeroTime))
	assert.Empty(store.Get(zeroTime, zeroTime))

	// Invocation with no affected data
	assert.NoError(store.Put(TimePoint{now, uint64(2)}))
	assert.NoError(store.Delete(zeroTime, now.Add(-time.Minute)))
	actual = store.Get(zeroTime, zeroTime)
	assert.Len(actual, 1)
	assert.Equal(actual[0].Value, uint64(2))

	// Invocation with start time
	assert.NoError(store.Delete(now.Add(2*time.Minute), zeroTime))
	actual = store.Get(zeroTime, zeroTime)
	assert.Len(actual, 1)
	assert.Equal(actual[0].Value, uint64(2))

	// Invocation with error (start after end)
	assert.Error(store.Delete(now, now.Add(-time.Minute)))
}
