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
package statstore

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestLast tests all flows of the Last method.
func TestLast(t *testing.T) {
	// epsilon: 10, resolution: 1 minute, total: 1 hour, no percentiles
	store := NewStatStore(10, time.Minute, 60, []float64{})
	assert := assert.New(t)
	now := time.Now().Truncate(time.Minute)

	// Invocation with nothing in the StatStore - no result
	last, max, err := store.Last()
	assert.Error(err)
	assert.Equal(last, TimePoint{})
	assert.Equal(max, uint64(0))

	// Put 5 Points in the same minute. Average: 10029, Max: 50000
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(55),
	}))
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(time.Second),
		Value:     uint64(1),
	}))
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(12),
	}))
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(1 * time.Second),
		Value:     uint64(77),
	}))
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(1 * time.Second),
		Value:     uint64(50000),
	}))

	// Put one point in a previous minute, should be ignored.
	assert.Error(store.Put(TimePoint{
		Timestamp: now.Add(-2 * time.Minute),
		Value:     uint64(100000),
	}))

	last, max, err = store.Last()
	assert.NoError(err)
	assert.Equal(last.Timestamp, now)
	assert.Equal(last.Value, uint64(10029))
	assert.Equal(max, uint64(50000))

	// Put one value from the next minute
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(92),
	}))

	// Invocation where Last is a "fake" point added because of a missed resolution.
	last, max, err = store.Last()
	assert.NoError(err)
	assert.Equal(last.Timestamp, now.Add(time.Minute))
	assert.Equal(last.Value, uint64(92))
	assert.Equal(max, uint64(92))

	// Put one value from two more minutes later
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(10000),
	}))

	last, max, err = store.Last()
	assert.NoError(err)
	assert.Equal(last.Timestamp, now.Add(3*time.Minute))
	assert.Equal(last.Value, uint64(10000))
	assert.Equal(max, uint64(10000))
}

// TestMax tests all flows of the Max method.
func TestMax(t *testing.T) {
	// epsilon: 50, resolution: 1 minute, total: 5 minutes, no percentiles
	store := NewStatStore(50, time.Minute, 5, []float64{})
	assert := assert.New(t)
	now := time.Now().Truncate(time.Minute)

	// Invocation with nothing in the StatStore - no result
	max, err := store.Max()
	assert.Error(err)
	assert.Equal(max, uint64(0))

	// Put 3 Points in the same minute.  Max: 88
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(55),
	}))
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(time.Second),
		Value:     uint64(88),
	}))
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(21),
	}))

	// Invocation with elements only in lastPut
	max, err = store.Max()
	assert.Error(err)
	assert.Equal(max, uint64(0))

	// Put 1 Point in the next minute.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(199),
	}))

	// Invocation where the previous minute is now accessible
	max, err = store.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(199))

	// Put 1 Point in the next minute.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(22),
	}))

	// Put one point in a previous minute, should be ignored.
	assert.Error(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(100000),
	}))

	// Put one point in the next minute, same bucket as before.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(40),
	}))

	// Put one point in the next minute.
	// Even though the max is greater, this minute is currently in lastPut,
	// so it is excluded from the max.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(511),
	}))

	// Invocation with three minutes in three different buckets
	max, err = store.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(511))

	// Put one value from the next minute
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(550),
	}))

	// Invocation with four minutes
	max, err = store.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(550))

	// Call again to assert validity of the cache
	max, err = store.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(550))
}

// TestGet tests all flows of the Get method.
// Seven resolutions are stored in total, causing two rewinds.
func TestGet(t *testing.T) {
	// epsilon: 100, resolution: 1 minute, total: 5 minutes, no percentiles
	store := NewStatStore(100, time.Minute, 5, []float64{})
	assert := assert.New(t)
	require := require.New(t)
	now := time.Now().Truncate(time.Minute)
	zeroTime := time.Time{}

	// Invocation with nothing in the StatStore - empty result
	res := store.Get(zeroTime, zeroTime)
	assert.Len(res, 0)

	// Put 3 Points in the same minute.  Average: 150, Max: 190
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(120),
	}))
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(190),
	}))
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(time.Second),
		Value:     uint64(140),
	}))

	// Put 1 Point in the next minute.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(599),
	}))

	// Invocation with one element in the StatStore
	res = store.Get(zeroTime, zeroTime)
	require.Len(res, 2)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(599),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now,
		Value:     uint64(200),
	})
	// Put 1 Point in the next minute.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(22),
	}))

	// Put one point in a previous minute, should be ignored.
	assert.Error(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(100000),
	}))

	// Invocation with two elements in the StatStore
	res = store.Get(zeroTime, zeroTime)
	require.Len(res, 3)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(22),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[2], TimePoint{
		Timestamp: now,
		Value:     uint64(200),
	})

	// Put one point in the next minute, same bucket as before.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(110),
	}))

	// Put one point in the next minute.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(511),
	}))

	// Invocation with three elements in the StatStore
	res = store.Get(zeroTime, zeroTime)
	require.Len(res, 4)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(511),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(100),
	})
	assert.Equal(res[2], TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[3], TimePoint{
		Timestamp: now,
		Value:     uint64(200),
	})

	// Put one value from the next minute. Same bucket as the previous minute
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(540),
	}))

	// Put one value from the next minute. Same bucket as the previous minute
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(5 * time.Minute),
		Value:     uint64(550),
	}))

	// Invocation with a full StatStore and a multi-resolution bucket
	res = store.Get(zeroTime, zeroTime)
	require.Len(res, 6)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(5 * time.Minute),
		Value:     uint64(550),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[2], TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[3], TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(100),
	})
	assert.Equal(res[4], TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[5], TimePoint{
		Timestamp: now,
		Value:     uint64(200),
	})

	// Put one value from the next minute. Different bucket than the previous minute
	// This Put should cause a rewind for the first minute that was stored
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(6 * time.Minute),
		Value:     uint64(750),
	}))

	// Invocation after one rewind
	res = store.Get(zeroTime, zeroTime)
	require.Len(res, 6)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(6 * time.Minute),
		Value:     uint64(750),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now.Add(5 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[2], TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[3], TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[4], TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(100),
	})
	assert.Equal(res[5], TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(600),
	})

	// Cause one more rewind
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(7 * time.Minute),
		Value:     uint64(998),
	}))

	// Invocation after second rewind
	res = store.Get(zeroTime, zeroTime)
	require.Len(res, 6)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(7 * time.Minute),
		Value:     uint64(998),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now.Add(6 * time.Minute),
		Value:     uint64(800),
	})
	assert.Equal(res[2], TimePoint{
		Timestamp: now.Add(5 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[3], TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[4], TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[5], TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(100),
	})

	// Invocation with start after end
	res = store.Get(now.Add(10*time.Minute), now)
	assert.Len(res, 0)

	// Invocation with mid-length start-end range
	res = store.Get(now.Add(3*time.Minute), now.Add(5*time.Minute))
	assert.Len(res, 3)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(5 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[2], TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(600),
	})

	// Invocation with full-length start-end range.
	// The first TimePoint is ignored, as it is equal to start
	res = store.Get(now.Add(2*time.Minute), now.Add(6*time.Minute))
	assert.Len(res, 4)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(6 * time.Minute),
		Value:     uint64(800),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now.Add(5 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[2], TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(600),
	})
	assert.Equal(res[3], TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(600),
	})

	// Invocation with start-end range outside of the scope of values.
	res = store.Get(now.Add(-2*time.Minute), now.Add(time.Minute))
	assert.Len(res, 0)

	// Put one value from 10 minutes since the last Put.
	// This Put should force the entire StatStore to be filled with 1000.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(25 * time.Minute),
		Value:     uint64(1500),
	}))

	// Invocation after a future Put. Everything in between is placed in the last bucket
	res = store.Get(zeroTime, zeroTime)
	require.Len(res, 6)
	assert.Equal(res[0], TimePoint{
		Timestamp: now.Add(25 * time.Minute),
		Value:     uint64(1500),
	})
	assert.Equal(res[1], TimePoint{
		Timestamp: now.Add(24 * time.Minute),
		Value:     uint64(1000),
	})
	assert.Equal(res[2], TimePoint{
		Timestamp: now.Add(23 * time.Minute),
		Value:     uint64(1000),
	})
	assert.Equal(res[3], TimePoint{
		Timestamp: now.Add(22 * time.Minute),
		Value:     uint64(1000),
	})
	assert.Equal(res[4], TimePoint{
		Timestamp: now.Add(21 * time.Minute),
		Value:     uint64(1000),
	})
	assert.Equal(res[5], TimePoint{
		Timestamp: now.Add(20 * time.Minute),
		Value:     uint64(1000),
	})

}

// TestAverage tests all flows of the Average method.
func TestAverage(t *testing.T) {
	// epsilon: 100, resolution: 1 minute, total: 5 minutes, no percentiles
	store := NewStatStore(100, time.Minute, 5, []float64{})
	assert := assert.New(t)
	now := time.Now().Truncate(time.Minute)

	// Invocation with nothing in the StatStore - error
	avg, err := store.Average()
	assert.Error(err)
	assert.Equal(avg, uint64(0))

	// Populate StatStore
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(190),
	}))

	// Put 1 Point in the next minute. Same bucket (200)
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(199),
	}))

	// Invocation with one element in the StatStore
	avg, err = store.Average()
	assert.NoError(err)
	assert.Equal(avg, 200)

	// Put one Point in the next minute. Same bucket (200)
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(120),
	}))

	// Put one point in the next minute. Different bucket (600)
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(511),
	}))

	// Put one point in the next minute. Same bucket (600)
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(599),
	}))

	// Put one point in the next minute. StatStore is full now
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(5 * time.Minute),
		Value:     uint64(1081),
	}))

	// Invocation with five elements in the StatStore
	avg, err = store.Average()
	assert.NoError(err)
	assert.Equal(avg, uint64(360))

	// Call again to assert validity of the cache
	avg, err = store.Average()
	assert.NoError(err)
	assert.Equal(avg, uint64(360))
}

// TestPercentile tests all flows of the Percentile method.
func TestPercentile(t *testing.T) {
	// epsilon: 50, resolution: 1 minute, total: 5 minutes, two percentiles
	store := NewStatStore(50, time.Minute, 5, []float64{0.5, 0.95})
	assert := assert.New(t)
	now := time.Now().Truncate(time.Minute)

	// Invocation with nothing in the StatStore - error
	pc, err := store.Percentile(0.95)
	assert.Error(err)
	assert.Equal(pc, uint64(0))

	// Populate StatStore
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(190),
	}))

	// Put 1 Point in the next minute. Same bucket (200)
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(time.Minute),
		Value:     uint64(199),
	}))

	// Invocation with an unsupported percentile
	pc, err = store.Percentile(0.2)
	assert.Error(err)
	assert.Equal(pc, uint64(0))

	// Invocation with one element in the StatStore
	pc, err = store.Percentile(0.5)
	assert.NoError(err)
	assert.Equal(pc, 200)
	pc, err = store.Percentile(0.95)
	assert.NoError(err)
	assert.Equal(pc, 200)

	// Put one Point in the next minute. Different bucket (550)
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(2 * time.Minute),
		Value:     uint64(532),
	}))

	// Put one point in the next minute. Same bucket (550)
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(3 * time.Minute),
		Value:     uint64(511),
	}))

	// Put one point in the next minute. Different bucket (50)
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(4 * time.Minute),
		Value:     uint64(30),
	}))

	// Put one point in the next minute. StatStore is full now
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(5 * time.Minute),
		Value:     uint64(50),
	}))

	// Invocation with five elements in the StatStore
	pc, err = store.Percentile(0.5)
	assert.NoError(err)
	assert.Equal(pc, uint64(200))
	pc, err = store.Percentile(0.95)
	assert.NoError(err)
	assert.Equal(pc, uint64(550))

	// Call again to assert validity of the cache
	pc, err = store.Percentile(0.5)
	assert.NoError(err)
	assert.Equal(pc, uint64(200))
	pc, err = store.Percentile(0.95)
	assert.NoError(err)
	assert.Equal(pc, uint64(550))
}

// TestIsEmpty tests all flows of the IsEmpty method.
func TestIsEmpty(t *testing.T) {
	// epsilon: 50, resolution: 1 minute, total: 5 minutes, no percentiles
	store := NewStatStore(50, time.Minute, 5, []float64{})
	assert := assert.New(t)
	now := time.Now().Truncate(time.Minute)

	// Invocation with nothing in the StatStore
	assert.True(store.IsEmpty())

	// Put one point in the StatStore.
	assert.NoError(store.Put(TimePoint{
		Timestamp: now,
		Value:     uint64(30),
	}))

	// Invocation with values only in the lastPut field.
	assert.True(store.IsEmpty())

	// Put one point in the next minute. StatStore is not empty now
	assert.NoError(store.Put(TimePoint{
		Timestamp: now.Add(1 * time.Minute),
		Value:     uint64(50),
	}))

	// Invocation with a value in the StatStore
	assert.False(store.IsEmpty())
}

// TestMaxSize tests all flows of the MaxSize method.
func TestMaxSize(t *testing.T) {
	assert := assert.New(t)

	// Invocation with a StatStore of 5 minutes, 1 min resolution.
	store := NewStatStore(50, time.Minute, 5, []float64{})
	assert.Equal(5*time.Minute, store.MaxSize())

	// Invocation with a StatStore of 1 hour, 5 min resolution.
	store = NewStatStore(50, 5*time.Minute, 12, []float64{})
	assert.Equal(time.Hour, store.MaxSize())

	// Invocation with a StatStore of 1 hour, 1 hour resolution.
	store = NewStatStore(50, time.Hour, 1, []float64{})
	assert.Equal(time.Hour, store.MaxSize())

	// Invocation with a StatStore of 1 minute, 10 second resolution.
	store = NewStatStore(50, 10*time.Second, 6, []float64{})
	assert.Equal(time.Minute, store.MaxSize())
}
