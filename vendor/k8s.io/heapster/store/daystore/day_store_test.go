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

package daystore

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	statstore "k8s.io/heapster/store/statstore"
)

// TestPutError tests the error flow of Put
func TestPutError(t *testing.T) {
	assert := assert.New(t)
	now := time.Now().Truncate(time.Minute)

	ds := NewDayStore(10, time.Minute)

	// Put a normal Point
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now,
		Value:     uint64(100),
	}))

	// Put a Point in the past
	assert.Error(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(-time.Minute),
		Value:     uint64(100),
	}))
}

// TestNewDayStore tests the validity of the NewDayStore constructor.
func TestNewDayStore(t *testing.T) {
	assert := assert.New(t)

	// Invocation with a resolution of 1 minute
	ds := NewDayStore(10, time.Minute)
	assert.Equal(ds.Hour.MaxSize(), time.Hour)

	// Invocation with a resolution of 6 minutes
	ds = NewDayStore(10, 6*time.Minute)
	assert.Equal(ds.Hour.MaxSize(), time.Hour)

	// Invocation with a resolution of 1 Hour
	ds = NewDayStore(10, time.Hour)
	assert.Equal(ds.Hour.MaxSize(), time.Hour)

	// Invocation with a resolution of 11 minutes.
	// The window should be 66 minutes in that case
	ds = NewDayStore(10, 11*time.Minute)
	target := time.Time{}.Add(time.Hour).Add(6 * time.Minute).Sub(time.Time{})
	assert.Equal(ds.Hour.MaxSize(), target)
}

// TestDayStore tests all methods of a DayStore.
func TestDayStore(t *testing.T) {
	ds := NewDayStore(100, time.Minute)
	assert := assert.New(t)
	now := time.Now().Truncate(time.Minute)

	// Invocations with nothing in the dayStore
	avg, err := ds.Average()
	assert.Error(err)
	assert.Equal(avg, 0)

	max, err := ds.Max()
	assert.Error(err)
	assert.Equal(max, 0)

	nf, err := ds.NinetyFifth()
	assert.Error(err)
	assert.Equal(nf, 0)

	// Put in 1 hour of data
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now,
		Value:     uint64(100),
	}))
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(30 * time.Minute),
		Value:     uint64(200),
	}))

	// No data has been flushed to the window, the Average, Max and NinetyFifth should be -
	// equal to that of the Hour StatStore
	avg, err = ds.Average()
	assert.NoError(err)
	avgHr, err := ds.Hour.Average()
	assert.NoError(err)
	assert.Equal(avg, avgHr)

	// The max is in the last hour
	max, err = ds.Max()
	assert.NoError(err)
	maxHr, err := ds.Hour.Max()
	assert.NoError(err)
	assert.Equal(max, maxHr)

	// The Ninetyfifth percentile does not take the last hour into account
	nf, err = ds.NinetyFifth()
	assert.NoError(err)
	nfHr, err := ds.Hour.Percentile(0.95)
	assert.NoError(err)
	assert.Equal(nf, nfHr)

	// Put in 11 more hours of data.
	for i := 1; i <= 6; i++ {
		assert.NoError(ds.Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour),
			Value:     uint64((2*i + 1) * 100),
		}))
		assert.NoError(ds.Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour).Add(30 * time.Minute),
			Value:     uint64((2 * (i + 1)) * 100),
		}))
	}

	for i := 1; i <= 5; i++ {
		assert.NoError(ds.Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i+6) * time.Hour),
			Value:     uint64((2*i - 1) * 100),
		}))
		assert.NoError(ds.Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i+6) * time.Hour).Add(30 * time.Minute),
			Value:     uint64((2 * i) * 100),
		}))
	}

	// 13th Hour
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(12 * time.Hour),
		Value:     uint64(1100),
	}))
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(12 * time.Hour).Add(30 * time.Minute),
		Value:     uint64(1500),
	}))
	// flush the second value of the 13th hour
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(12 * time.Hour).Add(32 * time.Minute),
		Value:     uint64(100),
	}))

	// Half-Day Invocations.
	// 12 Hours have been flushed to the DayStore
	// The current hour has two values flushed

	// The average is expected to be the average of the first 12 hours.
	avg, err = ds.Average()
	assert.NoError(err)
	assert.Equal(avg, uint64(666))

	// The max is in the last hour
	max, err = ds.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(1500))

	// The Ninetyfifth percentile does not take the last hour into account
	nf, err = ds.NinetyFifth()
	assert.NoError(err)
	assert.Equal(nf, uint64(1400))

	// Assert validity of the cache by performing the same checks
	avg, err = ds.Average()
	assert.NoError(err)
	assert.Equal(avg, uint64(666))
	max, err = ds.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(1500))
	nf, err = ds.NinetyFifth()
	assert.NoError(err)
	assert.Equal(nf, uint64(1400))

	// Put in 12 more hours of data, at half-hour intervals.
	for i := 1; i <= 7; i++ {
		assert.NoError(ds.Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i+12) * time.Hour),
			Value:     uint64((2*i - 1) * 100),
		}))
		assert.NoError(ds.Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i+12) * time.Hour).Add(30 * time.Minute),
			Value:     uint64((2 * i) * 100),
		}))
	}
	for i := 1; i <= 3; i++ {
		assert.NoError(ds.Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i+19) * time.Hour),
			Value:     uint64((2*i - 1) * 100),
		}))
		assert.NoError(ds.Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i+19) * time.Hour).Add(30 * time.Minute),
			Value:     uint64((2 * i) * 100),
		}))
	}

	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(23 * time.Hour),
		Value:     uint64(21000),
	}))
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(23 * time.Hour).Add(30 * time.Minute),
		Value:     uint64(700),
	}))

	// 25th hour stored, current last hour of the DayStore
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(24 * time.Hour),
		Value:     uint64(900),
	}))
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(24 * time.Hour).Add(30 * time.Minute),
		Value:     uint64(1000),
	}))

	// 26th hour
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(25 * time.Hour),
		Value:     uint64(100),
	}))
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(25 * time.Hour).Add(30 * time.Minute),
		Value:     uint64(100),
	}))

	// Full-Day Invocations.
	// The DayStore has rewinded by one hour already
	avg, err = ds.Average()
	assert.NoError(err)
	assert.Equal(avg, uint64(1108))

	max, err = ds.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(21000))

	nf, err = ds.NinetyFifth()
	assert.NoError(err)
	assert.Equal(nf, uint64(21000))

	// Rewind once more.
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(26 * time.Hour),
		Value:     uint64(5000),
	}))
	assert.NoError(ds.Put(statstore.TimePoint{
		Timestamp: now.Add(26 * time.Hour).Add(30 * time.Minute),
		Value:     uint64(5000),
	}))

	// Invocations after second rewind
	// Drop in average because of the 26th hour having a value of 100
	nf, err = ds.NinetyFifth()
	assert.NoError(err)
	assert.Equal(nf, uint64(21000))

	max, err = ds.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(21000))

	avg, err = ds.Average()
	assert.NoError(err)
	assert.Equal(avg, uint64(1097))

	// Assert validity of the cache
	avg, err = ds.Average()
	assert.NoError(err)
	assert.Equal(avg, uint64(1097))

	max, err = ds.Max()
	assert.NoError(err)
	assert.Equal(max, uint64(21000))

	nf, err = ds.NinetyFifth()
	assert.NoError(err)
	assert.Equal(nf, uint64(21000))

	// Get Invocation, a full hour should be available plus the lastPut
	res := ds.Hour.Get(time.Time{}, time.Time{})
	assert.Len(res, 61)
}
