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

package model

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"k8s.io/heapster/store/daystore"
	"k8s.io/heapster/store/statstore"
)

// TestLatestsTimestamp tests all flows of latestTimeStamp.
func TestLatestTimestamp(t *testing.T) {
	assert := assert.New(t)
	past := time.Unix(1434212566, 0)
	future := time.Unix(1434212800, 0)
	assert.Equal(latestTimestamp(past, future), future)
	assert.Equal(latestTimestamp(future, past), future)
	assert.Equal(latestTimestamp(future, future), future)
}

// TestNewInfoType tests both flows of the InfoType constructor.
func TestNewInfoType(t *testing.T) {
	var (
		metrics = make(map[string]*daystore.DayStore)
		labels  = make(map[string]string)
		context = make(map[string]*statstore.TimePoint)
	)
	metrics["test"] = newDayStore()
	labels["name"] = "test"
	assert := assert.New(t)

	// Invocation with no parameters
	new_infotype := newInfoType(nil, nil, nil)
	assert.Empty(new_infotype.Metrics)
	assert.Empty(new_infotype.Labels)

	// Invocation with all parameters
	new_infotype = newInfoType(metrics, labels, context)
	assert.Equal(new_infotype.Metrics, metrics)
	assert.Equal(new_infotype.Labels, labels)
	assert.Equal(new_infotype.Context, context)
}

// TestAddContainerToMap tests all the flows of addContainerToMap.
func TestAddContainerToMap(t *testing.T) {
	new_map := make(map[string]*ContainerInfo)

	// First Call: A new ContainerInfo is created
	cinfo := addContainerToMap("new_container", new_map)

	assert := assert.New(t)
	assert.NotNil(cinfo)
	assert.NotNil(cinfo.Metrics)
	assert.NotNil(cinfo.Labels)
	assert.Equal(new_map["new_container"], cinfo)

	// Second Call: A ContainerInfo is already available for that key
	new_cinfo := addContainerToMap("new_container", new_map)
	assert.Equal(new_map["new_container"], new_cinfo)
	assert.Equal(cinfo, new_cinfo)
}

// TestAddTimePoints tests all flows of addTimePoints.
func TestAddTimePoints(t *testing.T) {
	var (
		tp1 = statstore.TimePoint{
			Timestamp: time.Unix(1434212800, 0),
			Value:     uint64(500),
		}
		tp2 = statstore.TimePoint{
			Timestamp: time.Unix(1434212805, 0),
			Value:     uint64(700),
		}
		assert = assert.New(t)
	)
	new_tp := addTimePoints(tp1, tp2)
	assert.Equal(new_tp.Timestamp, tp2.Timestamp)
	assert.Equal(new_tp.Value, tp1.Value+tp2.Value)
}

// TestPopTPSlice tests all flows of PopTPSlice.
func TestPopTPSlice(t *testing.T) {
	var (
		tp1 = statstore.TimePoint{
			Timestamp: time.Now(),
			Value:     uint64(3),
		}
		tp2 = statstore.TimePoint{
			Timestamp: time.Now(),
			Value:     uint64(4),
		}
		assert = assert.New(t)
	)

	// Invocations with no popped element
	assert.Nil(popTPSlice(nil))
	assert.Nil(popTPSlice(&[]statstore.TimePoint{}))

	// Invocation with one element
	tps := []statstore.TimePoint{tp1}
	res := popTPSlice(&tps)
	assert.Equal(*res, tp1)
	assert.Empty(tps)

	// Invocation with two elements
	tps = []statstore.TimePoint{tp1, tp2}
	res = popTPSlice(&tps)
	assert.Equal(*res, tp1)
	assert.Len(tps, 1)
	assert.Equal(tps[0], tp2)
}

// TestAddMatchingTimeseries tests the normal flow of addMatchingTimeseries.
func TestAddMatchingTimeseries(t *testing.T) {
	var (
		tp11 = statstore.TimePoint{
			Timestamp: time.Unix(1434212800, 0),
			Value:     uint64(4),
		}
		tp21 = statstore.TimePoint{
			Timestamp: time.Unix(1434212805, 0),
			Value:     uint64(9),
		}
		tp31 = statstore.TimePoint{
			Timestamp: time.Unix(1434212807, 0),
			Value:     uint64(10),
		}
		tp41 = statstore.TimePoint{
			Timestamp: time.Unix(1434212850, 0),
			Value:     uint64(533),
		}

		tp12 = statstore.TimePoint{
			Timestamp: time.Unix(1434212800, 0),
			Value:     uint64(4),
		}
		tp22 = statstore.TimePoint{
			Timestamp: time.Unix(1434212806, 0),
			Value:     uint64(8),
		}
		tp32 = statstore.TimePoint{
			Timestamp: time.Unix(1434212807, 0),
			Value:     uint64(11),
		}
		tp42 = statstore.TimePoint{
			Timestamp: time.Unix(1434212851, 0),
			Value:     uint64(534),
		}
		tp52 = statstore.TimePoint{
			Timestamp: time.Unix(1434212859, 0),
			Value:     uint64(538),
		}
		assert = assert.New(t)
	)
	// Invocation with 1+1 data points
	tps1 := []statstore.TimePoint{tp11}
	tps2 := []statstore.TimePoint{tp12}
	new_ts := addMatchingTimeseries(tps1, tps2)
	assert.Len(new_ts, 1)
	assert.Equal(new_ts[0].Timestamp, tp11.Timestamp)
	assert.Equal(new_ts[0].Value, uint64(8))

	// Invocation with 3+1 data points
	tps1 = []statstore.TimePoint{tp52, tp42, tp11}
	tps2 = []statstore.TimePoint{tp12}
	new_ts = addMatchingTimeseries(tps1, tps2)
	assert.Len(new_ts, 3)
	assert.Equal(new_ts[0].Timestamp, tp52.Timestamp)
	assert.Equal(new_ts[0].Value, tp52.Value+tp12.Value)
	assert.Equal(new_ts[1].Timestamp, tp42.Timestamp)
	assert.Equal(new_ts[1].Value, tp42.Value+tp12.Value)
	assert.Equal(new_ts[2].Timestamp, tp11.Timestamp)
	assert.Equal(new_ts[2].Value, tp11.Value+tp12.Value)

	// Invocation with 4+5 data points
	tps1 = []statstore.TimePoint{tp41, tp31, tp21, tp11}
	tps2 = []statstore.TimePoint{tp52, tp42, tp32, tp22, tp12}
	new_ts = addMatchingTimeseries(tps1, tps2)
	assert.Len(new_ts, 7)
	assert.Equal(new_ts[0].Timestamp, tp52.Timestamp)
	assert.Equal(new_ts[0].Value, tp52.Value+tp41.Value)

	assert.Equal(new_ts[1].Timestamp, tp42.Timestamp)
	assert.Equal(new_ts[1].Value, tp42.Value+tp41.Value)

	assert.Equal(new_ts[2].Timestamp, tp41.Timestamp)
	assert.Equal(new_ts[2].Value, tp41.Value+tp32.Value)

	assert.Equal(new_ts[3].Timestamp, tp31.Timestamp)
	assert.Equal(new_ts[3].Value, tp31.Value+tp32.Value)

	assert.Equal(new_ts[4].Timestamp, tp22.Timestamp)
	assert.Equal(new_ts[4].Value, tp22.Value+tp21.Value)

	assert.Equal(new_ts[5].Timestamp, tp21.Timestamp)
	assert.Equal(new_ts[5].Value, tp21.Value+tp12.Value)

	assert.Equal(new_ts[6].Timestamp, tp11.Timestamp)
	assert.Equal(new_ts[6].Value, tp11.Value+tp12.Value)
}

// TestAddMatchingTimeseriesEmpty tests the alternate flows of addMatchingTimeseries.
// Three permutations of empty parameters are tested.
func TestAddMatchingTimeseriesEmpty(t *testing.T) {
	var (
		tp12 = statstore.TimePoint{
			Timestamp: time.Unix(1434212800, 0),
			Value:     uint64(4),
		}
		tp22 = statstore.TimePoint{
			Timestamp: time.Unix(1434212806, 0),
			Value:     uint64(8),
		}
		tp32 = statstore.TimePoint{
			Timestamp: time.Unix(1434212807, 0),
			Value:     uint64(11),
		}
		assert = assert.New(t)
	)
	empty_tps := []statstore.TimePoint{}
	tps := []statstore.TimePoint{tp12, tp22, tp32}

	// First call: first argument is empty
	new_ts := addMatchingTimeseries(empty_tps, tps)
	assert.Equal(new_ts, tps)

	// Second call: second argument is empty
	new_ts = addMatchingTimeseries(tps, empty_tps)
	assert.Equal(new_ts, tps)

	// Third call: both arguments are empty
	new_ts = addMatchingTimeseries(empty_tps, empty_tps)
	assert.Equal(new_ts, empty_tps)
}

// TestInstantFromCumulativeMetric tests all the flows of instantFromCumulativeMetric.
func TestInstantFromCumulativeMetric(t *testing.T) {
	var (
		new_value = uint64(15390000000000)
		now       = time.Now().Round(time.Second)
		assert    = assert.New(t)
	)
	afterNow := now.Add(3 * time.Second)
	oldTP := &statstore.TimePoint{
		Timestamp: now,
		Value:     uint64(13851000000000),
	}

	// Invocation with nil prev argument
	val, err := instantFromCumulativeMetric(new_value, afterNow, nil)
	assert.Error(err)
	assert.Equal(val, uint64(0))

	// Invocation with regular arguments
	val, err = instantFromCumulativeMetric(new_value, afterNow, oldTP)
	assert.NoError(err)
	assert.Equal(val, 513*1000)

	// Second Invocation with regular arguments, prev TP has changed
	newerVal := uint64(15900000000000)
	val, err = instantFromCumulativeMetric(newerVal, afterNow.Add(time.Second), oldTP)
	assert.NoError(err)
	assert.Equal(val, 510*1000)

}

// TestGetStats tests all flows of getStats.
func TestGetStats(t *testing.T) {
	var (
		metrics = make(map[string]*daystore.DayStore)
		now     = time.Now()
		assert  = assert.New(t)
	)
	// Populate a new DayStore with two hours of data
	metrics[cpuLimit] = newDayStore()
	for i := 0; i < 2; i++ {
		metrics[cpuLimit].Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour),
			Value:     uint64(50),
		})
		metrics[cpuLimit].Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour).Add(15 * time.Minute),
			Value:     uint64(500),
		})
		metrics[cpuLimit].Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour).Add(30 * time.Minute),
			Value:     uint64(2000),
		})
		metrics[cpuLimit].Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour).Add(45 * time.Minute),
			Value:     uint64(1000),
		})
		metrics[cpuLimit].Put(statstore.TimePoint{
			Timestamp: now.Add(time.Duration(i) * time.Hour).Add(59 * time.Minute),
			Value:     uint64(15000),
		})
	}

	// flushes the latest TimePoint to the last Hour
	metrics[cpuLimit].Put(statstore.TimePoint{
		Timestamp: now.Add(2 * time.Hour),
		Value:     uint64(300000),
	})

	metrics[memUsage] = newDayStore()
	metrics[memUsage].Put(statstore.TimePoint{
		Timestamp: now,
		Value:     uint64(50),
	})
	metrics[memUsage].Put(statstore.TimePoint{
		Timestamp: now.Add(30 * time.Minute),
		Value:     uint64(2000),
	})
	metrics[memUsage].Put(statstore.TimePoint{
		// flushes the previous TimePoint to the last Hour
		Timestamp: now.Add(60 * time.Minute),
		Value:     uint64(1000),
	})
	info := newInfoType(metrics, nil, nil)

	res, _ := getStats(info)
	assert.Len(res, 2)
	assert.Equal(res[cpuLimit].Minute.Average, uint64(300000))
	assert.Equal(res[cpuLimit].Minute.Max, uint64(300000))
	assert.Equal(res[cpuLimit].Minute.NinetyFifth, uint64(300000))

	assert.Equal(res[cpuLimit].Hour.Average, uint64(1133))
	assert.Equal(res[cpuLimit].Hour.Max, uint64(300000))
	assert.Equal(res[cpuLimit].Hour.NinetyFifth, uint64(2000))

	assert.Equal(res[cpuLimit].Day.Max, res[cpuLimit].Hour.Max)
	assert.Equal(res[cpuLimit].Day.Average, res[cpuLimit].Hour.Average)
	assert.Equal(res[cpuLimit].Day.NinetyFifth, res[cpuLimit].Hour.NinetyFifth)
}

func TestEpsilonFromMetric(t *testing.T) {
	assert := assert.New(t)
	assert.Equal(cpuLimitEpsilon, epsilonFromMetric(cpuLimit))
	assert.Equal(cpuUsageEpsilon, epsilonFromMetric(cpuUsage))
	assert.Equal(memLimitEpsilon, epsilonFromMetric(memLimit))
	assert.Equal(memUsageEpsilon, epsilonFromMetric(memUsage))
	assert.Equal(memWorkingEpsilon, epsilonFromMetric(memWorking))
	assert.Equal(fsLimitEpsilon, epsilonFromMetric(fsLimit))
	assert.Equal(fsUsageEpsilon, epsilonFromMetric(fsUsage))
	assert.Equal(defaultEpsilon, epsilonFromMetric("other"))
}
