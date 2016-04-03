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
	"fmt"
	"strings"
	"time"

	"k8s.io/heapster/store/daystore"
	"k8s.io/heapster/store/statstore"
)

// latestTimestamp returns its largest time.Time argument
func latestTimestamp(first time.Time, second time.Time) time.Time {
	if first.After(second) {
		return first
	}
	return second
}

// newInfoType is an InfoType Constructor, which returns a new InfoType.
// Initial fields for the new InfoType can be provided as arguments.
// A nil argument results in a newly-allocated map for that field.
func newInfoType(metrics map[string]*daystore.DayStore, labels map[string]string, context map[string]*statstore.TimePoint) InfoType {
	if metrics == nil {
		metrics = make(map[string]*daystore.DayStore)
	}
	if labels == nil {
		labels = make(map[string]string)
	}
	if context == nil {
		context = make(map[string]*statstore.TimePoint)
	}
	return InfoType{
		Creation: time.Time{},
		Metrics:  metrics,
		Labels:   labels,
		Context:  context,
	}
}

// addContainerToMap creates or finds a ContainerInfo element under a map[string]*ContainerInfo
func addContainerToMap(container_name string, dict map[string]*ContainerInfo) *ContainerInfo {
	var container_ptr *ContainerInfo

	if val, ok := dict[container_name]; ok {
		// A container already exists under that name, return the address
		container_ptr = val
	} else {
		container_ptr = &ContainerInfo{
			InfoType: newInfoType(nil, nil, nil),
		}
		dict[container_name] = container_ptr
	}
	return container_ptr
}

// addTimePoints adds the values of two TimePoints as uint64.
// addTimePoints returns a new TimePoint with the added Value fields
// and the Timestamp of the first TimePoint.
func addTimePoints(tp1 statstore.TimePoint, tp2 statstore.TimePoint) statstore.TimePoint {
	maxTS := tp1.Timestamp
	if maxTS.Before(tp2.Timestamp) {
		maxTS = tp2.Timestamp
	}
	return statstore.TimePoint{
		Timestamp: maxTS,
		Value:     tp1.Value + tp2.Value,
	}
}

// popTPSlice pops the first element of a TimePoint Slice, removing it from the slice.
// popTPSlice receives a *[]TimePoint and returns its first element.
func popTPSlice(tps_ptr *[]statstore.TimePoint) *statstore.TimePoint {
	if tps_ptr == nil {
		return nil
	}
	tps := *tps_ptr
	if len(tps) == 0 {
		return nil
	}
	res := tps[0]
	if len(tps) == 1 {
		(*tps_ptr) = tps[0:0]
	}
	(*tps_ptr) = tps[1:]
	return &res
}

// addMatchingTimeseries performs addition over two timeseries with unique timestamps.
// addMatchingTimeseries returns a []TimePoint of the resulting aggregated timeseries.
// Assumes time-descending order of both []TimePoint parameters and the return slice.
func addMatchingTimeseries(left []statstore.TimePoint, right []statstore.TimePoint) []statstore.TimePoint {
	var cur_left *statstore.TimePoint
	var cur_right *statstore.TimePoint
	result := []statstore.TimePoint{}

	// Merge timeseries into result until either one is empty
	cur_left = popTPSlice(&left)
	cur_right = popTPSlice(&right)
	for cur_left != nil && cur_right != nil {
		result = append(result, addTimePoints(*cur_left, *cur_right))
		if cur_left.Timestamp.Equal(cur_right.Timestamp) {
			cur_left = popTPSlice(&left)
			cur_right = popTPSlice(&right)
		} else if cur_left.Timestamp.After(cur_right.Timestamp) {
			cur_left = popTPSlice(&left)
		} else {
			cur_right = popTPSlice(&right)
		}
	}
	if cur_left == nil && cur_right != nil {
		result = append(result, *cur_right)
	} else if cur_left != nil && cur_right == nil {
		result = append(result, *cur_left)
	}

	// Append leftover elements from non-empty timeseries
	if len(left) > 0 {
		result = append(result, left...)
	} else if len(right) > 0 {
		result = append(result, right...)
	}

	return result
}

// instantFromCumulativeMetric calculates the value of an instantaneous metric from two
// points of a cumulative metric, such as cpu/usage.
// The inputs are the value and timestamp of the newer cumulative datapoint,
// and a pointer to a TimePoint holding the previous cumulative datapoint.
func instantFromCumulativeMetric(value uint64, stamp time.Time, prev *statstore.TimePoint) (uint64, error) {
	if prev == nil {
		return uint64(0), fmt.Errorf("unable to calculate instant metric with nil previous TimePoint")
	}
	if !stamp.After(prev.Timestamp) {
		return uint64(0), fmt.Errorf("the previous TimePoint is not earlier in time than the newer one")
	}
	tdelta := uint64(stamp.Sub(prev.Timestamp).Nanoseconds())
	// Divide metric by nanoseconds that have elapsed, multiply by 1000 to get an unsigned metric
	if value < prev.Value {
		return uint64(0), fmt.Errorf("the provided value %d is less than the previous one %d", value, prev.Value)
	}
	// Divide metric by nanoseconds that have elapsed, multiply by 1000 to get an unsigned metric
	vdelta := (value - prev.Value) * 1000

	instaVal := vdelta / tdelta
	prev.Value = value
	prev.Timestamp = stamp
	return instaVal, nil
}

// getStats extracts derived stats from an InfoType and their timestamp.
func getStats(info InfoType) (map[string]StatBundle, time.Time) {
	res := make(map[string]StatBundle)
	var timestamp time.Time
	for key, ds := range info.Metrics {
		last, lastMax, _ := ds.Hour.Last()
		timestamp = last.Timestamp
		minAvg := last.Value
		minPct := lastMax
		minMax := lastMax
		hourAvg, _ := ds.Hour.Average()
		hourPct, _ := ds.Hour.Percentile(0.95)
		hourMax, _ := ds.Hour.Max()
		dayAvg, _ := ds.Average()
		dayPct, _ := ds.NinetyFifth()
		dayMax, _ := ds.Max()

		res[key] = StatBundle{
			Minute: Stats{
				Average:     minAvg,
				NinetyFifth: minPct,
				Max:         minMax,
			},
			Hour: Stats{
				Average:     hourAvg,
				NinetyFifth: hourPct,
				Max:         hourMax,
			},
			Day: Stats{
				Average:     dayAvg,
				NinetyFifth: dayPct,
				Max:         dayMax,
			},
		}
	}
	return res, timestamp
}

func epsilonFromMetric(metric string) uint64 {
	switch metric {
	case cpuLimit:
		return cpuLimitEpsilon
	case cpuUsage:
		return cpuUsageEpsilon
	case memLimit:
		return memLimitEpsilon
	case memUsage:
		return memUsageEpsilon
	case memWorking:
		return memWorkingEpsilon
	default:
		if strings.Contains(metric, fsLimit) {
			return fsLimitEpsilon
		}

		if strings.Contains(metric, fsUsage) {
			return fsUsageEpsilon
		}

		return defaultEpsilon
	}
}
