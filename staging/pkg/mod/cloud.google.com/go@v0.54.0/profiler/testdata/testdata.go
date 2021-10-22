// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package testdata provides some useful data sets for testing purposes.
package testdata

import (
	"github.com/google/pprof/profile"
)

var functions = []*profile.Function{
	{ID: 1, Name: "main", SystemName: "main", Filename: "main.go"},
	{ID: 2, Name: "foo", SystemName: "foo", Filename: "foo.go"},
	{ID: 3, Name: "foo_caller", SystemName: "foo_caller", Filename: "foo.go"},
}

const mainBinary = "/bin/main"

var mappings = []*profile.Mapping{
	{
		ID:              1,
		Start:           0x10000,
		Limit:           0x40000,
		File:            mainBinary,
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
	{
		ID:              2,
		Start:           0x1000,
		Limit:           0x4000,
		File:            "/lib/lib.so",
		HasFunctions:    true,
		HasFilenames:    true,
		HasLineNumbers:  true,
		HasInlineFrames: true,
	},
}

var locations = []*profile.Location{
	{
		ID:      1,
		Mapping: mappings[1],
		Address: 0x1000,
		Line: []profile.Line{
			{Function: functions[0], Line: 1},
		},
	},
	{
		ID:      2,
		Mapping: mappings[0],
		Address: 0x2000,
		Line: []profile.Line{
			{Function: functions[1], Line: 2},
			{Function: functions[2], Line: 1},
		},
	},
}

// HeapProfileCollected1 represents a heap profile which could be collected using
// pprof.WriteHeapProfile().
var HeapProfileCollected1 = &profile.Profile{
	DurationNanos: 10e9,
	SampleType: []*profile.ValueType{
		{Type: "alloc_objects", Unit: "count"},
		{Type: "alloc_space", Unit: "bytes"},
		{Type: "inuse_objects", Unit: "count"},
		{Type: "inuse_space", Unit: "bytes"},
	},
	Sample: []*profile.Sample{{
		Location: []*profile.Location{locations[0], locations[1]},
		Value:    []int64{10, 160, 10, 160},
		NumLabel: map[string][]int64{
			"bytes": {16},
		},
		NumUnit: map[string][]string{
			"bytes": {"bytes"},
		},
	}},
	Location: locations,
	Function: functions,
	Mapping:  mappings,
}

// HeapProfileUploaded represents the heap profile bytes we would expect to
// be uploaded if HeapProfileCollected1 were returned when profiling.
var HeapProfileUploaded = func() *profile.Profile {
	p := HeapProfileCollected1.Copy()
	p.Sample[0].Value = []int64{0, 0, 10, 160}
	return p
}()

// HeapProfileCollected2 represents a heap profile which could be collected using
// pprof.WriteHeapProfile().
var HeapProfileCollected2 = func() *profile.Profile {
	p := HeapProfileCollected1.Copy()
	p.Sample[0].Value = []int64{11, 176, 11, 176}
	return p
}()

// AllocProfileUploaded represents the allocation profile bytes we would expect
// to be uploaded if HeapProfileCollected1 was returned when first profiling
// and HeapProfileCollect2 was return when profiling the second time.
var AllocProfileUploaded = func() *profile.Profile {
	p := HeapProfileCollected1.Copy()
	p.DurationNanos = 5e9
	p.SampleType = []*profile.ValueType{
		{Type: "alloc_objects", Unit: "count"},
		{Type: "alloc_space", Unit: "bytes"},
	}
	p.Sample[0].Value = []int64{1, 16}
	return p
}()
