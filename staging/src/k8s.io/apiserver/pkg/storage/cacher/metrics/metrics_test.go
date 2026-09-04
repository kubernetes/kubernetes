/*
Copyright 2026 The Kubernetes Authors.

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

package metrics

import "testing"

func TestTerminationReason(t *testing.T) {
	testCases := []struct {
		name                        string
		inputLen, resultLen, resCap int
		want                        string
	}{{
		name: "result full means the serve loop was not consuming",
		// The common case: the serve loop is blocked downstream, so both
		// buffers back up.
		inputLen: 10, resultLen: 10, resCap: 10,
		want: "result_full",
	}, {
		name:     "result empty means the events never reached the serve loop",
		inputLen: 10, resultLen: 0, resCap: 10,
		want: "result_empty",
	}, {
		name:     "partially drained result",
		inputLen: 10, resultLen: 4, resCap: 10,
		want: "result_partial",
	}, {
		name:     "over capacity is still full",
		inputLen: 10, resultLen: 11, resCap: 10,
		want: "result_full",
	}, {
		name: "unbuffered result channel is never reported as full",
		// cap 0 would otherwise make len >= cap trivially true.
		inputLen: 10, resultLen: 0, resCap: 0,
		want: "result_empty",
	}}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if got := TerminationReason(tc.inputLen, tc.resultLen, tc.resCap); got != tc.want {
				t.Errorf("TerminationReason(%d, %d, %d) = %q, want %q", tc.inputLen, tc.resultLen, tc.resCap, got, tc.want)
			}
		})
	}
}

func TestChanSizeBucket(t *testing.T) {
	testCases := []struct {
		chanSize int
		want     string
	}{
		{chanSize: 0, want: "10"},
		{chanSize: 10, want: "10"},
		{chanSize: 11, want: "11-50"},
		{chanSize: 50, want: "11-50"},
		{chanSize: 51, want: "51-200"},
		{chanSize: 200, want: "51-200"},
		{chanSize: 201, want: "201-1000"},
		{chanSize: 1000, want: "201-1000"},
	}
	for _, tc := range testCases {
		if got := ChanSizeBucket(tc.chanSize); got != tc.want {
			t.Errorf("ChanSizeBucket(%d) = %q, want %q", tc.chanSize, got, tc.want)
		}
	}
}

func TestSerializationCacheObserversAreNilSafe(t *testing.T) {
	// cachingObjects built outside a cacher carry no observers.
	var observers *SerializationCacheObservers
	observers.RecordHit()
	observers.RecordMiss()
}
