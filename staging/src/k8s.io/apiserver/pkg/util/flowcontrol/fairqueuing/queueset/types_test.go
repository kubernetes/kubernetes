/*
Copyright 2021 The Kubernetes Authors.

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

package queueset

import (
	"testing"
)

func TestGetNextFinish(t *testing.T) {
	var virtualStart float64 = 100
	var G float64 = 60
	tests := []struct {
		name                  string
		requests              []*request
		virtualFinishExpected float64
	}{
		{
			name: "for the oldest request",
			requests: []*request{
				{width: 5},
				{width: 6},
				{width: 7},
			},
			virtualFinishExpected: virtualStart + (5 * G),
		},
		{
			name:                  "queue does not have any request waiting",
			requests:              []*request{},
			virtualFinishExpected: virtualStart,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			requests := newRequestFIFO()
			for i := range test.requests {
				requests.Enqueue(test.requests[i])
			}

			q := &queue{requests: requests}
			q.virtualStart = virtualStart
			virtualFinishGot := q.GetNextFinish(G)
			if test.virtualFinishExpected != virtualFinishGot {
				t.Errorf("Expected virtual finish time: %.9fs, but got: %.9fs", test.virtualFinishExpected, virtualFinishGot)
			}
		})
	}
}
