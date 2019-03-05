/*
Copyright 2015 The Kubernetes Authors.

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

package socketmask

import (
	"reflect"
	"testing"
)

func TestGetSocketmask(t *testing.T) {
	tcases := []struct {
		name                    string
		socketmask              []SocketMask
		maskholder              []string
		count                   int
		expectedMaskHolder      []string
		expectedFinalSocketMask SocketMask
	}{
		{
			name:                    "Empty MaskHolder and count set as 0",
			socketmask:              []SocketMask{{1, 0}, {0, 1}, {1, 1}},
			maskholder:              []string{""},
			count:                   0,
			expectedMaskHolder:      []string{"10", "01", "11"},
			expectedFinalSocketMask: SocketMask{1, 0},
		},
		{
			name:                    "MaskHolder non zero, count set as 1",
			socketmask:              []SocketMask{{1, 0}, {0, 1}, {1, 1}},
			maskholder:              []string{"10", "01", "11"},
			count:                   1,
			expectedMaskHolder:      []string{"10", "01", "11"},
			expectedFinalSocketMask: SocketMask{1, 0},
		},

		{
			name:                    "Empty MaskHolder and count set as 0",
			socketmask:              []SocketMask{{0, 1}, {1, 1}},
			maskholder:              []string{""},
			count:                   0,
			expectedMaskHolder:      []string{"01", "11"},
			expectedFinalSocketMask: SocketMask{0, 1},
		},
		{
			name:                    "MaskHolder non zero, count set as 1",
			socketmask:              []SocketMask{{0, 1}, {1, 1}},
			maskholder:              []string{"01", "11"},
			count:                   1,
			expectedMaskHolder:      []string{"01", "11"},
			expectedFinalSocketMask: SocketMask{0, 1},
		},
	}

	for _, tc := range tcases {

		sm := NewSocketMask(nil)
		actualSocketMask, actualMaskHolder := sm.GetSocketMask(tc.socketmask, tc.maskholder, tc.count)
		if !reflect.DeepEqual(actualSocketMask, tc.expectedFinalSocketMask) {
			t.Errorf("Expected final socketmask to be %v, got %v", tc.expectedFinalSocketMask, actualSocketMask)
		}

		if !reflect.DeepEqual(actualMaskHolder, tc.expectedMaskHolder) {
			t.Errorf("Expected maskholder to be %v, got %v", tc.expectedMaskHolder, actualMaskHolder)
		}

	}
}

func TestNewSocketMask(t *testing.T) {
	tcases := []struct {
		name         string
		Mask         []int64
		expectedMask []int64
	}{
		{
			name:         "Mask as an int64 binary array",
			Mask:         []int64{1, 0},
			expectedMask: []int64{1, 0},
		},
	}

	for _, tc := range tcases {

		sm := NewSocketMask(nil)

		if reflect.DeepEqual(sm, tc.expectedMask) {
			t.Errorf("Expected socket mask to be %v, got %v", tc.expectedMask, sm)
		}
	}
}
