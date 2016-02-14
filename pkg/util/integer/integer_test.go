/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package integer

import "testing"

func TestIntMax(t *testing.T) {
	tests := []struct {
		nums        []int
		expectedMax int
	}{
		{
			nums:        []int{-1, 0},
			expectedMax: 0,
		},
		{
			nums:        []int{-1, -2},
			expectedMax: -1,
		},
		{
			nums:        []int{0, 1},
			expectedMax: 1,
		},
		{
			nums:        []int{1, 2},
			expectedMax: 2,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		if max := IntMax(test.nums[0], test.nums[1]); max != test.expectedMax {
			t.Errorf("expected %v,  got %v", test.expectedMax, max)
		}
	}
}

func TestIntMin(t *testing.T) {
	tests := []struct {
		nums        []int
		expectedMin int
	}{
		{
			nums:        []int{-1, 0},
			expectedMin: -1,
		},
		{
			nums:        []int{-1, -2},
			expectedMin: -2,
		},
		{
			nums:        []int{0, 1},
			expectedMin: 0,
		},
		{
			nums:        []int{1, 2},
			expectedMin: 1,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		if min := IntMin(test.nums[0], test.nums[1]); min != test.expectedMin {
			t.Errorf("expected %v,  got %v", test.expectedMin, min)
		}
	}
}

func TestInt64Max(t *testing.T) {
	tests := []struct {
		nums        []int64
		expectedMax int64
	}{
		{
			nums:        []int64{-1, 0},
			expectedMax: 0,
		},
		{
			nums:        []int64{-1, -2},
			expectedMax: -1,
		},
		{
			nums:        []int64{0, 1},
			expectedMax: 1,
		},
		{
			nums:        []int64{1, 2},
			expectedMax: 2,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		if max := Int64Max(test.nums[0], test.nums[1]); max != test.expectedMax {
			t.Errorf("expected %v,  got %v", test.expectedMax, max)
		}
	}
}

func TestInt64Min(t *testing.T) {
	tests := []struct {
		nums        []int64
		expectedMin int64
	}{
		{
			nums:        []int64{-1, 0},
			expectedMin: -1,
		},
		{
			nums:        []int64{-1, -2},
			expectedMin: -2,
		},
		{
			nums:        []int64{0, 1},
			expectedMin: 0,
		},
		{
			nums:        []int64{1, 2},
			expectedMin: 1,
		},
	}

	for i, test := range tests {
		t.Logf("executing scenario %d", i)
		if min := Int64Min(test.nums[0], test.nums[1]); min != test.expectedMin {
			t.Errorf("expected %v,  got %v", test.expectedMin, min)
		}
	}
}
