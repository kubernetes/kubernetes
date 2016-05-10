/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rand

import (
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestString(t *testing.T) {
	valid := "0123456789abcdefghijklmnopqrstuvwxyz"
	for _, l := range []int{0, 1, 2, 10, 123} {
		s := String(l)
		if len(s) != l {
			t.Errorf("expected string of size %d, got %q", l, s)
		}
		for _, c := range s {
			if !strings.ContainsRune(valid, c) {
				t.Errorf("expected valid charaters, got %v", c)
			}
		}
	}
}

// Confirm that panic occurs on invalid input.
func TestRangePanic(t *testing.T) {
	defer func() {
		if err := recover(); err == nil {
			t.Errorf("Panic didn't occur!")
		}
	}()
	// Should result in an error...
	Intn(0)
}

func TestIntn(t *testing.T) {
	// 0 is invalid.
	for _, max := range []int{1, 2, 10, 123} {
		inrange := Intn(max)
		if inrange < 0 || inrange > max {
			t.Errorf("%v out of range (0,%v)", inrange, max)
		}
	}
}

func TestPerm(t *testing.T) {
	Seed(5)
	rand.Seed(5)
	for i := 1; i < 20; i++ {
		actual := Perm(i)
		expected := rand.Perm(i)
		for j := 0; j < i; j++ {
			if actual[j] != expected[j] {
				t.Errorf("Perm call result is unexpected")
			}
		}
	}
}

func TestShuffle(t *testing.T) {
	Seed(5) // Arbitrary RNG seed for deterministic testing.
	have := []int{0, 1, 2, 3, 4}
	want := []int{3, 2, 4, 1, 0} // "have" shuffled, with RNG at Seed(5).
	got := append([]int{}, have...)
	Shuffle(sort.IntSlice(got))
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Shuffle(%v) => %v, want %v", have, got, want)
	}
}
