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

package rand

import (
	"math/rand"
	"strings"
	"testing"
)

const (
	maxRangeTestCount = 500
	testStringLength  = 32
)

func TestString(t *testing.T) {
	valid := "bcdfghjklmnpqrstvwxz2456789"
	for _, l := range []int{0, 1, 2, 10, 123} {
		s := String(l)
		if len(s) != l {
			t.Errorf("expected string of size %d, got %q", l, s)
		}
		for _, c := range s {
			if !strings.ContainsRune(valid, c) {
				t.Errorf("expected valid characters, got %v", c)
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
	r := rand.New(rand.NewSource(5))
	for i := 1; i < 20; i++ {
		actual := Perm(i)
		expected := r.Perm(i)
		for j := 0; j < i; j++ {
			if actual[j] != expected[j] {
				t.Errorf("Perm call result is unexpected")
			}
		}
	}
}

func TestIntnRange(t *testing.T) {
	// 0 is invalid.
	for min, max := range map[int]int{1: 2, 10: 123, 100: 500} {
		for i := 0; i < maxRangeTestCount; i++ {
			inrange := IntnRange(min, max)
			if inrange < min || inrange >= max {
				t.Errorf("%v out of range (%v,%v)", inrange, min, max)
			}
		}
	}
}

func TestInt63nRange(t *testing.T) {
	// 0 is invalid.
	for min, max := range map[int64]int64{1: 2, 10: 123, 100: 500} {
		for i := 0; i < maxRangeTestCount; i++ {
			inrange := Int63nRange(min, max)
			if inrange < min || inrange >= max {
				t.Errorf("%v out of range (%v,%v)", inrange, min, max)
			}
		}
	}
}

func BenchmarkRandomStringGeneration(b *testing.B) {
	b.ResetTimer()
	var s string
	for i := 0; i < b.N; i++ {
		s = String(testStringLength)
	}
	b.StopTimer()
	if len(s) == 0 {
		b.Fatal(s)
	}
}
